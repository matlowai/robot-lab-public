"""Live crowd: closed-loop pedestrians for the avoidance bake-off. One model, stepped next to the robot at 10 Hz,
used by the 2-D benchmark (sim2d), Isaac (tools/run_scenario_isaac.py) and, ported to torch, the RL env.

Why closed-loop: the first crowds (crowd.py) were keyframed against a nominal patrol timetable. Contenders that
walked faster than the timetable outran their encounters (30-40 s ahead by the last legs), so the scores measured
speed as much as avoidance. Here the people that matter react to where the robot actually is.

Three kinds of people, all "authorized" (nothing they do is an incident):
  wanderer   a book reader walking between random destinations in the yard; never looks at the robot
  loiterer   stands still (some on the patrol route), shuffles a metre or two now and then
  hunter     posted beside a patrol leg, milling about its post. When it SEES the robot (within trigger_m, inside
             its vision cone, line of sight not blocked by a building) it LOCKS ON and walks to intercept: it
             predicts the robot's motion (constant velocity, estimated from the robot's recent positions) and
             re-aims every reaim_s. Within commit_m it COMMITS: eyes back on the book, straight line, no more
             re-aiming, overshoot_m past the meeting point. Then it cools down and goes back to its post.

The commit rule is what makes a lock escapable: a pursuer that re-aims until contact at walking speed can't be
dodged, so there'd be nothing to learn. Evading means reading the collision course early (the robot only has its
lidar) and being somewhere else by the time the committed line arrives.

Deterministic: the same seed and the same robot trajectory give the same crowd. Different contenders drive the
robot differently, so their crowds diverge after the first reaction; that's the point.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from events.zones import Compound

PERSON_R = 0.30
DT = 0.1
WANDERER, LOITERER, HUNTER = 0, 1, 2
KIND_NAMES = ("wanderer", "loiterer", "hunter")
# per-person state
WALK, PAUSE, LOCKED, COMMITTED, COOLDOWN = range(5)
STATE_NAMES = ("walk", "pause", "locked", "committed", "cooldown")


@dataclass(frozen=True)
class CrowdConfig:
    n_wanderers: int = 30
    n_loiterers: int = 8
    n_hunters: int = 12
    walk_speed: tuple[float, float] = (0.8, 1.4)
    hunter_speed: tuple[float, float] = (1.0, 1.5)
    pause_s: tuple[float, float] = (0.0, 4.0)
    loiter_shift_s: tuple[float, float] = (15.0, 40.0)
    trigger_m: float = 12.0  # the robot must be this close for a lock ...
    vision_fov_deg: float = 140.0  # ... inside the vision cone while walking (a paused hunter looks all around) ...
    # ... with a clear line of sight (buildings block it)
    reaim_s: float = 0.5
    commit_m: tuple[float, float] = (2.5, 4.5)  # per hunter; re-aiming stops inside this distance
    overshoot_m: float = 4.0  # committed hunters walk this far past where the robot was when they committed
    lock_timeout_s: float = 20.0
    lose_sight_s: float = 3.0  # a lock is dropped after this long without seeing the robot
    cooldown_s: float = 25.0
    max_locks: int = 2  # per hunter per run
    post_offset_m: tuple[float, float] = (4.0, 9.0)  # hunter posts: this far to the side of a patrol leg
    post_radius_m: float = 4.0  # hunters mill about within this of their post
    site_margin_m: float = 2.0
    building_margin_m: float = 1.0
    predict_cap_s: float = 6.0  # intercept prediction horizon

    @classmethod
    def from_dict(cls, d: dict | None) -> "CrowdConfig":
        d = dict(d or {})
        for k, v in list(d.items()):
            if isinstance(v, list):
                d[k] = tuple(v)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def patrol_legs(compound: Compound) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    stops = [compound.charging_station] + [p for _, p in compound.route] + [compound.charging_station]
    return list(zip(stops, stops[1:]))


def building_segments(compound: Compound) -> np.ndarray:
    segs = []
    for b in compound.spec.get("buildings", []):
        x0, y0, x1, y1 = b["rect"]
        segs += [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]
    return np.array(segs, dtype=np.float64).reshape(-1, 4)


def segment_blocked(p, q, segs: np.ndarray) -> bool:
    """True when the segment p->q crosses any of segs (proper or touching intersection)."""
    if len(segs) == 0:
        return False
    ax, ay, bx, by = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    rx, ry = q[0] - p[0], q[1] - p[1]
    sx, sy = bx - ax, by - ay
    den = rx * sy - ry * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((ax - p[0]) * sy - (ay - p[1]) * sx) / den
        u = ((ax - p[0]) * ry - (ay - p[1]) * rx) / den
    return bool(np.any((np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)))


def intercept_point(hunter, speed: float, robot, robot_vel, cap_s: float) -> tuple[float, float]:
    """Where to walk to meet a robot moving at constant velocity; the robot itself if it can't be caught in cap_s."""
    px, py = robot[0] - hunter[0], robot[1] - hunter[1]
    vx, vy = robot_vel
    a = vx * vx + vy * vy - speed * speed
    b = 2 * (px * vx + py * vy)
    c = px * px + py * py
    tau = None
    if abs(a) < 1e-9:
        if b < 0:
            tau = -c / b
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            r = math.sqrt(disc)
            roots = sorted(x for x in ((-b - r) / (2 * a), (-b + r) / (2 * a)) if x > 0)
            tau = roots[0] if roots else None
    if tau is None:
        tau = 0.0
    tau = min(tau, cap_s)
    return robot[0] + vx * tau, robot[1] + vy * tau


@dataclass
class LockRecord:
    hunter: str
    t_lock: float
    t_commit: float | None = None
    t_end: float | None = None
    end_reason: str | None = None  # passed | timeout | lost_sight | run_end
    min_clearance_m: float = math.inf  # robot-person clearance while locked or committed
    commit_dist_m: float | None = None

    @property
    def hit(self) -> bool:
        return self.min_clearance_m < 0


@dataclass
class LiveCrowd:
    compound: Compound
    cfg: CrowdConfig
    seed: int
    ids: list[str] = field(default_factory=list)
    kind: np.ndarray = None
    pos: np.ndarray = None
    heading: np.ndarray = None
    speed: np.ndarray = None
    state: np.ndarray = None
    target: np.ndarray = None
    until: np.ndarray = None  # pause end / next loiter shift / cooldown end
    post: np.ndarray = None
    commit_m: np.ndarray = None
    locks_used: np.ndarray = None
    lock_t: np.ndarray = None
    last_aim_t: np.ndarray = None
    last_seen_t: np.ndarray = None
    travel_left: np.ndarray = None
    t: float = 0.0
    locks: list[LockRecord] = field(default_factory=list)

    # --- construction ---------------------------------------------------------------------------------------------
    @classmethod
    def create(cls, compound: Compound, cfg: CrowdConfig, seed: int) -> "LiveCrowd":
        c = cls(compound, cfg, seed)
        c.rng = np.random.default_rng(seed)
        c.bsegs = building_segments(compound)
        c._robot_hist: list[tuple[float, float, float]] = []
        c._open_lock: dict[int, LockRecord] = {}
        n = cfg.n_wanderers + cfg.n_loiterers + cfg.n_hunters
        c.ids = [f"person_{i + 1:04d}" for i in range(n)]
        c.kind = np.array([WANDERER] * cfg.n_wanderers + [LOITERER] * cfg.n_loiterers + [HUNTER] * cfg.n_hunters)
        c.pos, c.target, c.post = np.zeros((n, 2)), np.zeros((n, 2)), np.full((n, 2), np.nan)
        c.heading, c.speed = np.zeros(n), np.zeros(n)
        c.state, c.until = np.full(n, WALK), np.zeros(n)
        c.commit_m, c.locks_used = np.zeros(n), np.zeros(n, dtype=int)
        c.lock_t, c.last_aim_t, c.last_seen_t = np.full(n, -1e9), np.full(n, -1e9), np.full(n, -1e9)
        c.travel_left = np.zeros(n)
        legs = patrol_legs(compound)
        robot0 = compound.charging_station
        for i in range(n):
            k = c.kind[i]
            if k == WANDERER:
                c.pos[i] = c._free_point(avoid=robot0)
                c.speed[i] = c.rng.uniform(*cfg.walk_speed)
                c._new_destination(i)
            elif k == LOITERER:
                if i % 2 == 0:  # every other loiterer stands on the patrol route itself (where people may stand)
                    for _ in range(100):
                        a, b = legs[c.rng.integers(len(legs))]
                        spot = _lerp(a, b, c.rng.uniform(0.3, 0.7))
                        if c._usable(spot) and math.dist(spot, robot0) > 6.0:
                            break
                    c.pos[i] = c._free_point(near=spot, radius=1.0, avoid=robot0)
                else:
                    c.pos[i] = c._free_point(avoid=robot0)
                c.state[i], c.until[i] = PAUSE, c.rng.uniform(*cfg.loiter_shift_s)
                c.target[i] = c.pos[i]
            else:
                h = i - cfg.n_wanderers - cfg.n_loiterers
                a, b = legs[h % len(legs)]  # round-robin over the legs: every leg gets its hunters
                c.post[i] = c._post_beside(a, b)
                c.pos[i] = c._free_point(near=c.post[i], radius=cfg.post_radius_m, avoid=robot0)
                c.speed[i] = c.rng.uniform(*cfg.hunter_speed)
                c.commit_m[i] = c.rng.uniform(*cfg.commit_m)
                c._new_destination(i)
        return c

    def _usable(self, p) -> bool:
        cfg, m = self.cfg, self.cfg.site_margin_m
        w, h = self.compound.spec["size_m"]
        if not (m <= p[0] <= w - m and m <= p[1] <= h - m):
            return False
        for b in self.compound.spec.get("buildings", []):
            x0, y0, x1, y1 = b["rect"]
            bm = cfg.building_margin_m
            if x0 - bm <= p[0] <= x1 + bm and y0 - bm <= p[1] <= y1 + bm:
                return False
        z = self.compound.zone_at(p)
        return not (z and z.restricted)

    def _path_ok(self, p, q) -> bool:
        n = max(2, int(math.dist(p, q) / 1.0))
        return all(self._usable(_lerp(p, q, k / n)) for k in range(n + 1)) and not segment_blocked(p, q, self.bsegs)

    def _free_point(self, near=None, radius=None, avoid=None, tries=500):
        w, h = self.compound.spec["size_m"]
        for _ in range(tries):
            if near is None:
                p = (self.rng.uniform(0, w), self.rng.uniform(0, h))
            else:
                r, a = radius * math.sqrt(self.rng.uniform()), self.rng.uniform(-math.pi, math.pi)
                p = (near[0] + r * math.cos(a), near[1] + r * math.sin(a))
            if self._usable(p) and (avoid is None or math.dist(p, avoid) > 6.0):
                return np.array(p)
        raise RuntimeError(f"no free point near {near}")

    def _post_beside(self, a, b):
        L = math.dist(a, b)
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        for _ in range(200):
            s = self.rng.uniform(0.25, 0.75) * L
            off = self.rng.choice((-1, 1)) * self.rng.uniform(*self.cfg.post_offset_m)
            p = (a[0] + ux * s - uy * off, a[1] + uy * s + ux * off)
            if self._usable(p) and math.dist(p, self.compound.charging_station) > 6.0:
                return np.array(p)
        raise RuntimeError(f"no hunter post beside leg {a}->{b}")

    def _new_destination(self, i: int) -> None:
        near = None if self.kind[i] == WANDERER else self.post[i]
        radius = None if near is None else self.cfg.post_radius_m
        for _ in range(50):
            q = self._free_point(near=near, radius=radius)
            if math.dist(q, self.pos[i]) > 2.0 and self._path_ok(self.pos[i], q):
                break
        else:  # nowhere reachable: stay put (arrive at once, then pause) rather than walk an unchecked path
            q = self.pos[i].copy()
        self.target[i] = q
        self.heading[i] = math.atan2(q[1] - self.pos[i][1], q[0] - self.pos[i][0])
        self.state[i] = WALK

    # --- perception -----------------------------------------------------------------------------------------------
    def _robot_velocity(self) -> tuple[float, float]:
        """World-frame robot velocity over the last ~0.5 s of positions: all a hunter can see."""
        h = self._robot_hist
        if len(h) < 2:
            return 0.0, 0.0
        t1, x1, y1 = h[-1]
        t0, x0, y0 = next((e for e in h if t1 - e[0] <= 0.5 + 1e-9), h[0])
        dt = t1 - t0
        return ((x1 - x0) / dt, (y1 - y0) / dt) if dt > 1e-9 else (0.0, 0.0)

    def sees(self, i: int, robot) -> bool:
        d = math.dist(self.pos[i], robot)
        if d > self.cfg.trigger_m:
            return False
        if self.state[i] == WALK:  # walking hunters look where they're going; paused ones look around
            bearing = math.atan2(robot[1] - self.pos[i][1], robot[0] - self.pos[i][0])
            off = abs((bearing - self.heading[i] + math.pi) % (2 * math.pi) - math.pi)
            if off > math.radians(self.cfg.vision_fov_deg) / 2:
                return False
        return not segment_blocked(self.pos[i], robot, self.bsegs)

    # --- stepping -------------------------------------------------------------------------------------------------
    def step(self, t: float, robot_xy, robot_r: float = 0.55) -> None:
        """Advance everyone to time t (in DT steps) given the robot's position at t. robot_r is only for scoring."""
        while self.t + 1e-9 < t:
            self._step_once(robot_xy, robot_r)

    def _step_once(self, robot, robot_r: float) -> None:
        cfg, dt = self.cfg, DT
        self.t = round(self.t + dt, 6)
        t = self.t
        robot = (float(robot[0]), float(robot[1]))
        self._robot_hist.append((t, *robot))
        if len(self._robot_hist) > 20:
            self._robot_hist.pop(0)
        rvel = self._robot_velocity()
        self._robot_now, self._robot_r = robot, robot_r
        for i in range(len(self.ids)):
            k, s = self.kind[i], self.state[i]
            if k == LOITERER:
                if t >= self.until[i]:
                    q = self._free_point(near=self.pos[i], radius=2.0)
                    self.pos[i] = q if self._path_ok(self.pos[i], q) else self.pos[i]
                    self.until[i] = t + self.rng.uniform(*cfg.loiter_shift_s)
                continue
            if k == HUNTER and s in (WALK, PAUSE, COOLDOWN):
                if s == COOLDOWN and t >= self.until[i]:
                    self._new_destination(i)
                    s = self.state[i]
                if s != COOLDOWN and self.locks_used[i] < cfg.max_locks and self.sees(i, robot):
                    self.state[i], s = LOCKED, LOCKED
                    self.locks_used[i] += 1
                    self.lock_t[i], self.last_seen_t[i], self.last_aim_t[i] = t, t, -1e9
                    self._open_lock[i] = LockRecord(self.ids[i], t)
                    self.locks.append(self._open_lock[i])
            if s == LOCKED:
                if self.sees(i, robot) or math.dist(self.pos[i], robot) <= cfg.trigger_m and not segment_blocked(
                        self.pos[i], robot, self.bsegs):
                    self.last_seen_t[i] = t  # a locked hunter keeps its eyes on the robot: no cone
                d = math.dist(self.pos[i], robot)
                if t - self.lock_t[i] > cfg.lock_timeout_s or t - self.last_seen_t[i] > cfg.lose_sight_s:
                    self._release(i, "timeout" if t - self.lock_t[i] > cfg.lock_timeout_s else "lost_sight")
                    continue
                commit = d <= self.commit_m[i]
                if commit or t - self.last_aim_t[i] >= cfg.reaim_s - 1e-9:  # the commit line is always a fresh aim
                    aim = intercept_point(self.pos[i], self.speed[i], robot, rvel, cfg.predict_cap_s)
                    self.heading[i] = math.atan2(aim[1] - self.pos[i][1], aim[0] - self.pos[i][0])
                    self.last_aim_t[i] = t
                self._advance(i, self.speed[i] * dt)
                if commit:
                    self.state[i] = COMMITTED
                    self.travel_left[i] = d + cfg.overshoot_m - self.speed[i] * dt
                    rec = self._open_lock[i]
                    rec.t_commit, rec.commit_dist_m = t, round(d, 3)
            elif s == COMMITTED:
                step = self.speed[i] * dt
                self._advance(i, step)
                self.travel_left[i] -= step
                if self.travel_left[i] <= 0:
                    self._release(i, "passed")
            elif s == PAUSE:
                if t >= self.until[i]:
                    self._new_destination(i)
            elif s == WALK:
                to = self.target[i] - self.pos[i]
                d = float(np.hypot(*to))
                step = self.speed[i] * dt
                if d <= step:
                    self.pos[i] = self.target[i].copy()
                    self.state[i], self.until[i] = PAUSE, t + self.rng.uniform(*cfg.pause_s)
                else:
                    self.pos[i] = self.pos[i] + to / d * step
            elif s == COOLDOWN:  # walk back toward the post while cooling down
                to = self.post[i] - self.pos[i]
                d = float(np.hypot(*to))
                if d > 0.5:
                    self.heading[i] = math.atan2(to[1], to[0])
                    self._advance(i, min(d, self.speed[i] * 0.7 * dt))
        for i, rec in self._open_lock.items():
            rec.min_clearance_m = min(rec.min_clearance_m, math.dist(self.pos[i], robot) - PERSON_R - robot_r)

    def _advance(self, i: int, step: float) -> None:
        """Move along the current heading unless that would walk into a building or off the site: then stop there."""
        q = self.pos[i] + step * np.array([math.cos(self.heading[i]), math.sin(self.heading[i])])
        w, h = self.compound.spec["size_m"]
        inside = 0.5 <= q[0] <= w - 0.5 and 0.5 <= q[1] <= h - 0.5
        if inside and not segment_blocked(self.pos[i], q, self.bsegs):
            self.pos[i] = q

    def _release(self, i: int, reason: str) -> None:
        rec = self._open_lock.pop(i)
        rec.min_clearance_m = min(rec.min_clearance_m, math.dist(self.pos[i], self._robot_now) - PERSON_R - self._robot_r)
        rec.t_end, rec.end_reason = self.t, reason
        self.state[i], self.until[i] = COOLDOWN, self.t + self.cfg.cooldown_s

    def finish(self) -> None:
        for i in list(self._open_lock):
            self._release(i, "run_end")

    # --- views ----------------------------------------------------------------------------------------------------
    def positions(self) -> np.ndarray:
        return self.pos.copy()

    def locked_on(self) -> list[str]:
        return [self.ids[i] for i in self._open_lock]

    def lock_summary(self) -> dict:
        done = [r for r in self.locks]
        return {"locks": len(done), "committed": sum(1 for r in done if r.t_commit is not None),
                "hit": sum(1 for r in done if r.hit), "evaded": sum(1 for r in done if r.t_commit is not None and not r.hit),
                "records": [asdict(r) | {"hit": r.hit} for r in done]}


def _lerp(a, b, f):
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)


def scenario_crowd(scenario) -> LiveCrowd | None:
    """The live crowd a scenario asks for (its `crowd:` block), or None for keyframed-only scenarios."""
    block = scenario.raw.get("crowd")
    if not block:
        return None
    return LiveCrowd.create(Compound.load(scenario.compound), CrowdConfig.from_dict(block.get("config")),
                            int(block["seed"]))
