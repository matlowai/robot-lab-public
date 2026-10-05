"""FROZEN copy of robots/spot/local_planner.py at commit d017268 (baseline heuristic_v1). Do not edit."""

"""Reactive local planner: avoid whatever the lidar sees, while heading for the next checkpoint.

A dynamic-window-style search. Every cycle it rolls candidate (vx, wz) commands forward over a short horizon
against the latest scan (points in the robot frame) and picks the best mix of progress, heading and clearance.
If no forward motion is safe it turns toward open space, and if even that isn't possible it stops and waits.
It knows nothing about people, scenarios or the map: an obstacle is any lidar return. ObstacleTracker turns
consecutive scans into per-point velocities (clusters matched between scans), so a trajectory is checked
against where each obstacle WILL be, not where it is: someone walking straight at the robot gets dodged.
Pure Python, so it's testable without a simulator. Nav2 (M2) is the heavier successor.
"""

import math
from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True)
class PlannerConfig:
    robot_radius: float = 0.55  # Spot's footprint as a circle (1.1 m long)
    safety_margin: float = 0.35  # extra clearance a trajectory must keep
    horizon_s: float = 2.0
    dt: float = 0.1
    vx_samples: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    wz_samples: int = 11  # spread evenly over [-max_wz, max_wz]
    max_wz: float = 1.0
    arrival_m: float = 0.4
    w_progress: float = 1.0
    w_heading: float = 0.5
    w_clearance: float = 0.8
    clearance_cap: float = 2.0  # clearance beyond this earns nothing extra
    w_speed: float = 0.15


@dataclass(frozen=True)
class Plan:
    cmd: tuple[float, float, float]  # (vx, vy, wz), body frame
    status: str  # arrived | moving | turning | blocked
    clearance: float  # predicted minimum clearance of the chosen command (m, inf if no obstacles)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _to_robot(pose: tuple[float, float, float], p: Point) -> Point:
    x, y, yaw = pose
    dx, dy = p[0] - x, p[1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * dx + s * dy, -s * dx + c * dy)


def _rollout(vx: float, wz: float, cfg: PlannerConfig):
    x = y = th = 0.0
    for _ in range(int(round(cfg.horizon_s / cfg.dt))):
        th += wz * cfg.dt
        x += vx * math.cos(th) * cfg.dt
        y += vx * math.sin(th) * cfg.dt
        yield x, y, th


def _min_clearance(path, points: list[Point], vels: list[Point], radius: float, dt: float) -> float:
    """Closest the rollout comes to any obstacle point, each point moving at its estimated velocity."""
    best = math.inf
    for k, (x, y, _) in enumerate(path):
        t = (k + 1) * dt
        for (px, py), (vx, vy) in zip(points, vels):
            d = math.hypot(px + vx * t - x, py + vy * t - y) - radius
            if d < best:
                best = d
    return best


class ObstacleTracker:
    """Scan-to-scan velocity estimates. Clusters consecutive scan points, matches cluster centroids to the previous
    scan in the world frame, smooths the velocity, and hands back a velocity per point in the robot frame.
    Clusters wider than `static_extent` (walls, buildings) are treated as static."""

    def __init__(self, cluster_gap=0.5, gap_per_m=0.06, match_dist=1.5, alpha=0.5, max_speed=3.0,
                 static_extent=2.0, min_speed=0.25, min_matches=2):
        self.cluster_gap, self.gap_per_m, self.match_dist, self.alpha = cluster_gap, gap_per_m, match_dist, alpha
        self.max_speed, self.static_extent = max_speed, static_extent
        self.min_speed, self.min_matches = min_speed, min_matches
        self._tracks: list[tuple[Point, Point, int]] = []  # (world centroid, world velocity, consecutive matches)
        self._t: float | None = None

    def update(self, t: float, pose: tuple[float, float, float], scan: list[Point]) -> list[Point]:
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        world = [(x + c * px - s * py, y + s * px + c * py) for px, py in scan]
        clusters: list[list[int]] = []
        for i, w in enumerate(world):
            gap = self.cluster_gap + self.gap_per_m * math.hypot(*scan[i])  # far points are sparser
            if clusters and math.dist(world[clusters[-1][-1]], w) <= gap:
                clusters[-1].append(i)
            else:
                clusters.append([i])
        dt = (t - self._t) if self._t is not None else None
        tracks, vel_of_point = [], [(0.0, 0.0)] * len(scan)
        for members in clusters:
            pts = [world[i] for i in members]
            cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
            extent = math.dist(pts[0], pts[-1])
            v, matches, trusted = (0.0, 0.0), 0, (0.0, 0.0)
            if dt and len(pts) >= 2 and extent <= self.static_extent and self._tracks:  # lone grazing hits slide
                (px, py), (pvx, pvy), pm = min(self._tracks, key=lambda tr: math.dist(tr[0], (cx, cy)))
                if math.dist((px, py), (cx, cy)) <= self.match_dist:
                    raw = ((cx - px) / dt, (cy - py) / dt)
                    v = (self.alpha * raw[0] + (1 - self.alpha) * pvx, self.alpha * raw[1] + (1 - self.alpha) * pvy)
                    speed = math.hypot(*v)
                    if speed > self.max_speed:
                        v = (v[0] * self.max_speed / speed, v[1] * self.max_speed / speed)
                    matches = pm + 1
                    if matches >= self.min_matches and math.hypot(*v) >= self.min_speed:
                        trusted = v
            tracks.append(((cx, cy), v, matches))
            v = trusted
            robot_v = (c * v[0] + s * v[1], -s * v[0] + c * v[1])  # world -> robot frame (rotation only)
            for i in members:
                vel_of_point[i] = robot_v
        self._tracks, self._t = tracks, t
        return vel_of_point


def plan(pose: tuple[float, float, float], goal: Point, scan: list[Point], cfg: PlannerConfig = PlannerConfig(),
         velocities: list[Point] | None = None) -> Plan:
    """pose: (x, y, yaw) site frame. goal: site frame. scan: obstacle points in the robot frame (x forward).
    velocities: per-point velocity in the robot frame (from ObstacleTracker); None = everything static."""
    gx, gy = _to_robot(pose, goal)
    dist0 = math.hypot(gx, gy)
    if dist0 <= cfg.arrival_m:
        return Plan((0.0, 0.0, 0.0), "arrived", math.inf)
    reach = max(cfg.vx_samples) * cfg.horizon_s + cfg.robot_radius + cfg.clearance_cap
    vels = velocities if velocities is not None else [(0.0, 0.0)] * len(scan)
    keep = [i for i, p in enumerate(scan) if math.hypot(*p) <= reach + math.hypot(*vels[i]) * cfg.horizon_s]
    near, near_v = [scan[i] for i in keep], [vels[i] for i in keep]
    here = min((math.hypot(*p) for p in near), default=math.inf) - cfg.robot_radius

    best, best_score = None, -math.inf
    for vx in cfg.vx_samples:
        for i in range(cfg.wz_samples):
            wz = -cfg.max_wz + 2 * cfg.max_wz * i / (cfg.wz_samples - 1)
            if vx == 0.0 and wz == 0.0:
                continue
            path = list(_rollout(vx, wz, cfg))
            clear = _min_clearance(path, near, near_v, cfg.robot_radius, cfg.dt)
            if clear < cfg.safety_margin:
                continue
            # progress = closest approach along the rollout, so a short final hop isn't punished for overshooting
            closest = min(range(len(path)), key=lambda k: math.hypot(gx - path[k][0], gy - path[k][1]))
            ex, ey, eth = path[closest]
            progress = dist0 - math.hypot(gx - ex, gy - ey)
            gap = math.hypot(gx - ex, gy - ey)
            heading = 1.0 if gap <= cfg.arrival_m else math.cos(_wrap(math.atan2(gy - ey, gx - ex) - eth))
            score = (cfg.w_progress * progress + cfg.w_heading * heading
                     + cfg.w_clearance * min(clear, cfg.clearance_cap) + cfg.w_speed * vx)
            if score > best_score:
                best, best_score = (vx, wz, clear), score
    if best is None:  # nothing keeps the margin: never just freeze, take whatever keeps the most clearance
        fallback = max(((vx, -cfg.max_wz + 2 * cfg.max_wz * i / (cfg.wz_samples - 1)) for vx in cfg.vx_samples
                        for i in range(cfg.wz_samples)),
                       key=lambda c: _min_clearance(list(_rollout(c[0], c[1], cfg)), near, near_v, cfg.robot_radius, cfg.dt))
        clear = _min_clearance(list(_rollout(*fallback, cfg)), near, near_v, cfg.robot_radius, cfg.dt)
        return Plan((fallback[0], 0.0, fallback[1]), "blocked", clear)
    vx, wz, clear = best
    # slow down on the final approach so the policy doesn't overshoot the checkpoint
    vx = min(vx, max(0.25, dist0))
    return Plan((vx, 0.0, wz), "moving" if vx > 0 else "turning", clear)


def _corridor_clear(angle: float, length: float, half_width: float, points: list[Point], vels: list[Point],
                    speed: float, horizon_s: float = 6.0, dt: float = 0.25) -> bool:
    """Swept check: if the robot drove straight along `angle` at `speed` (stopping at `length`), would any obstacle
    point, moving at its estimated velocity, come within `half_width` of it at the same moment?"""
    ca, sa = math.cos(angle), math.sin(angle)
    steps = int(horizon_s / dt) + 1
    for (px, py), (vx, vy) in zip(points, vels):
        for k in range(steps):
            t = k * dt
            s = min(speed * t, length)
            if math.hypot(px + vx * t - s * ca, py + vy * t - s * sa) < half_width:
                return False
    return True


class LocalPlanner:
    """Stateful wrapper: tracks obstacle motion across scans and commits to a side when it has to go around
    something, so it doesn't dither between two equally good detours."""

    SWEEP_DEG = (15, 30, 45, 60, 75, 90)

    def __init__(self, cfg: PlannerConfig = PlannerConfig(), lookahead_m: float = 4.0, commit_s: float = 3.0):
        self.cfg, self.lookahead_m, self.commit_s = cfg, lookahead_m, commit_s
        self.tracker = ObstacleTracker()
        self._side, self._side_until = 0, -math.inf
        self.subgoal: Point | None = None

    def step(self, t: float, pose: tuple[float, float, float], goal: Point, scan: list[Point]) -> Plan:
        vels = self.tracker.update(t, pose, scan)
        gx, gy = _to_robot(pose, goal)
        dist = math.hypot(gx, gy)
        heading = math.atan2(gy, gx)
        length = min(dist, self.lookahead_m)
        half = self.cfg.robot_radius + self.cfg.safety_margin
        speed = max(self.cfg.vx_samples)
        target = goal
        if dist > self.cfg.arrival_m and not _corridor_clear(heading, length, half, scan, vels, speed):
            committed = t < self._side_until and self._side
            if committed:  # exhaust the side we already committed to before even considering the other one
                order = [(self._side, d) for d in self.SWEEP_DEG] + [(-self._side, d) for d in self.SWEEP_DEG]
            else:  # fresh decision: smallest deviation wins, left before right on ties
                order = [(side, d) for d in self.SWEEP_DEG for side in (1, -1)]
            chosen = None
            for side, deg in order:
                a = heading + side * math.radians(deg)
                if _corridor_clear(a, self.lookahead_m, half, scan, vels, speed):
                    chosen = (a, side)
                    break
            if chosen:
                a, side = chosen
                self._side, self._side_until = side, t + self.commit_s
                x, y, yaw = pose
                target = (x + self.lookahead_m * math.cos(yaw + a), y + self.lookahead_m * math.sin(yaw + a))
        self.subgoal = None if target == goal else target
        p = plan(pose, target, scan, self.cfg, vels)
        if p.status == "arrived" and target != goal:  # reached a detour point, not the checkpoint
            p = Plan((0.0, 0.0, 0.0), "moving", p.clearance)
        return p
