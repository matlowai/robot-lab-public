"""Fast 2-D patrol benchmark for avoidance contenders: the whole compound patrol in seconds, many seeds in parallel.

The world is the compound's walls (fence, buildings) plus the scenario's people as moving circles, a numpy
ray-cast lidar with the same geometry as robots/spot/lidar.py, and a unicycle robot with acceleration limits
(so it can't turn on a sixpence the way an ideal point would). It is a screening tool: every claim still has to
survive Isaac (tools/run_scenario_isaac.py), where the legs are a learned policy and contacts have physics.

    uv run python -m benchmarks.avoidance.sim2d --seeds 1-200 --controllers control,heuristic --workers 64

Controllers take (t, pose, goal, scan) and return (vx, vy, wz, status): body-frame forward, sideways (+left) and
turn rates, as Spot's walking policy accepts them. Scan = robot-frame points, exactly as in Isaac.
"""

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.avoidance.crowd import generate, generate_live, parse_seeds
from benchmarks.avoidance.pedestrians import scenario_crowd
from events.zones import Compound
from robots.spot.controller import command as straight_line
from robots.spot.local_planner import LocalPlanner
from scenarios.model import Scenario
from scenarios.oracle import _actor_at

REPO = Path(__file__).resolve().parents[2]
PERSON_R, SPOT_R, NEAR_MISS_M = 0.30, 0.55, 0.30
DT, ARRIVAL_M = 0.1, 0.4
ACC_VX, ACC_VY, ACC_WZ = 1.5, 1.5, 3.0  # m/s^2, m/s^2, rad/s^2
STUCK_S = 45.0


# --- controllers -----------------------------------------------------------------------------------------------

class Control:
    """Negative control: straight at the checkpoint, no avoidance."""
    name = "control"

    def step(self, t, pose, goal, scan):
        (vx, vy, wz), arrived = straight_line(pose, goal)
        return vx, vy, wz, "arrived" if arrived else "moving"


class Heuristic:
    name = "heuristic"

    def __init__(self):
        self.planner = LocalPlanner()

    def step(self, t, pose, goal, scan):
        p = self.planner.step(t, pose, goal, scan)
        return p.cmd[0], p.cmd[1], p.cmd[2], p.status


class HeuristicV1:
    """The planner exactly as it passed the book readers in Isaac (commit d017268): no sidestep, soft commitment.
    Kept as a named baseline so every report shows what the fixes bought."""
    name = "heuristic_v1"

    def __init__(self):
        from benchmarks.avoidance import _planner_v1
        self.planner = _planner_v1.LocalPlanner()

    def step(self, t, pose, goal, scan):
        p = self.planner.step(t, pose, goal, scan)
        return p.cmd[0], p.cmd[1], p.cmd[2], p.status


CONTROLLERS = {"control": Control, "heuristic": Heuristic, "heuristic_v1": HeuristicV1}


def register(name: str, factory) -> None:
    CONTROLLERS[name] = factory


# --- world -----------------------------------------------------------------------------------------------------

def wall_segments(compound: Compound) -> np.ndarray:
    segs = []
    b = compound.boundary
    for i in range(len(b)):
        segs.append((*b[i], *b[(i + 1) % len(b)]))  # gates closed: the whole perimeter is a wall
    for bld in compound.spec.get("buildings", []):
        x0, y0, x1, y1 = bld["rect"]
        segs += [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]
    return np.array(segs, dtype=np.float64)


def lidar(pose, segs: np.ndarray, people: np.ndarray, rays=181, fov=math.radians(270), max_range=12.0, start=0.65):
    x, y, yaw = pose
    rel = np.linspace(-fov / 2, fov / 2, rays)
    ang = yaw + rel
    dx, dy = np.cos(ang), np.sin(ang)
    best = np.full(rays, max_range)
    if len(segs):
        ax, ay, bx, by = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
        ex, ey = bx - ax, by - ay
        den = dx[:, None] * ey[None, :] - dy[:, None] * ex[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((ax - x)[None, :] * ey[None, :] - (ay - y)[None, :] * ex[None, :]) / den
            u = ((ax - x)[None, :] * dy[:, None] - (ay - y)[None, :] * dx[:, None]) / den
        ok = (np.abs(den) > 1e-12) & (t > start) & (u >= 0) & (u <= 1)
        best = np.minimum(best, np.where(ok, t, np.inf).min(axis=1))
    if len(people):
        fx, fy = x - people[:, 0], y - people[:, 1]
        b = dx[:, None] * fx[None, :] + dy[:, None] * fy[None, :]
        c = (fx * fx + fy * fy - PERSON_R ** 2)[None, :]
        disc = b * b - c
        with np.errstate(invalid="ignore"):
            t = -b - np.sqrt(disc)
        ok = (disc >= 0) & (t > start)
        best = np.minimum(best, np.where(ok, t, np.inf).min(axis=1))
    hit = best < max_range
    return list(zip((best * np.cos(rel))[hit].tolist(), (best * np.sin(rel))[hit].tolist()))


def wall_clearance(p, segs: np.ndarray) -> float:
    ax, ay, bx, by = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    ex, ey = bx - ax, by - ay
    t = np.clip(((p[0] - ax) * ex + (p[1] - ay) * ey) / np.maximum(ex * ex + ey * ey, 1e-12), 0, 1)
    return float(np.min(np.hypot(p[0] - (ax + t * ex), p[1] - (ay + t * ey)))) - SPOT_R


# --- episode ---------------------------------------------------------------------------------------------------

def run_episode(seed: int, controller: str, max_s: float = 700.0, scenario_path: str | None = None,
                live: str | None = None, trace_path: str | None = None) -> dict:
    """One whole patrol. trace_path (opt-in, default off): also write a per-step trace there (``Trace``, .npz) for
    contact forensics. Tracing only reads state, so the returned row is identical with or without it."""
    compound = Compound.load(REPO / "sim/compound/compound_spec.yaml")
    if scenario_path:
        scenario = Scenario.load(scenario_path)
    else:
        doc = generate_live(seed, live) if live else generate(seed)
        tmp = REPO / "data/scenarios/crowd" / f"{doc['id']}.yaml"
        if not tmp.exists():
            from benchmarks.avoidance.crowd import to_scenario_file
            to_scenario_file(doc, tmp.parent)
        scenario = Scenario.load(tmp)
    segs = wall_segments(compound)
    people_actors = [a for a in scenario.actors if a.cls == "person"]
    crowd = scenario_crowd(scenario)
    crowd_ids = crowd.ids if crowd else []
    ctrl = CONTROLLERS[controller]()
    stops = [pos for _, pos in compound.route] + [compound.charging_station]
    x, y = compound.charging_station
    yaw, vx, vy, wz = 0.0, 0.0, 0.0, 0.0
    goal_i, last_progress_t, best_dist = 0, 0.0, math.inf
    min_clear = {a.id: math.inf for a in people_actors} | {pid: math.inf for pid in crowd_ids}
    min_wall, wall_contacts, plan_s, n_plans, status_counts = math.inf, 0, 0.0, 0, {}
    t, done, stuck = 0.0, False, False
    tr = Trace(people_actors, crowd) if trace_path else None
    while t < max_s:
        people = []
        for a in people_actors:
            at = _actor_at(a, t)
            if at is not None:
                (px, py), _ = at
                people.append((px, py))
                min_clear[a.id] = min(min_clear[a.id], math.hypot(px - x, py - y) - PERSON_R - SPOT_R)
        if crowd:
            crowd.step(t, (x, y), SPOT_R)
            cp = crowd.pos
            dd = np.hypot(cp[:, 0] - x, cp[:, 1] - y) - PERSON_R - SPOT_R
            for pid, d in zip(crowd_ids, dd):
                if d < min_clear[pid]:
                    min_clear[pid] = float(d)
            people += [tuple(p) for p in cp]
        wc = wall_clearance((x, y), segs)
        min_wall = min(min_wall, wc)
        wall_contacts += wc < 0
        goal = stops[goal_i]
        scan = lidar((x, y, yaw), segs, np.array(people) if people else np.zeros((0, 2)))
        t0 = time.perf_counter()
        cvx, cvy, cwz, status = ctrl.step(t, (x, y, yaw), goal, scan)
        plan_s += time.perf_counter() - t0
        n_plans += 1
        status_counts[status] = status_counts.get(status, 0) + 1
        if tr is not None:
            tr.record(t, (x, y, yaw), (vx, vy, wz), (cvx, cvy, cwz), status, goal_i, people_actors, crowd, ctrl)
        if status == "arrived" or math.dist((x, y), goal) <= ARRIVAL_M:
            goal_i += 1
            best_dist, last_progress_t = math.inf, t
            if goal_i == len(stops):
                done = True
                break
            continue
        d = math.dist((x, y), goal)
        if d < best_dist - 0.5:
            best_dist, last_progress_t = d, t
        elif t - last_progress_t > STUCK_S:
            stuck = True
            break
        vx += max(-ACC_VX * DT, min(ACC_VX * DT, cvx - vx))
        vy += max(-ACC_VY * DT, min(ACC_VY * DT, cvy - vy))
        wz += max(-ACC_WZ * DT, min(ACC_WZ * DT, cwz - wz))
        yaw += wz * DT
        c, s_ = math.cos(yaw), math.sin(yaw)
        x, y = x + (vx * c - vy * s_) * DT, y + (vx * s_ + vy * c) * DT
        t += DT
    near = sorted(p for p, c in min_clear.items() if c < NEAR_MISS_M)
    locks = None
    if crowd:
        crowd.finish()
        locks = crowd.lock_summary()
        hunters_hit = {r["hunter"] for r in locks["records"] if r["hit"]}
        locks["ambient_contacts"] = sorted(p for p, c in min_clear.items() if c < 0 and p not in hunters_hit)
    if tr is not None:
        tr.save(trace_path, seed=seed, controller=controller, scenario=scenario.id, segs=segs, stops=stops,
                crowd=crowd)
    return {
        "seed": seed, "scenario": scenario.id, "controller": controller, "completed": done, "stuck": stuck,
        "time_s": round(t, 1), "checkpoints": min(goal_i, len(stops) - 1), "of": len(stops) - 1,
        "people": len(people_actors), "min_clearance_m": {k: round(v, 3) for k, v in min_clear.items()},
        "closest_m": round(min(min_clear.values()), 3) if min_clear else None,
        "near_misses": near, "contacts": sorted(p for p, c in min_clear.items() if c < 0),
        "min_wall_clearance_m": round(min_wall, 3), "wall_contact_steps": int(wall_contacts),
        "ms_per_decision": round(1000 * plan_s / max(n_plans, 1), 2), "statuses": status_counts,
        **({"locks": locks} if locks else {}),
    }


class Trace:
    """Opt-in per-step record of one episode (run_episode(trace_path=...)), for contact forensics.

    One row per controller call (an arrival step repeats t): robot pose and simulator velocity at t (before this
    step's command is applied), the command, status, goal index, the controller's own velocity estimate if it keeps
    one (``ctrl.vel``, NaN otherwise), and every person's position (NaN while a keyframed actor is absent), crowd
    state and heading. Only reads state; never touches an RNG."""

    STATUS = ("moving", "turning", "arrived", "blocked")

    def __init__(self, people_actors, crowd):
        self.ids = [a.id for a in people_actors] + (list(crowd.ids) if crowd else [])
        self.n_kf = len(people_actors)
        self.rows, self.pos, self.state, self.heading = [], [], [], []

    def record(self, t, pose, vel, cmd, status, goal_i, people_actors, crowd, ctrl) -> None:
        cv = getattr(ctrl, "vel", None)
        cv = (math.nan,) * 3 if cv is None else tuple(float(v) for v in cv)
        code = self.STATUS.index(status) if status in self.STATUS else len(self.STATUS)
        self.rows.append((t, *pose, *vel, *cmd, code, goal_i, *cv))
        pos = np.full((len(self.ids), 2), np.nan)
        st, hd = np.full(len(self.ids), -1, dtype=np.int8), np.full(len(self.ids), np.nan)
        for k, a in enumerate(people_actors):
            at = _actor_at(a, t)
            if at is not None:
                pos[k] = at[0]
        if crowd:
            pos[self.n_kf:], st[self.n_kf:], hd[self.n_kf:] = crowd.pos, crowd.state, crowd.heading
        self.pos.append(pos.astype(np.float32))
        self.state.append(st)
        self.heading.append(hd.astype(np.float32))

    COLUMNS = ("t", "x", "y", "yaw", "vx", "vy", "wz", "cvx", "cvy", "cwz", "status", "goal_i", "ctrl_vx", "ctrl_vy",
               "ctrl_wz")

    def save(self, path, *, seed, controller, scenario, segs, stops, crowd) -> None:
        kind = np.full(len(self.ids), -1, dtype=np.int8)  # -1 = keyframed actor, else pedestrians.KIND_NAMES index
        if crowd:
            kind[self.n_kf:] = crowd.kind
        meta = {"seed": seed, "controller": controller, "scenario": scenario, "columns": list(self.COLUMNS),
                "status_names": list(self.STATUS), "ids": self.ids,
                "locks": [asdict(r) | {"hit": r.hit} for r in crowd.locks] if crowd else []}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, robot=np.array(self.rows, dtype=np.float64), pos=np.stack(self.pos),
                            state=np.stack(self.state), heading=np.stack(self.heading), kind=kind,
                            segs=np.asarray(segs, dtype=np.float64), stops=np.asarray(stops, dtype=np.float64),
                            meta=np.array(json.dumps(meta)))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def summarize(rows: list[dict]) -> dict:
    out = {}
    for name in sorted({r["controller"] for r in rows}):
        rs = [r for r in rows if r["controller"] == name]
        n = len(rs)
        nm = sum(1 for r in rs if r["near_misses"])
        ct = sum(1 for r in rs if r["contacts"])
        comp = sum(1 for r in rs if r["completed"])
        closest = sorted(r["closest_m"] for r in rs if r["closest_m"] is not None)
        times = sorted(r["time_s"] for r in rs if r["completed"])
        out[name] = {
            "episodes": n,
            "completed": comp, "completed_ci": [round(v, 3) for v in wilson(comp, n)],
            "episodes_with_near_miss": nm, "near_miss_rate_ci": [round(v, 3) for v in wilson(nm, n)],
            "episodes_with_contact": ct, "contact_rate_ci": [round(v, 3) for v in wilson(ct, n)],
            "closest_m_median": closest[len(closest) // 2] if closest else None,
            "closest_m_p05": closest[int(0.05 * len(closest))] if closest else None,
            "time_s_median": times[len(times) // 2] if times else None,
            "stuck": sum(1 for r in rs if r["stuck"]),
            "wall_contact_episodes": sum(1 for r in rs if r["wall_contact_steps"]),
            "ms_per_decision_median": sorted(r["ms_per_decision"] for r in rs)[n // 2],
        }
        lk = [r["locks"] for r in rs if r.get("locks")]
        if lk:
            locks, committed = sum(x["locks"] for x in lk), sum(x["committed"] for x in lk)
            hit = sum(x["hit"] for x in lk)
            out[name]["live_crowd"] = {
                "locks": locks, "committed": committed, "hit": hit, "evaded": sum(x["evaded"] for x in lk),
                "hit_per_commit": round(hit / committed, 3) if committed else None,
                "hit_per_commit_ci": [round(v, 3) for v in wilson(hit, committed)],
                "locks_per_run": round(locks / len(lk), 2),
                "episodes_with_ambient_contact": sum(1 for x in lk if x["ambient_contacts"]),
            }
    return out


def _job(args):
    return run_episode(*args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-20")
    ap.add_argument("--controllers", default="control,heuristic")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=str(REPO / "data/benchmarks"))
    ap.add_argument("--live", default=None, help="live closed-loop crowd tier (crowd.LIVE_TIERS: base | hard)")
    a = ap.parse_args()
    seeds, ctrls = parse_seeds(a.seeds), a.controllers.split(",")
    for s in seeds:  # write scenario files once, before the workers race to create them
        from benchmarks.avoidance.crowd import to_scenario_file
        to_scenario_file(generate_live(s, a.live) if a.live else generate(s), REPO / "data/scenarios/crowd")
    jobs = [(s, c, 700.0, None, a.live) for c in ctrls for s in seeds]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(_job, jobs))
    out = Path(a.out) / f"sim2d-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {"seeds": a.seeds, "controllers": ctrls, "wall_s": round(time.time() - t0, 1), "by_controller": summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"OUT {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
