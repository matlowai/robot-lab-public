"""Contact forensics for the sim2d live-crowd benchmark: one record per contact, from the opt-in per-step traces.

    # 1. traced eval (rl/eval.py --trace-dir; episodes and summary are unchanged by tracing)
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m benchmarks.avoidance.rl.eval --ckpt <ckpt> --seeds 121-200 \
        --controllers rl,heuristic --live hard --out <out> --trace-dir <traces>
    # 2. records + tables (numpy only)
    PYTHONPATH=. python tools/contact_forensics.py records --traces "rl_v3=<traces>/rl-*.npz" ... \
        --episodes rl_v3=<out>/rl-eval-*/episodes.jsonl ... --out <dir>
    # 3. top-down strips of chosen contacts (needs matplotlib)
    PYTHONPATH=. python tools/contact_forensics.py render --records <dir>/contacts.jsonl --traces ... --pick ...

A contact is an ONSET: the step where a person's clearance (centre distance - PERSON_R - SPOT_R) goes below 0
from >= 0. sim2d's own metric counts people per run (min clearance < 0), so one person can make several onsets.
Who: wanderer | loiterer | hunter_unlocked (walk / pause / cooldown at the onset) | hunter_lock (locked /
committed: a hunter doing its job, scored separately as hit-per-commit). "Ambient" = everything but hunter_lock.

Sensor facts used (sim2d.lidar = rl/env.py): 181 rays over 270 deg (+-135 deg from the heading; the 90 deg behind
is blind), 12 m range, returns closer than 0.65 m from the robot centre are dropped, people are 0.30 m circles and
occlude what's behind them. "Visible" at a step = at least one ray's first return is this person.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from benchmarks.avoidance.pedestrians import KIND_NAMES, STATE_NAMES
from benchmarks.avoidance.sim2d import DT, PERSON_R, SPOT_R

FOV = math.radians(270)
RAYS, MAX_RANGE, RAY_START = 181, 12.0, 0.65
REL = np.linspace(-FOV / 2, FOV / 2, RAYS)
LOCKED, COMMITTED = STATE_NAMES.index("locked"), STATE_NAMES.index("committed")
RECENT_S = 0.6  # RL's scan memory: scans at t, t - 0.3, t - 0.6 (avoid-v2)
LATE_S = 1.0  # first seen less than this before contact = "seen late" (a 1 m/s robot needs ~0.7 s to stop)
SLOW_MPS = 0.25  # robot speed at contact below this = stopped / slow
APPROACH_MPS = 0.2  # an approach component below this = "not moving into the other"


# --- loading ---------------------------------------------------------------------------------------------------

def load(path) -> dict:
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    cols = {c: i for i, c in enumerate(meta["columns"])}
    rob = z["robot"]
    return {"meta": meta, "rob": rob, "c": cols, "pos": z["pos"].astype(np.float64), "state": z["state"],
            "heading": z["heading"], "kind": z["kind"], "segs": z["segs"], "stops": z["stops"],
            "t": rob[:, cols["t"]], "xy": rob[:, [cols["x"], cols["y"]]], "yaw": rob[:, cols["yaw"]]}


# --- geometry --------------------------------------------------------------------------------------------------

def ray_hits(pose, segs, people):
    """Per-ray first-return range and the index of the person it hit (-1 = wall / nothing). sim2d.lidar's model."""
    x, y, yaw = pose
    ang = yaw + REL
    dx, dy = np.cos(ang), np.sin(ang)
    best = np.full(RAYS, MAX_RANGE)
    who = np.full(RAYS, -1)
    if len(segs):
        ax, ay, bx, by = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
        ex, ey = bx - ax, by - ay
        den = dx[:, None] * ey[None, :] - dy[:, None] * ex[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((ax - x)[None, :] * ey[None, :] - (ay - y)[None, :] * ex[None, :]) / den
            u = ((ax - x)[None, :] * dy[:, None] - (ay - y)[None, :] * dx[:, None]) / den
        ok = (np.abs(den) > 1e-12) & (t > RAY_START) & (u >= 0) & (u <= 1)
        best = np.minimum(best, np.where(ok, t, np.inf).min(axis=1))
    ok_p = np.isfinite(people[:, 0])
    if ok_p.any():
        pp = people[ok_p]
        idx = np.nonzero(ok_p)[0]
        fx, fy = x - pp[:, 0], y - pp[:, 1]
        b = dx[:, None] * fx[None, :] + dy[:, None] * fy[None, :]
        c = (fx * fx + fy * fy - PERSON_R ** 2)[None, :]
        disc = b * b - c
        with np.errstate(invalid="ignore"):
            t = -b - np.sqrt(disc)
        t = np.where((disc >= 0) & (t > RAY_START), t, np.inf)
        j = t.argmin(axis=1)
        tp = t[np.arange(RAYS), j]
        closer = tp < best
        best = np.where(closer, tp, best)
        who = np.where(closer, idx[j], who)
    return best, who


def wall_clearance(p, segs) -> float:
    ax, ay, bx, by = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    ex, ey = bx - ax, by - ay
    t = np.clip(((p[0] - ax) * ex + (p[1] - ay) * ey) / np.maximum(ex * ex + ey * ey, 1e-12), 0, 1)
    return float(np.min(np.hypot(p[0] - (ax + t * ex), p[1] - (ay + t * ey)))) - SPOT_R


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# --- per-contact records ---------------------------------------------------------------------------------------

def _row_at(tr, k0, dt_back: float) -> int:
    """Index of the last row at or before t[k0] - dt_back (0 if before the start)."""
    return max(0, int(np.searchsorted(tr["t"], tr["t"][k0] - dt_back + 1e-9, side="right")) - 1)


def _bearing(tr, k, p) -> float:
    d = tr["pos"][k, p] - tr["xy"][k]
    return math.degrees(_wrap(math.atan2(d[1], d[0]) - tr["yaw"][k]))


def _world_vel(xy, t, k, window=0.5):
    """Mean world velocity over the last `window` s before row k (displacement / elapsed)."""
    j = max(0, int(np.searchsorted(t, t[k] - window + 1e-9, side="left")))
    el = t[k] - t[j]
    return (xy[k] - xy[j]) / el if el > 1e-9 else np.zeros(2)


def _stop_avoids(tr, k0, p, lead_s: float, after_s: float = 2.0) -> bool | None:
    """True if braking to a stop from t0 - lead_s (all commands 0, sim2d's acceleration limits, from the recorded
    state) keeps clearance >= 0 against the person's recorded path until t0 + after_s. None if the trace ends."""
    from benchmarks.avoidance.sim2d import ACC_VX, ACC_VY, ACC_WZ
    t, c, rob = tr["t"], tr["c"], tr["rob"]
    ks = _row_at(tr, k0, lead_s)
    x, y, yaw = rob[ks, c["x"]], rob[ks, c["y"]], rob[ks, c["yaw"]]
    vx, vy, wz = rob[ks, c["vx"]], rob[ks, c["vy"]], rob[ks, c["wz"]]
    t_end = t[k0] + after_s
    if t[-1] < t_end - 1e-9:
        return None
    k = ks
    while t[k] < t_end - 1e-9:
        vx -= min(ACC_VX * DT, vx) if vx > 0 else max(-ACC_VX * DT, vx)
        vy -= min(ACC_VY * DT, vy) if vy > 0 else max(-ACC_VY * DT, vy)
        wz -= min(ACC_WZ * DT, wz) if wz > 0 else max(-ACC_WZ * DT, wz)
        yaw += wz * DT
        x, y = x + (vx * math.cos(yaw) - vy * math.sin(yaw)) * DT, y + (vx * math.sin(yaw) + vy * math.cos(yaw)) * DT
        k = int(np.searchsorted(t, t[k] + DT - 1e-9, side="left"))  # the next distinct time
        if k >= len(t):
            return None
        if math.hypot(tr["pos"][k, p, 0] - x, tr["pos"][k, p, 1] - y) - PERSON_R - SPOT_R < 0:
            return False
    return True


def contact_onsets(tr):
    """[(row, person)] where a person's clearance first goes below 0 (again)."""
    d = np.hypot(tr["pos"][..., 0] - tr["xy"][:, None, 0], tr["pos"][..., 1] - tr["xy"][:, None, 1]) - PERSON_R - SPOT_R
    d = np.where(np.isfinite(d), d, np.inf)
    inside = d < 0
    prev = np.vstack([np.zeros((1, d.shape[1]), bool), inside[:-1]])
    ks, ps = np.nonzero(inside & ~prev)
    return list(zip(ks.tolist(), ps.tolist())), d


def who_of(tr, k, p) -> str:
    kind = int(tr["kind"][p])
    if kind < 0:
        return "keyframed"
    name = KIND_NAMES[kind]
    if name != "hunter":
        return name
    return "hunter_lock" if int(tr["state"][k, p]) in (LOCKED, COMMITTED) else "hunter_unlocked"


def record(tr, k0, p, dist, episode_row: dict | None) -> dict:
    t, c, rob = tr["t"], tr["c"], tr["rob"]
    t0 = float(t[k0])
    meta = tr["meta"]
    pid = meta["ids"][p]
    who = who_of(tr, k0, p)
    ctr = np.hypot(*(tr["pos"][:, p] - tr["xy"]).T)  # centre distance per row

    # visibility over the last 3 s (only distinct times)
    k_start = _row_at(tr, k0, 3.0)
    vis_t, vis = [], []
    for k in range(k_start, k0 + 1):
        if k > k_start and t[k] == t[k - 1]:
            continue
        _, whohit = ray_hits((*tr["xy"][k], tr["yaw"][k]), tr["segs"], tr["pos"][k])
        vis_t.append(t[k])
        vis.append(bool((whohit == p).any()))
    vis_t, vis = np.array(vis_t), np.array(vis)
    # what the controller could have known: scans strictly before the contact step
    pre = vis[:-1] if len(vis) > 1 else vis
    pre_t = vis_t[:-1] if len(vis) > 1 else vis_t
    seen_any_3s = bool(pre.any())
    first_seen_s = float(t0 - pre_t[np.argmax(pre)]) if seen_any_3s else 0.0  # capped at the 3 s window
    seen_recent = bool(pre[pre_t >= t0 - RECENT_S - 1e-9].any())  # inside RL's scan memory (t, t-0.3, t-0.6)
    seen_frac_2s = float(pre[pre_t >= t0 - 2.0 - 1e-9].mean()) if (pre_t >= t0 - 2.0 - 1e-9).any() else 0.0

    # bearings and field of view
    bear = {f"bearing_{lab}": round(_bearing(tr, _row_at(tr, k0, s), p), 1) for lab, s in (("0s", 0), ("1s", 1.0), ("2s", 2.0))}
    in_fov = {f"in_fov_{lab[8:]}": abs(v) <= 135.0 for lab, v in bear.items()}

    # relative motion at the onset (0.5 s mean velocities); n = unit vector robot -> person
    vr = _world_vel(tr["xy"], t, k0)
    vp = _world_vel(tr["pos"][:, p], t, k0)
    rel = tr["pos"][k0, p] - tr["xy"][k0]
    n = rel / max(np.hypot(*rel), 1e-9)
    robot_app, person_app = float(vr @ n), float(-vp @ n)
    closing = robot_app + person_app
    # who closed the gap over the last 2 s: each step's displacement projected on the robot -> person direction
    kk = [k for k in range(_row_at(tr, k0, 2.0), k0 + 1) if k == 0 or t[k] != t[k - 1]]
    r_close = p_close = 0.0
    for ka, kb in zip(kk, kk[1:]):
        d_ = tr["pos"][ka, p] - tr["xy"][ka]
        n_ = d_ / max(np.hypot(*d_), 1e-9)
        r_close += float((tr["xy"][kb] - tr["xy"][ka]) @ n_)
        p_close -= float((tr["pos"][kb, p] - tr["pos"][ka, p]) @ n_)
    tot = r_close + p_close
    robot_share = r_close / tot if tot > 1e-6 else float("nan")
    if not tot > 1e-6 or robot_share <= 0.35:
        mover = "person_closed"
    elif robot_share >= 0.65:
        mover = "robot_closed"
    else:
        mover = "both_closed"
    robot_speed = float(np.hypot(rob[k0, c["vx"]], rob[k0, c["vy"]]))
    person_speed = float(np.hypot(*vp))

    # time from entering 2 m (centre to centre) to contact
    out = np.nonzero(ctr[:k0] > 2.0)[0]
    t_2m = float(t0 - t[out[-1] + 1]) if len(out) else float("nan")

    # the robot's reaction over the 2 s before contact
    k2 = _row_at(tr, k0, 2.0)
    sl = slice(k2, k0 + 1)
    sp = np.hypot(rob[sl, c["vx"]], rob[sl, c["vy"]])
    speed_2s, speed_1s = float(sp[0]), float(np.hypot(rob[_row_at(tr, k0, 1.0), c["vx"]], rob[_row_at(tr, k0, 1.0), c["vy"]]))
    braked = float(sp.min()) < speed_2s - 0.3
    turned = float(np.abs(rob[sl, c["wz"]]).max()) > 0.4
    sidestep = float(np.abs(rob[sl, c["vy"]]).max()) > 0.2
    v_a, v_b = _world_vel(tr["xy"], t, k2), _world_vel(tr["xy"], t, _row_at(tr, k0, 0.3))
    swerve = abs(math.degrees(_wrap(math.atan2(v_b[1], v_b[0]) - math.atan2(v_a[1], v_a[0])))) \
        if min(np.hypot(*v_a), np.hypot(*v_b)) > 0.2 else 0.0
    cvx = rob[sl, c["cvx"]]

    # encounter geometry: the person's walking direction relative to the robot's heading, 1 s before contact
    k1 = _row_at(tr, k0, 1.0)
    vp1 = _world_vel(tr["pos"][:, p], t, k1, window=1.0)
    if np.hypot(*vp1) < 0.2:
        geometry = "standing"
    else:
        rel_dir = abs(math.degrees(_wrap(math.atan2(vp1[1], vp1[0]) - tr["yaw"][k1])))
        geometry = "same_direction" if rel_dir < 45 else ("head_on" if rel_dir > 135 else "crossing")

    # counterfactual for oblivious people (their next seconds don't depend on the robot): brake to a stop from
    # t0 - L (command 0, sim2d's acceleration limits) and keep the recorded person path; still a contact?
    cf = {}
    if who in ("wanderer", "loiterer"):
        for L in (1.0, 2.0):
            cf[f"cf_stop_{L:.0f}s_avoids"] = _stop_avoids(tr, k0, p, L)

    # where on the patrol
    stops = tr["stops"]
    gi = int(rob[k0, c["goal_i"]])
    prev_stop = stops[gi - 1] if gi > 0 else stops[-1]  # the patrol starts (and ends) at the charger = stops[-1]
    d_prev = float(np.hypot(*(tr["xy"][k0] - prev_stop)))
    d_next = float(np.hypot(*(tr["xy"][k0] - stops[gi])))
    wc = wall_clearance(tr["xy"][k0], tr["segs"])
    goal_dir = math.atan2(*(stops[gi] - tr["xy"][k0])[::-1])
    head_err = abs(math.degrees(_wrap(goal_dir - tr["yaw"][k0])))

    # category (mutually exclusive, in this order)
    speed_last1s = float(np.hypot(rob[_row_at(tr, k0, 1.0):k0 + 1, c["vx"]], rob[_row_at(tr, k0, 1.0):k0 + 1, c["vy"]]).mean())
    if not seen_any_3s:
        cat = "a1_never_seen"
    elif not seen_recent:
        cat = "a2_seen_then_lost"
    elif first_seen_s < LATE_S:
        cat = "a3_seen_late"
    elif speed_last1s < SLOW_MPS:
        cat = "b_person_into_slow_robot"
    elif mover == "person_closed":
        cat = "c2_person_closed_on_moving_robot"
    else:
        cat = "c_robot_closed_on_seen_person"

    return {
        "controller": meta["controller"], "seed": meta["seed"], "person": pid, "who": who,
        "ambient": who != "hunter_lock",
        "sim2d_ambient": bool(episode_row and pid in episode_row.get("locks", {}).get("ambient_contacts", [])),
        "t": round(t0, 1), "row": int(k0), "state": STATE_NAMES[int(tr["state"][k0, p])] if tr["kind"][p] >= 0 else None,
        **bear, **in_fov,
        "first_seen_s": round(first_seen_s, 2), "seen_any_3s": seen_any_3s, "seen_recent": seen_recent,
        "seen_frac_2s": round(seen_frac_2s, 2), "speed_last1s": round(speed_last1s, 2),
        "robot_close_2s": round(r_close, 2), "person_close_2s": round(p_close, 2),
        "robot_share_2s": round(robot_share, 2) if math.isfinite(robot_share) else None,
        "robot_speed": round(robot_speed, 2), "robot_speed_1s": round(speed_1s, 2), "robot_speed_2s": round(speed_2s, 2),
        "person_speed": round(person_speed, 2),
        "robot_approach": round(robot_app, 2), "person_approach": round(person_app, 2), "closing": round(closing, 2),
        "mover": mover, "t_from_2m_s": round(t_2m, 2) if math.isfinite(t_2m) else None,
        "braked": bool(braked), "turned": bool(turned), "sidestep": bool(sidestep), "swerve_deg": round(swerve, 1),
        "reacted": bool(braked or swerve > 30.0), "geometry": geometry, **cf,
        "cmd_vx_mean_2s": round(float(cvx.mean()), 2), "cmd_vx_min_2s": round(float(cvx.min()), 2),
        "status": int(rob[k0, c["status"]]),
        "leg": gi, "d_prev_stop": round(d_prev, 1), "d_next_stop": round(d_next, 1),
        "near_stop": min(d_prev, d_next) < 3.0, "wall_clearance": round(wc, 2), "near_wall": wc < 2.0,
        "heading_err_to_goal_deg": round(head_err, 1),
        "category": cat,
    }


def exposure(tr, radius=4.0) -> dict:
    """Person-seconds within `radius` m (centre distance) of the robot, by kind; and the run's duration."""
    t = tr["t"]
    distinct = np.concatenate([[True], np.diff(t) > 1e-9])
    ctr = np.hypot(tr["pos"][..., 0] - tr["xy"][:, None, 0], tr["pos"][..., 1] - tr["xy"][:, None, 1])
    near = (ctr < radius) & distinct[:, None]
    out = {"duration_s": float(t[-1] - t[0])}
    kinds = tr["kind"]
    for k, name in enumerate(KIND_NAMES):
        out[f"{name}_s"] = float(near[:, kinds == k].sum() * DT)
    return out


# --- commands --------------------------------------------------------------------------------------------------

def _trace_sets(specs):
    """["label=glob", ...] -> {label: [paths]}"""
    out = {}
    for s in specs:
        label, _, pat = s.partition("=")
        out[label] = sorted(glob.glob(pat))
    return out


def _episode_rows(specs):
    """["label=episodes.jsonl", ...] -> {(label, controller, seed): row}. Labelled, because two checkpoints
    evaluated under the same controller name ("rl") would otherwise overwrite each other."""
    rows = {}
    for spec in specs:
        label, _, path = spec.partition("=")
        for line in open(path):
            r = json.loads(line)
            rows[(label, r["controller"], r["seed"])] = r
    return rows


def cmd_records(a):
    sets = _trace_sets(a.traces)
    episodes = _episode_rows(a.episodes)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs, expo = [], []
    for label, paths in sets.items():
        for path in paths:
            tr = load(path)
            ep = episodes.get((label, tr["meta"]["controller"], tr["meta"]["seed"]))
            onsets, dist = contact_onsets(tr)
            for k0, p in onsets:
                r = record(tr, k0, p, dist, ep)
                r["label"] = label
                recs.append(r)
            e = exposure(tr)
            e |= {"label": label, "seed": tr["meta"]["seed"]}
            expo.append(e)
        print(f"{label}: {len(paths)} traces")
    (out / "contacts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (out / "exposure.jsonl").write_text("".join(json.dumps(r) + "\n" for r in expo))
    print(tables(recs, expo))
    return 0


def _pct(n, d):
    return f"{n} ({100 * n / d:.0f}%)" if d else "0"


def tables(recs, expo) -> str:
    """Markdown tables: ambient onsets by who / category / mover, runs with ambient contact, exposure rates."""
    labels = list(dict.fromkeys(r["label"] for r in expo))
    runs = {lab: sum(1 for e in expo if e["label"] == lab) for lab in labels}
    amb = {lab: [r for r in recs if r["label"] == lab and r["ambient"]] for lab in labels}
    lines = []

    def table(title, key, order=None):
        vals = order or sorted({r[key] for lab in labels for r in amb[lab]}, key=str)
        lines.append(f"\n**{title}** (ambient contact onsets)\n")
        lines.append("| " + key + " | " + " | ".join(labels) + " |")
        lines.append("|---|" + "---|" * len(labels))
        for v in vals:
            lines.append(f"| {v} | " + " | ".join(_pct(sum(1 for r in amb[lab] if r[key] == v), len(amb[lab])) for lab in labels) + " |")
        lines.append("| **total** | " + " | ".join(str(len(amb[lab])) for lab in labels) + " |")

    lines.append("| | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    lines.append("| runs | " + " | ".join(str(runs[lab]) for lab in labels) + " |")
    lines.append("| runs with an ambient onset | " + " | ".join(
        str(len({r['seed'] for r in amb[lab]})) for lab in labels) + " |")
    lines.append("| runs with sim2d ambient contact | " + " | ".join(
        str(len({r['seed'] for r in recs if r['label'] == lab and r['sim2d_ambient']})) for lab in labels) + " |")
    lines.append("| ambient onsets | " + " | ".join(str(len(amb[lab])) for lab in labels) + " |")
    lines.append("| hunter-lock onsets | " + " | ".join(
        str(sum(1 for r in recs if r['label'] == lab and not r['ambient'])) for lab in labels) + " |")
    dur = {lab: sum(e["duration_s"] for e in expo if e["label"] == lab) for lab in labels}
    lines.append("| patrol time (s, all runs) | " + " | ".join(f"{dur[lab]:.0f}" for lab in labels) + " |")
    for kind in ("wanderer", "loiterer", "hunter"):
        ex = {lab: sum(e[f"{kind}_s"] for e in expo if e["label"] == lab) for lab in labels}
        lines.append(f"| {kind}-s within 4 m | " + " | ".join(f"{ex[lab]:.0f}" for lab in labels) + " |")
    wex = {lab: sum(e["wanderer_s"] for e in expo if e["label"] == lab) for lab in labels}
    lines.append("| wanderer onsets per 100 wanderer-s within 4 m | " + " | ".join(
        f"{100 * sum(1 for r in amb[lab] if r['who'] == 'wanderer') / max(wex[lab], 1e-9):.2f}" for lab in labels) + " |")
    table("Who", "who", ["wanderer", "loiterer", "hunter_unlocked"])
    table("Category", "category", ["a1_never_seen", "a2_seen_then_lost", "a3_seen_late", "b_person_into_slow_robot",
                                   "c_robot_closed_on_seen_person", "c2_person_closed_on_moving_robot"])
    table("Who closed the gap over the last 2 s", "mover", ["robot_closed", "both_closed", "person_closed"])
    table("In the lidar FOV at contact", "in_fov_0s", [True, False])
    table("In the lidar FOV 1 s before", "in_fov_1s", [True, False])
    table("Reacted in the 2 s before (braked >= 0.3 m/s or swerved > 30 deg)", "reacted", [True, False])
    table("Braked (speed fell >= 0.3 m/s) in the 2 s before", "braked", [True, False])
    table("Encounter geometry (person's direction vs robot heading, 1 s before)", "geometry",
          ["crossing", "head_on", "same_direction", "standing"])
    table("Near a patrol stop (< 3 m)", "near_stop", [True, False])
    table("Near a wall (< 2 m clearance)", "near_wall", [True, False])

    lines.append("\n**Counterfactual: brake to a stop, wanderer / loiterer onsets** (their paths don't depend on the "
                 "robot; share of onsets a stop would have avoided)\n")
    lines.append("| | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for key in ("cf_stop_1s_avoids", "cf_stop_2s_avoids"):
        cells = []
        for lab in labels:
            v = [r[key] for r in amb[lab] if r.get(key) is not None]
            cells.append(f"{sum(v)}/{len(v)}" if v else "-")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")

    def med(lab, key, cond=lambda r: True):
        v = sorted(r[key] for r in amb[lab] if cond(r) and r[key] is not None)
        return f"{v[len(v) // 2]:.2f}" if v else "-"

    lines.append("\n**Medians** (ambient onsets)\n")
    lines.append("| | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for key in ("robot_speed", "robot_speed_2s", "person_speed", "robot_approach", "person_approach", "closing",
                "first_seen_s", "robot_share_2s", "speed_last1s", "t_from_2m_s", "cmd_vx_mean_2s", "cmd_vx_min_2s", "swerve_deg", "wall_clearance"):
        lines.append(f"| {key} | " + " | ".join(med(lab, key) for lab in labels) + " |")
    return "\n".join(lines)


def cmd_tables(a):
    recs = [json.loads(l) for l in open(Path(a.records) / "contacts.jsonl")]
    expo = [json.loads(l) for l in open(Path(a.records) / "exposure.jsonl")]
    print(tables(recs, expo))
    return 0


# --- rendering -------------------------------------------------------------------------------------------------

COLORS = {"wanderer": "#2f6fdb", "loiterer": "#7a7a7a", "hunter": "#e0892b", "hunter_locked": "#d62728",
          "robot": "#111111", "target": "#b000b0"}


def draw_frame(ax, tr, k, p_focus, half=7.0, trail_s=3.0, title=None):
    from matplotlib.patches import Circle, Wedge
    x, y = tr["xy"][k]
    yaw = tr["yaw"][k]
    ax.set_xlim(x - half, x + half)
    ax.set_ylim(y - half, y + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    # blind cone behind the robot (the 90 deg the lidar does not cover), out to 4 m
    ax.add_patch(Wedge((x, y), 4.0, math.degrees(yaw) + 135, math.degrees(yaw) + 225, color="#f3c0c0", alpha=0.6, lw=0))
    for s in tr["segs"]:
        ax.plot([s[0], s[2]], [s[1], s[3]], color="#444444", lw=2)
    rng, who = ray_hits((x, y, yaw), tr["segs"], tr["pos"][k])
    ang = yaw + REL
    hit = rng < MAX_RANGE
    ax.scatter(x + rng[hit] * np.cos(ang[hit]), y + rng[hit] * np.sin(ang[hit]), s=3,
               c=np.where(who[hit] == p_focus, COLORS["target"], "#20a020"), zorder=3)
    k_tr = _row_at(tr, k, trail_s)
    ax.plot(tr["xy"][k_tr:k + 1, 0], tr["xy"][k_tr:k + 1, 1], color=COLORS["robot"], lw=1, alpha=0.6)
    ax.add_patch(Circle((x, y), SPOT_R, fc="none", ec=COLORS["robot"], lw=1.5, zorder=4))
    ax.arrow(x, y, 0.9 * math.cos(yaw), 0.9 * math.sin(yaw), width=0.06, color=COLORS["robot"], zorder=5)
    near = np.nonzero(np.hypot(*(tr["pos"][k] - tr["xy"][k]).T) < half * 1.5)[0]
    for p in near:
        kind = int(tr["kind"][p])
        name = KIND_NAMES[kind] if kind >= 0 else "wanderer"
        if name == "hunter" and int(tr["state"][k, p]) in (LOCKED, COMMITTED):
            name = "hunter_locked"
        col = COLORS["target"] if p == p_focus else COLORS[name]
        px, py = tr["pos"][k, p]
        ax.add_patch(Circle((px, py), PERSON_R, fc=col, ec="k" if p == p_focus else "none", lw=1, alpha=0.9, zorder=4))
        if p == p_focus:
            ax.plot(tr["pos"][k_tr:k + 1, p, 0], tr["pos"][k_tr:k + 1, p, 1], color=col, lw=1, ls="--")
    if title:
        ax.set_title(title, fontsize=8)


def render_strip(tr, rec, path, offsets=(-3.0, -2.0, -1.0, -0.5, 0.0)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k0 = rec["row"]
    p = tr["meta"]["ids"].index(rec["person"])
    fig, axes = plt.subplots(1, len(offsets), figsize=(2.3 * len(offsets), 2.7), dpi=90)
    c = tr["c"]
    for ax, off in zip(axes, offsets):
        k = _row_at(tr, k0, -off)
        v = math.hypot(tr["rob"][k, c["vx"]], tr["rob"][k, c["vy"]])
        _, whohit = ray_hits((*tr["xy"][k], tr["yaw"][k]), tr["segs"], tr["pos"][k])
        seen = "seen" if (whohit == p).any() else "NOT seen"
        draw_frame(ax, tr, k, p, title=f"t{off:+.1f}s  v={v:.2f}  {seen}")
    fig.suptitle(f"{rec['label']} seed {rec['seed']} t={rec['t']}s  {rec['who']}  {rec['category']}  "
                 f"robot {rec['robot_approach']:+.2f} / person {rec['person_approach']:+.2f} m/s", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_gif(tr, rec, path, before_s=4.0, after_s=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    k0 = rec["row"]
    p = tr["meta"]["ids"].index(rec["person"])
    frames = []
    ks = [k for k in range(_row_at(tr, k0, before_s), min(len(tr["t"]), k0 + int(after_s / DT) + 1))
          if k == 0 or tr["t"][k] != tr["t"][k - 1]]
    for k in ks[::2]:
        fig, ax = plt.subplots(figsize=(3.2, 3.4), dpi=80)
        v = math.hypot(tr["rob"][k, tr["c"]["vx"]], tr["rob"][k, tr["c"]["vy"]])
        draw_frame(ax, tr, k, p, title=f"{rec['label']} s{rec['seed']}  t={tr['t'][k] - tr['t'][k0]:+.1f}s  v={v:.2f}")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0, optimize=True)


def cmd_render(a):
    recs = [json.loads(l) for l in open(a.records)]
    sets = _trace_sets(a.traces)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for pick in a.pick:  # label:seed:person:t
        label, seed, person, t = pick.split(":")
        rec = next(r for r in recs if r["label"] == label and r["seed"] == int(seed) and r["person"] == person
                   and abs(r["t"] - float(t)) < 0.05)
        path = next(p for p in sets[label] if re.search(rf"-{int(seed):03d}\.npz$", p))
        tr = load(path)
        stem = f"{label}-s{int(seed):03d}-{person}-t{float(t):.1f}"
        render_strip(tr, rec, out / f"{stem}.png")
        if a.gif:
            render_gif(tr, rec, out / f"{stem}.gif")
        print(out / f"{stem}.png")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("records")
    r.add_argument("--traces", nargs="+", required=True, help="label=glob of trace .npz files")
    r.add_argument("--episodes", nargs="*", default=[],
                   help="label=episodes.jsonl (the eval's rows for that trace set; for sim2d's per-run ambient flag)")
    r.add_argument("--out", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--records", required=True, help="directory with contacts.jsonl + exposure.jsonl")
    d = sub.add_parser("render")
    d.add_argument("--records", required=True, help="contacts.jsonl")
    d.add_argument("--traces", nargs="+", required=True)
    d.add_argument("--pick", nargs="+", required=True, help="label:seed:person:t")
    d.add_argument("--out", required=True)
    d.add_argument("--gif", action="store_true")
    a = ap.parse_args(argv)
    return {"records": cmd_records, "tables": cmd_tables, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
