"""Closed-loop people in the RL env (benchmarks/avoidance/rl/hunters.py) against the numpy reference
(benchmarks/avoidance/pedestrians.py): same initial state + same scripted robot trajectory -> same positions and
states every step, through lock, re-aim, commit, pass, cooldown, lose-sight, timeout, vision cone, walls and
max_locks. Plus the env-level wiring (presets, episode lock/commit/hit stats).

RNG-driven choices (new destinations, pause lengths) are replaced on BOTH sides by the same deterministic choices
(destination = home + one of four fixed offsets, cycling; pause = 37 % of pause_s), so the parity window covers
full walk/pause/lock/cooldown cycles. Needs torch (skipped otherwise); runs as a plain script too:
    PYTHONPATH=. /mnt/weights/ai/isaac/IsaacLab/.venv/bin/python tests/test_rl_hunters.py
"""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    import pytest
    torch = pytest.importorskip("torch")
except ModuleNotFoundError:  # plain-script mode
    pytest = None
    import torch

from benchmarks.avoidance import pedestrians as P
from benchmarks.avoidance.rl import env as E
from benchmarks.avoidance.rl import hunters as H
from events.zones import Compound

REPO = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")
O = np.array([75.0, 50.0])  # scenario origin in the compound (the numpy side runs in world coordinates)
OFFS = np.array([(3.0, 0.0), (0.0, 3.0), (-3.0, 0.0), (0.0, -3.0)])
PAUSE_FRAC = 0.37
STEPS = 250
CFG = replace(P.CrowdConfig(), cooldown_s=4.0, lock_timeout_s=6.0, max_locks=2)


# --- scenarios (local coordinates; robot velocity profile as (until_t, vx, vy) pieces) ----------------------------

def _hunter(pos, speed, commit, state=P.PAUSE, heading=0.0, target=None, until=1e9, post=None):
    return dict(kind=P.HUNTER, pos=pos, speed=speed, commit=commit, state=state, heading=heading,
                target=target if target is not None else pos, until=until, home=post if post is not None else pos)


def _wanderer(pos, speed, target, heading=None):
    h = math.atan2(target[1] - pos[1], target[0] - pos[0]) if heading is None else heading
    return dict(kind=P.WANDERER, pos=pos, speed=speed, commit=0.0, state=P.WALK, heading=h, target=target, until=0.0,
                home=pos)


SCENARIOS = [
    # A: paused hunter beside a straight line; the robot sidesteps just after the commit and evades; a wanderer
    dict(robot0=(-12.0, 0.0), vel=[(9.4, 1.0, 0.0), (11.0, 0.0, 0.5), (99, 1.0, 0.0)], walls=[],
         people=[_hunter((3.0, 6.0), 1.2, 3.0), _wanderer((-5.0, -8.0), 1.0, (-2.0, -8.0))]),
    # B: hunter walking away from an overtaking robot: outside its vision cone until the robot is ahead
    dict(robot0=(-6.0, 0.0), vel=[(99, 1.6, 0.0)], walls=[],
         people=[_hunter((0.0, 3.0), 0.8, 2.5, state=P.WALK, heading=0.0, target=(40.0, 3.0), post=(0.0, 3.0))]),
    # C: the robot slips behind a wall: the chasing hunter's steps are refused at it and it loses sight; a second
    #    hunter never gets a line of sight past another wall
    dict(robot0=(0.0, -3.0), vel=[(99, 1.5, 0.0)], walls=[(1.03, -6.0, 1.03, 9.0), (9.0, 1.0, 20.0, 1.0)],
         people=[_hunter((0.0, 8.0), 1.2, 3.0), _hunter((14.0, 5.0), 1.2, 3.0)]),
    # D: a fast robot circling a slow hunter that commits late: the lock times out
    dict(robot0=None, circle=(7.0, 2.2), walls=[],
         people=[_hunter((0.0, 0.0), 0.8, 0.5)]),
    # E: hard-tier hunter, robot walking straight on: committed hit, cooldown, re-lock until max_locks
    dict(robot0=(-12.0, 0.0), vel=[(99, 1.0, 0.0)], walls=[],
         people=[_hunter((6.0, 4.0), 1.7, 2.0), _hunter((20.0, -5.0), 1.6, 2.2, state=P.WALK, heading=math.pi,
                                                         target=(14.0, -5.0), post=(17.0, -5.0))]),
]


def robot_track(sc) -> np.ndarray:
    """[STEPS + 1, 2] robot positions at t = k / 10 (float64)."""
    out = np.zeros((STEPS + 1, 2))
    if sc.get("circle"):
        r, v = sc["circle"]
        w = v / r
        for k in range(STEPS + 1):
            out[k] = (r * math.cos(w * k * 0.1), r * math.sin(w * k * 0.1))
        return out
    p = np.array(sc["robot0"])
    out[0] = p
    for k in range(1, STEPS + 1):
        t = (k - 1) * 0.1
        vx, vy = next((vx, vy) for until, vx, vy in sc["vel"] if t < until - 1e-9)
        p = p + np.array([vx, vy]) * 0.1
        out[k] = p
    return out


# --- the two implementations, with the same deterministic "random" choices -------------------------------------

class _FakeRng:
    def uniform(self, lo=0.0, hi=1.0, size=None):
        return lo + PAUSE_FRAC * (hi - lo)


def numpy_crowd(sc, compound) -> P.LiveCrowd:
    ppl = sc["people"]
    n = len(ppl)
    c = P.LiveCrowd(compound, CFG, 0)
    c.rng, c._robot_hist, c._open_lock, c.locks = _FakeRng(), [], {}, []
    c.bsegs = np.array([(ax + O[0], ay + O[1], bx + O[0], by + O[1]) for ax, ay, bx, by in sc["walls"]]).reshape(-1, 4)
    c.ids = [f"person_{i + 1:04d}" for i in range(n)]
    c.kind = np.array([p["kind"] for p in ppl])
    c.pos = np.array([p["pos"] for p in ppl], dtype=np.float64) + O
    c.target = np.array([p["target"] for p in ppl], dtype=np.float64) + O
    home = np.array([p["home"] for p in ppl], dtype=np.float64) + O
    c.post = np.where((c.kind == P.HUNTER)[:, None], home, np.nan)
    c.heading = np.array([p["heading"] for p in ppl], dtype=np.float64)
    c.speed = np.array([p["speed"] for p in ppl], dtype=np.float64)
    c.state = np.array([p["state"] for p in ppl])
    c.until = np.array([p["until"] for p in ppl], dtype=np.float64)
    c.commit_m = np.array([p["commit"] for p in ppl], dtype=np.float64)
    c.locks_used = np.zeros(n, dtype=int)
    c.lock_t, c.last_aim_t, c.last_seen_t = np.full(n, -1e9), np.full(n, -1e9), np.full(n, -1e9)
    c.travel_left = np.zeros(n)
    count = [0] * n

    def new_destination(i):
        q = home[i] + OFFS[(count[i] + i) % 4]
        count[i] += 1
        c.target[i] = q
        c.heading[i] = math.atan2(q[1] - c.pos[i][1], q[0] - c.pos[i][0])
        c.state[i] = P.WALK

    c._new_destination = new_destination
    return c


class ScriptedCrowd(H.TorchCrowd):
    """TorchCrowd with the parity test's deterministic destinations and pauses."""

    def _sample_destinations(self, mask, segs, sact):
        offs = torch.as_tensor(OFFS, dtype=self.dtype)
        idx = (self.count + torch.arange(self.h)[None, :]) % 4
        self.count = self.count + mask.long()
        return self.post + offs[idx]

    def _sample_pause(self, mask):
        lo, hi = self.cfg.pause_s
        return torch.full((self.n, self.h), lo + PAUSE_FRAC * (hi - lo), dtype=torch.float64)


def torch_crowd(scs, dtype) -> tuple[ScriptedCrowd, torch.Tensor, torch.Tensor]:
    n, h = len(scs), max(len(s["people"]) for s in scs)
    tc = ScriptedCrowd(n, h, CFG, CPU, dtype=dtype)
    tc.count = torch.zeros(n, h, dtype=torch.long)
    W = max(1, max(len(s["walls"]) for s in scs))
    segs, sact = torch.zeros(n, W, 4, dtype=dtype), torch.zeros(n, W, dtype=torch.bool)
    for e, sc in enumerate(scs):
        for j, w in enumerate(sc["walls"]):
            segs[e, j], sact[e, j] = torch.tensor(w, dtype=dtype), True
        for i, p in enumerate(sc["people"]):
            tc.kind[e, i], tc.active[e, i] = p["kind"], True
            tc.pos[e, i], tc.target[e, i] = torch.tensor(p["pos"], dtype=dtype), torch.tensor(p["target"], dtype=dtype)
            tc.post[e, i] = torch.tensor(p["home"], dtype=dtype)
            tc.heading[e, i], tc.speed[e, i], tc.commit_m[e, i] = p["heading"], p["speed"], p["commit"]
            tc.state[e, i], tc.until[e, i] = p["state"], p["until"]
    return tc, segs, sact


def run_parity(dtype):
    """Step both sides; returns (worst position error, worst heading error, state mismatches, numpy trace)."""
    compound = Compound.load(REPO / "sim/compound/compound_spec.yaml")
    crowds = [numpy_crowd(sc, compound) for sc in SCENARIOS]
    tc, segs, sact = torch_crowd(SCENARIOS, dtype)
    tracks = [robot_track(sc) for sc in SCENARIOS]
    worst_p = worst_h = 0.0
    mismatches = []
    trace = []  # per step: list over envs of (state copy, pos copy, heading copy, robot)
    for k in range(1, STEPS + 1):
        robot = np.array([tr[k] for tr in tracks])
        for c, r in zip(crowds, robot):
            c._step_once(r + O, 0.55)
        tc.step(torch.as_tensor(robot, dtype=dtype), torch.full((len(SCENARIOS),), k, dtype=torch.float64) / 10.0,
                segs, sact)
        snap = []
        for e, c in enumerate(crowds):
            m = len(c.ids)
            tp = tc.pos[e, :m].double().numpy() + O
            worst_p = max(worst_p, float(np.abs(tp - c.pos).max()))
            dh = np.abs((tc.heading[e, :m].double().numpy() - c.heading + np.pi) % (2 * np.pi) - np.pi)
            worst_h = max(worst_h, float(dh.max()))
            ts = tc.state[e, :m].numpy()
            if not np.array_equal(ts, c.state) or not np.array_equal(tc.locks_used[e, :m].numpy(), c.locks_used):
                mismatches.append((k, e, ts.tolist(), c.state.tolist()))
            snap.append((c.state.copy(), c.pos.copy(), c.heading.copy(), robot[e] + O, c.locks_used.copy()))
        trace.append(snap)
    return worst_p, worst_h, mismatches, crowds, tc, trace


def _check_parity(dtype, tol):
    worst_p, worst_h, mism, crowds, tc, _ = run_parity(dtype)
    assert not mism, f"state mismatch (step, env, torch, numpy): {mism[:5]}"
    assert worst_p < tol, worst_p
    assert worst_h < 10 * tol, worst_h
    for e, c in enumerate(crowds):
        c.finish()
        s = c.lock_summary()
        reasons = [r["end_reason"] for r in s["records"]]
        assert int(tc.n_locks[e]) == s["locks"], (e, int(tc.n_locks[e]), s["locks"])
        assert int(tc.n_commits[e]) == s["committed"], e
        assert int(tc.n_hits[e]) == s["hit"], (e, int(tc.n_hits[e]), s["hit"])
        for j, why in enumerate(H.RELEASE_REASONS):
            assert int(tc.n_released[e, j]) == reasons.count(why), (e, why, tc.n_released[e].tolist(), reasons)


def test_hunter_parity_float64():
    _check_parity(torch.float64, 1e-9)


def test_hunter_parity_float32_env_dtype():
    _check_parity(torch.float32, 1e-4)


def test_parity_window_covers_the_state_machine():
    """The scripted scenarios must actually exercise every transition the parity claims to cover."""
    _, _, _, crowds, _, trace = run_parity(torch.float64)
    seen = set()
    fov = math.radians(CFG.vision_fov_deg) / 2
    for e, c in enumerate(crowds):
        for k in range(1, len(trace)):
            (s0, p0, h0, _, lu0), (s1, p1, h1, r1, lu1) = trace[k - 1][e], trace[k][e]
            for i in range(len(c.ids)):
                a, b = s0[i], s1[i]
                if a != P.LOCKED and b == P.LOCKED:
                    seen.add("lock" if lu1[i] == 1 else "relock")
                if a == P.LOCKED and b == P.LOCKED and abs(h1[i] - h0[i]) > 1e-12:
                    seen.add("reaim")
                if b == P.COMMITTED and a != P.COMMITTED:
                    seen.add("commit")
                if a == P.COOLDOWN and b == P.WALK:
                    seen.add("cooldown_end")
                if a == P.PAUSE and b == P.WALK:
                    seen.add("pause_end")
                if a == P.WALK and b == P.PAUSE:
                    seen.add("arrive")
                if a == b == P.LOCKED and np.array_equal(p0[i], p1[i]):
                    seen.add("step_refused_at_wall")
                if b == P.COOLDOWN and a == P.COOLDOWN and not np.array_equal(p0[i], p1[i]):
                    seen.add("walk_back_to_post")
                if c.kind[i] == P.HUNTER and a == b == P.WALK and math.dist(p0[i], r1) <= CFG.trigger_m \
                        and lu0[i] < CFG.max_locks and not P.segment_blocked(p0[i], r1, c.bsegs):
                    bearing = math.atan2(r1[1] - p0[i][1], r1[0] - p0[i][0])
                    if abs((bearing - h0[i] + math.pi) % (2 * math.pi) - math.pi) > fov:
                        seen.add("outside_vision_cone")
                if c.kind[i] == P.HUNTER and b in (P.WALK, P.PAUSE) and lu1[i] >= CFG.max_locks \
                        and math.dist(p1[i], r1) <= CFG.trigger_m:
                    seen.add("max_locks_reached")
                if c.kind[i] == P.HUNTER and a in (P.WALK, P.PAUSE) and b == a and math.dist(p0[i], r1) <= CFG.trigger_m \
                        and P.segment_blocked(p0[i], r1, c.bsegs):
                    seen.add("line_of_sight_blocked")
                if c.kind[i] == P.WANDERER and a == P.WALK:
                    seen.add("wanderer_walk")
        c.finish()
        for r in c.lock_summary()["records"]:
            seen.add(r["end_reason"])
            if r["hit"]:
                seen.add("hit")
            elif r["t_commit"] is not None:
                seen.add("evaded")
    want = {"lock", "relock", "reaim", "commit", "passed", "timeout", "lost_sight", "cooldown_end", "pause_end",
            "arrive", "step_refused_at_wall", "walk_back_to_post", "outside_vision_cone", "max_locks_reached",
            "line_of_sight_blocked", "wanderer_walk", "hit", "evaded"}
    assert want <= seen, sorted(want - seen)


def test_intercept_point_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(300):
        h, r = rng.uniform(-10, 10, 2), rng.uniform(-10, 10, 2)
        v = rng.uniform(-2, 2, 2) * rng.integers(0, 2)
        s = float(rng.choice([rng.uniform(0.5, 2.0), float(np.hypot(*v))]))  # include |v| == speed (linear case)
        want = P.intercept_point(h, s, r, v, 6.0)
        got = H.intercept_point(torch.tensor(h), torch.tensor(s, dtype=torch.float64), torch.tensor(r),
                                torch.tensor(v), 6.0).numpy()
        assert np.abs(got - np.array(want)).max() < 1e-9, (h, s, r, v, got, want)


def test_segment_blocked_matches_reference():
    rng = np.random.default_rng(1)
    segs = rng.uniform(-10, 10, (6, 4))
    for _ in range(500):
        p, q = rng.uniform(-10, 10, 2), rng.uniform(-10, 10, 2)
        want = P.segment_blocked(p, q, segs)
        got = bool(H.seg_blocked(torch.tensor(p), torch.tensor(q), torch.tensor(segs), torch.ones(6, dtype=torch.bool)))
        assert got == want


# --- env wiring ---------------------------------------------------------------------------------------------------

def _greedy(obs):
    gc, gs = obs[:, 129], obs[:, 130]
    return torch.stack([torch.where(gc > 0.7, 1.0, -1.0), torch.zeros_like(gc), (4 * torch.atan2(gs, gc)).clamp(-1, 1)], -1)


def test_legacy_config_has_no_closed_loop_people():
    env = E.AvoidEnv(8, CPU, seed=0)
    assert env.crowd is None and E.preset("legacy") == E.EnvConfig()
    env.reset()
    _, _, _, info = env.step(torch.zeros(8, 3))
    assert {"locks", "commits", "hits"} <= set(info["episodes"])


def test_presets_place_hunters_that_lock_on():
    for name, tier in (("base", "base"), ("hard", "hard")):
        cfg = E.preset(name)
        assert cfg.crowd_tier == tier and cfg.hunter_slots >= 2
        env = E.AvoidEnv(256, CPU, seed=3, cfg=cfg)
        obs = env.reset()
        c = env.crowd
        hunters = c.active & (c.kind == P.HUNTER)
        assert hunters.sum(-1).float().mean() >= 1.5
        # posts beside the leg: 25-75 % along, post_offset_m to the side
        along = (c.post * env.goal[:, None, :]).sum(-1) / env.goal.norm(dim=-1, keepdim=True) ** 2
        side = (c.post[..., 0] * env.goal[:, None, 1] - c.post[..., 1] * env.goal[:, None, 0]).abs() / env.goal.norm(dim=-1, keepdim=True)
        ok = (along >= 0.25 - 1e-4) & (along <= 0.75 + 1e-4) & (side >= cfg.crowd_config().post_offset_m[0] - 1e-3)
        assert ok[hunters].float().mean() > 0.95
        if tier == "hard":
            assert (c.speed[hunters] >= 1.3 - 1e-5).all() and (c.commit_m[hunters] <= 3.0 + 1e-5).all()
        n = locks = commits = hits = 0
        for _ in range(400):
            obs, _, _, info = env.step(_greedy(obs))
            ep = info["episodes"]
            n += len(ep["ret"])
            locks += int((ep["locks"] > 0).sum())
            commits += int(ep["commits"].sum())
            hits += int(ep["hits"].sum())
            assert (ep["hits"] <= ep["commits"] + ep["locks"]).all()
            assert ((ep["hits"] == 0) | ep["contact"]).all()  # a hit is a contact, and contact ends the episode
        assert n > 100 and locks / n > 0.6, (name, locks / n)
        assert commits > 0 and hits / commits > 0.3  # a straight-at-the-goal robot is easy prey


def test_older_presets_are_pinned():
    # v1 / v2 were trained on exactly these settings (their config.json); hard-dense must not change them
    assert E.PRESETS["hard"] == dict(event_slots=5, max_people=8, event_probs=(0.15, 0.17, 0.10, 0.18, 0.25, 0.15),
                                     wanderer_slots=2, wanderer_p=0.5, hunter_slots=4,
                                     hunter_count_probs=(0.0, 0.15, 0.35, 0.30, 0.20), crowd_tier="hard")
    hard = E.preset("hard")
    assert hard.wanderer_start_min_m == 6.0 and hard.space_radius_m == 0.0 and hard.w_space == 0.0


def _wanderers_near(cfg, n=512, steps=150, radii=(2.0, 4.0, 12.0), seed=7):
    """Mean active wanderers within each radius of the robot per step, under the turn-to-goal driver."""
    env = E.AvoidEnv(n, CPU, seed=seed, cfg=cfg)
    obs = env.reset()
    acc = torch.zeros(len(radii), dtype=torch.float64)
    for _ in range(steps):
        obs, _, _, _ = env.step(_greedy(obs))
        c = env.crowd
        d = torch.hypot(*(c.pos - env.pos[:, None, :]).unbind(-1))
        w = c.active & (c.kind == P.WANDERER)
        acc += torch.tensor([float(((d <= r) & w).sum()) for r in radii], dtype=torch.float64)
    return (acc / (n * steps)).tolist()


def test_hard_dense_preset_crowd():
    cfg, hard = E.preset("hard-dense"), E.preset("hard")
    # the same hunters (count distribution and tier), event people and walls as "hard" ...
    for k in ("hunter_slots", "hunter_count_probs", "crowd_tier", "event_slots", "max_people", "event_probs",
              "wall_slots", "wall_probs", "goal_min_m", "goal_max_m", "max_steps"):
        assert getattr(cfg, k) == getattr(hard, k), k
    # ... plus 5 wanderer slots at p 0.45 (2.25 wanderers per episode on average vs 1), starting > 2.5 m away
    assert (cfg.wanderer_slots, cfg.wanderer_p, cfg.wanderer_start_min_m) == (5, 0.45, 2.5)
    env = E.AvoidEnv(4096, CPU, seed=1, cfg=cfg)
    env.reset()
    c = env.crowd
    assert c.h == 4 + 5
    wand = c.active & (c.kind == P.WANDERER)
    hunt = c.active & (c.kind == P.HUNTER)
    assert abs(wand.sum(-1).float().mean().item() - 2.25) < 0.1
    assert abs(hunt.sum(-1).float().mean().item() - 2.55) < 0.1  # E[n] of hunter_count_probs
    dist = torch.hypot(*(c.pos - env.pos[:, None, :]).unbind(-1))
    assert (dist[wand] > 2.5).float().mean() > 0.99 and (dist[hunt] > 6.0).float().mean() > 0.99
    # near the robot, wanderer density matches the sim2d hard tier (0.025 / 0.176 / 1.401 within 2 / 4 / 12 m under
    # the heuristic, seeds 1-32); "hard" had ~7x fewer within 4 m. Loose bounds: a short CPU run.
    w2, w4, w12 = _wanderers_near(cfg)
    assert 0.012 < w2 < 0.05 and 0.12 < w4 < 0.27 and 0.9 < w12 < 2.0, (w2, w4, w12)
    _, h4, _ = _wanderers_near(hard)
    assert h4 < 0.06 and w4 > 3 * h4, (h4, w4)


def test_hit_needs_contact_and_ends_the_episode():
    env = E.AvoidEnv(1, CPU, seed=0, cfg=E.preset("hard"))
    env.reset()
    c = env.crowd
    env.pos[0], env.yaw[0], env.vel[0] = torch.tensor([0.0, 0.0]), 0.0, torch.tensor([0.0, 0.0, 0.0])
    env.goal[0], env.pactive[0], env.sactive[0] = torch.tensor([20.0, 0.0]), False, False
    env.prev_dist[0] = 20.0
    c.active[0] = False
    c.active[0, 0], c.kind[0, 0] = True, P.HUNTER
    c.pos[0, 0], c.heading[0, 0], c.speed[0, 0] = torch.tensor([0.95, 0.0]), math.pi, 1.5
    c.state[0, 0], c.travel_left[0, 0], c.lock_hit[0, 0] = P.COMMITTED, 5.0, False
    c.n_hits[0] = 0
    _, _, d, info = env.step(torch.tensor([[-1.0, 0.0, 0.0]]))  # stand still: the committed hunter walks in
    assert d.item() and info["episodes"]["contact"].tolist() == [True]
    assert info["episodes"]["hits"].tolist() == [1]


def test_determinism_with_hunters():
    def run(seed):
        env = E.AvoidEnv(32, CPU, seed=seed, cfg=E.preset("hard"))
        g = torch.Generator().manual_seed(5)
        obs, out = env.reset(), []
        for _ in range(80):
            obs, r, _, _ = env.step(torch.rand(32, 3, generator=g) * 2 - 1)
            out.append((obs.clone(), r.clone(), env.crowd.pos.clone()))
        return out

    a, b = run(4), run(4)
    assert all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) and torch.equal(x[2], y[2]) for x, y in zip(a, b))


def test_obs_contract_unchanged_with_hunters():
    env = E.AvoidEnv(16, CPU, seed=0, cfg=E.preset("base"))
    obs = env.reset()
    assert obs.shape == (16, E.OBS_DIM) and E.OBS_DIM == 134 and E.OBS_VERSION == "avoid-v1"
    # hunters are lidar-visible: a hunter parked in front of the robot shows up in the forward sectors
    env.pactive[:] = False
    env.sactive[:] = False
    env.crowd.active[:] = False
    env.crowd.active[:, 0] = True
    env.crowd.pos[:, 0] = env.pos + 3.0 * torch.stack([torch.cos(env.yaw), torch.sin(env.yaw)], -1)
    sect = env._sectors()
    assert (sect[:, 30:34].amin(-1) < 0.3).all()


if __name__ == "__main__":
    import sys
    import time
    failed = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.2f}s)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
