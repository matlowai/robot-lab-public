"""The RL avoidance env and its sim2d controller: shapes, determinism, train/deploy feature parity, reward signs.

Needs torch (skipped otherwise). Runs under pytest, or as a plain script where pytest is missing:
    PYTHONPATH=. /mnt/weights/ai/isaac/IsaacLab/.venv/bin/python tests/test_rl_env.py
"""

import math
import tempfile
from pathlib import Path

import numpy as np

try:
    import pytest
    torch = pytest.importorskip("torch")
except ModuleNotFoundError:  # plain-script mode
    pytest = None
    import torch

from benchmarks.avoidance import sim2d
from benchmarks.avoidance.rl import env as E
from benchmarks.avoidance.rl.controller import RLController
from benchmarks.avoidance.rl.ppo import ActorCritic, PPOConfig

CPU = torch.device("cpu")


def _set_scene(env, pose, people=(), segs=(), vel=(0.0, 0.0, 0.0), goal=(20.0, 0.0), person_vel=None):
    """Overwrite env 0's state with a hand-made scene (people as (x, y), segments as (ax, ay, bx, by))."""
    env.pos[0] = torch.tensor(pose[:2])
    env.yaw[0] = pose[2]
    env.vel[0] = torch.tensor(vel)
    env.goal[0] = torch.tensor(goal)
    env.t[0] = 0.0
    env.steps[0] = 0
    env.pactive[0] = False
    env.sactive[0] = False
    for i, p in enumerate(people):
        env.p0[0, i] = torch.tensor(p)
        env.pvel[0, i] = torch.tensor(person_vel[i] if person_vel else (0.0, 0.0))
        env.pactive[0, i] = True
    for i, s in enumerate(segs):
        env.segs[0, i] = torch.tensor(s)
        env.sactive[0, i] = True
    env.prev_dist[0] = float(torch.linalg.norm(env.goal[0] - env.pos[0]))
    a0 = torch.tensor([2 * vel[0] / E.VX_MAX - 1, vel[1] / E.VY_MAX, vel[2] / E.WZ_MAX])
    env.prev_act[0] = a0


def _sim2d_sectors(pose, people, segs):
    scan = sim2d.lidar(pose, np.array(segs, dtype=np.float64).reshape(-1, 4), np.array(people, dtype=np.float64).reshape(-1, 2))
    return E.sectorize(E.ranges_from_points(scan)), scan


SCENES = [
    ((0.0, 0.0, 0.0), [(4.0, 0.2), (6.0, -1.5), (2.0, 3.0)], [(8.0, -5.0, 8.0, 5.0), (-3.0, 2.5, 10.0, 2.5)]),
    ((1.3, -0.7, 2.1), [(-1.0, 2.0), (0.5, 4.5), (-4.0, -1.0), (3.0, 0.0)], [(-6.0, -6.0, 6.0, -6.0), (2.0, 1.0, 2.5, 4.0)]),
    ((5.0, 5.0, -2.8), [], [(0.0, 0.0, 0.0, 10.0)]),
    ((0.0, 0.0, 1.0), [(0.3, 0.9)], []),  # person inside the 0.65 m dead zone on some rays
]


def test_sector_table_covers_every_ray_once():
    of = (np.arange(E.RAYS) * E.SECTORS) // E.RAYS
    assert set(np.unique(E.SECTOR_TABLE)) == set(range(E.RAYS))
    for k in range(E.SECTORS):
        assert set(E.SECTOR_TABLE[k]) == set(np.nonzero(of == k)[0])


def test_shapes_and_ranges():
    env = E.AvoidEnv(16, CPU, seed=0)
    obs = env.reset()
    assert obs.shape == (16, E.OBS_DIM) and obs.dtype == torch.float32
    a = torch.zeros(16, 3)
    obs, r, d, info = env.step(a)
    assert obs.shape == (16, E.OBS_DIM) and r.shape == (16,) and d.shape == (16,) and d.dtype == torch.bool
    assert info["final_obs"].shape == obs.shape
    assert (obs[:, :128] >= 0).all() and (obs[:, :128] <= 1).all()
    assert (obs[:, 128] >= 0).all() and (obs[:, 128] <= 1).all()
    assert torch.allclose(obs[:, 129] ** 2 + obs[:, 130] ** 2, torch.ones(16), atol=1e-5)


def test_determinism_with_fixed_seed():
    def run(seed):
        env = E.AvoidEnv(64, CPU, seed=seed)
        g = torch.Generator().manual_seed(123)
        obs, out = env.reset(), []
        for _ in range(60):
            a = torch.rand(64, 3, generator=g) * 2 - 1
            obs, r, d, _ = env.step(a)
            out.append((obs.clone(), r.clone(), d.clone()))
        return out

    a, b, c = run(7), run(7), run(8)
    assert all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) and torch.equal(x[2], y[2]) for x, y in zip(a, b))
    assert not all(torch.equal(x[0], y[0]) for x, y in zip(a, c))


def test_lidar_parity_env_vs_controller_path():
    env = E.AvoidEnv(1, CPU, seed=0)
    env.reset()
    for pose, people, segs in SCENES:
        _set_scene(env, pose, people, segs)
        got = env._sectors()[0].double().numpy()
        want, scan = _sim2d_sectors(pose, people, segs)
        assert np.abs(got - want).max() < 1e-4, (pose, np.abs(got - want).max())
        assert (want < 1).any() or not (people or segs)


def test_lidar_parity_random_scenes():
    rng = np.random.default_rng(0)
    env = E.AvoidEnv(1, CPU, seed=0)
    env.reset()
    worst = 0.0
    for _ in range(40):
        pose = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-math.pi, math.pi))
        people = [tuple(rng.uniform(-10, 10, 2)) for _ in range(rng.integers(0, 6))]
        segs = [tuple(rng.uniform(-12, 12, 4)) for _ in range(rng.integers(0, 3))]
        _set_scene(env, pose, people, segs)
        got = env._sectors()[0].double().numpy()
        want, _ = _sim2d_sectors(pose, people, segs)
        worst = max(worst, np.abs(got - want).max())
    assert worst < 1e-4, worst


def _tiny_ckpt(tmp: Path, obs_version: str = E.OBS_VERSION) -> Path:
    from benchmarks.avoidance.rl.train import save_ckpt
    torch.manual_seed(0)
    model = ActorCritic(E.obs_spec(obs_version)["dim"], E.ACT_DIM)
    opt = torch.optim.Adam(model.parameters())
    p = tmp / f"tiny-{obs_version}.pt"
    save_ckpt(p, model, opt, 0, 0, E.EnvConfig(obs_version=obs_version), PPOConfig(), (256, 256))
    return p


def test_full_observation_parity_env_vs_controller():
    """Same state -> the controller builds the env's exact 134-vector (scan now/prev, goal, velocity)."""
    with tempfile.TemporaryDirectory() as d:
        ctrl = RLController(_tiny_ckpt(Path(d)))
    env = E.AvoidEnv(1, CPU, seed=0)
    env.reset()
    pose, people, segs = SCENES[1]
    goal = (8.0, -3.0)
    _set_scene(env, pose, people, segs, vel=(0.0, 0.0, 0.0), goal=goal)
    env.prev_sect = env._sectors()
    want_first = E.build_obs(env.prev_sect, env.prev_sect, env._goal_feat(), env.vel)[0].numpy()
    _, scan = _sim2d_sectors(pose, people, segs)
    obs0, _ = ctrl.observe(0.0, pose, goal, scan)
    assert np.abs(obs0 - want_first).max() < 1e-4
    # one step later with the scene moved: prev must be the first scan, vel what the controller tracked
    ctrl.vel = np.array([0.4, -0.1, 0.3])
    pose2 = (pose[0] + 0.05, pose[1] - 0.02, pose[2] + 0.03)
    people2 = [(x + 0.1, y) for x, y in people]
    _set_scene(env, pose2, people2, segs, vel=tuple(ctrl.vel), goal=goal)
    now = env._sectors()
    want = E.build_obs(now, torch.as_tensor(obs0[None, :64]), env._goal_feat(), env.vel)[0].numpy()
    _, scan2 = _sim2d_sectors(pose2, people2, segs)
    obs1, _ = ctrl.observe(0.1, pose2, goal, scan2)
    assert np.abs(obs1 - want).max() < 1e-4


def test_moving_toward_goal_is_rewarded_and_away_is_not():
    env = E.AvoidEnv(1, CPU, seed=0)
    env.reset()
    _set_scene(env, (0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0), goal=(20.0, 0.0))
    _, r, d, _ = env.step(torch.tensor([[1.0, 0.0, 0.0]]))
    assert r.item() > 0 and not d.item()
    _set_scene(env, (0.0, 0.0, math.pi), vel=(1.0, 0.0, 0.0), goal=(20.0, 0.0))
    _, r, d, _ = env.step(torch.tensor([[1.0, 0.0, 0.0]]))
    assert r.item() < 0 and not d.item()


def test_contact_terminates_with_a_big_penalty():
    env = E.AvoidEnv(1, CPU, seed=0)
    env.reset()
    # person 0.9 m ahead (clearance 0.05 m) walking at the robot, which walks at it
    _set_scene(env, (0.0, 0.0, 0.0), people=[(0.9, 0.0)], person_vel=[(-1.0, 0.0)], vel=(1.0, 0.0, 0.0))
    _, r, d, info = env.step(torch.tensor([[1.0, 0.0, 0.0]]))
    assert d.item() and info["terminated"].item() and r.item() <= -9.0
    assert info["episodes"]["contact"].tolist() == [True]


def test_wall_contact_terminates_and_success_pays():
    env = E.AvoidEnv(1, CPU, seed=0)
    env.reset()
    _set_scene(env, (0.0, 0.0, 0.0), segs=[(0.62, -2.0, 0.62, 2.0)], vel=(1.0, 0.0, 0.0))
    _, r, d, info = env.step(torch.tensor([[1.0, 0.0, 0.0]]))
    assert d.item() and info["episodes"]["wall"].tolist() == [True] and r.item() <= -9.0
    _set_scene(env, (0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0), goal=(0.45, 0.0))
    _, r, d, info = env.step(torch.tensor([[1.0, 0.0, 0.0]]))
    assert d.item() and info["episodes"]["success"].tolist() == [True] and r.item() > 9.0


def test_controller_api():
    with tempfile.TemporaryDirectory() as d:
        ctrl = RLController(_tiny_ckpt(Path(d)))
    pose, people, segs = SCENES[0]
    _, scan = _sim2d_sectors(pose, people, segs)
    out = ctrl.step(0.0, pose, (15.0, 0.0), scan)
    assert isinstance(out, tuple) and len(out) == 4
    vx, vy, wz, status = out
    assert all(isinstance(v, float) for v in (vx, vy, wz))
    assert 0.0 <= vx <= 1.0 and -0.5 <= vy <= 0.5 and -1.0 <= wz <= 1.0
    assert status in {"moving", "arrived", "blocked", "turning"}
    assert ctrl.step(0.1, (14.8, 0.1, 0.0), (15.0, 0.0), [])[3] == "arrived"


# --- personal space (D44) --------------------------------------------------------------------------------------------

def test_personal_space_penalty_shape():
    R, w = 1.0, 0.2
    f = lambda c: E.personal_space_penalty(c, R, w)  # noqa: E731
    # zero outside the radius (and with nobody around), full weight at contact, never negative
    for c in (1.0, 1.001, 1.5, 12.0, math.inf):
        assert f(c) == 0.0
    assert f(0.0) == w and f(-0.2) == w
    cs = np.linspace(-0.2, 1.3, 151)
    vals = [f(float(c)) for c in cs]
    assert all(v >= 0.0 for v in vals)
    # monotone: strictly decreasing with clearance inside (0, R), flat outside
    inside = [v for c, v in zip(cs, vals) if 0.0 < c < R]
    assert all(a > b for a, b in zip(inside, inside[1:]))
    # smooth at the edge: the value and slope go to 0 at c = R
    assert f(R - 0.01) < 2e-4 * w and (f(R - 0.02) - f(R - 0.01)) / 0.01 < 0.01
    # the tensor path agrees with the scalar path; radius 0 or weight 0 turns it off
    t = E.personal_space_penalty(torch.tensor(cs, dtype=torch.float64), R, w)
    assert np.allclose(t.numpy(), vals)
    assert E.personal_space_penalty(0.2, 0.0, w) == 0.0 and E.personal_space_penalty(0.2, R, 0.0) == 0.0
    assert torch.equal(E.personal_space_penalty(torch.tensor([0.2, math.inf]), 0.0, w), torch.zeros(2))


def _space_term_at(env, clearance):
    """Robot standing still at the origin (action = zero command), one standing person straight ahead at the given
    clearance (None = nobody), no closed-loop people: returns (reward, the step's space term, its near-miss term)."""
    people = [(clearance + E.PERSON_R + E.SPOT_R, 0.0)] if clearance is not None else ()
    _set_scene(env, (0.0, 0.0, 0.0), people=people)
    if env.crowd is not None:
        env.crowd.active[0] = False
    env.ep_terms[0] = 0.0
    _, r, d, _ = env.step(torch.tensor([[-1.0, 0.0, 0.0]]))
    assert not d.item()
    k = E.REWARD_TERMS
    return r.item(), env.ep_terms[0, k.index("space")].item(), env.ep_terms[0, k.index("near")].item()


def test_personal_space_term_in_the_env():
    env = E.AvoidEnv(1, CPU, seed=0, cfg=E.preset("hard-dense"))
    env.reset()
    cfg = env.cfg
    assert cfg.space_radius_m == 1.0 and cfg.w_space == 0.2
    r_free, s_free, _ = _space_term_at(env, None)
    assert s_free == 0.0 and abs(r_free + cfg.time_penalty) < 1e-6  # standing still, alone: just the time cost
    # outside the radius: no penalty, same reward as alone
    for c in (1.05, 2.0, 5.0):
        r, s, n = _space_term_at(env, c)
        assert s == 0.0 and n == 0.0 and abs(r - r_free) < 1e-6
    # inside: negative, equal to personal_space_penalty, growing as the person gets closer
    prev_r, prev_s = r_free, 0.0
    for c in (0.9, 0.7, 0.5, 0.35, 0.2, 0.05):
        r, s, n = _space_term_at(env, c)
        want = E.personal_space_penalty(c, cfg.space_radius_m, cfg.w_space)
        assert s < 0 and abs(-s - want) < 1e-5, (c, s, want)
        assert abs(r - (r_free + s + n)) < 1e-5  # its own term, on top of the near-miss ramp
        assert s < prev_s and r < prev_r
        prev_r, prev_s = r, s
    # it is separate from the contact penalty: contact still terminates with the big one
    _set_scene(env, (0.0, 0.0, 0.0), people=[(0.80, 0.0)])
    env.crowd.active[0] = False
    env.ep_terms[0] = 0.0
    _, r, d, info = env.step(torch.tensor([[-1.0, 0.0, 0.0]]))
    assert d.item() and info["episodes"]["contact"].tolist() == [True]
    assert info["episodes"]["r_contact"].item() == -cfg.contact_penalty
    assert abs(info["episodes"]["r_space"].item() + cfg.w_space) < 1e-6
    # the older presets have no personal-space term (so v1 / v2 runs stay reproducible)
    for name in ("legacy", "base", "hard"):
        assert E.preset(name).w_space == 0.0 and E.preset(name).space_radius_m == 0.0
        env = E.AvoidEnv(1, CPU, seed=0, cfg=E.preset(name))
        env.reset()
        assert _space_term_at(env, 0.5)[1] == 0.0


def test_reward_terms_add_up_to_the_return():
    env = E.AvoidEnv(64, CPU, seed=2, cfg=E.preset("hard-dense"))
    env.reset()
    g = torch.Generator().manual_seed(0)
    seen = 0
    for _ in range(300):
        _, _, _, info = env.step(torch.rand(64, 3, generator=g) * 2 - 1)
        ep = info["episodes"]
        if len(ep["ret"]):
            total = sum(ep[f"r_{k}"] for k in E.REWARD_TERMS)
            assert torch.allclose(total, ep["ret"], atol=1e-3), (total, ep["ret"])
            assert ((ep["space_frac"] >= 0) & (ep["space_frac"] <= 1)).all()
            assert (ep["r_space"] <= 0).all()
            seen += len(ep["ret"])
    assert seen > 20


def test_train_set_and_ramp():
    import csv
    import json

    from benchmarks.avoidance.rl import train
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run"
        rc = train.main(["--envs", "8", "--hours", "0.0006", "--out", str(out), "--device", "cpu", "--print-every",
                         "100000", "--preset", "hard-dense", "--obs", V2, "--set", "wanderer_p=0.3",
                         "--ramp", "w_space=0:0.2:0.8"])
        assert rc == 0
        cfg = json.loads((out / "config.json").read_text())
        # the recorded config is the preset's target (plus --set), the ramp is in the args
        assert cfg["env_cfg"]["w_space"] == 0.2 and cfg["env_cfg"]["wanderer_p"] == 0.3
        assert cfg["args"]["ramp"] == ["w_space=0:0.2:0.8"]
        rows = list(csv.DictReader(open(out / "train.csv")))
        assert len(rows) > 3
        w = [float(r["cfg_w_space"]) for r in rows]
        assert w[0] == 0.0 and all(0.0 <= x <= 0.2 + 1e-9 for x in w) and all(a <= b for a, b in zip(w, w[1:]))
        assert all(float(r["r_space"]) == 0.0 for r in rows if float(r["cfg_w_space"]) == 0.0 and r["episodes"] != "0")
        ck = torch.load(out / "latest.pt", weights_only=False)
        assert ck["env_cfg"]["w_space"] == 0.2
        for bad in (["--ramp", "wanderer_slots=1:0:1"], ["--ramp", "w_space=0:0.5:0.2"], ["--set", "nope=1"]):
            try:
                train.main(["--envs", "8", "--hours", "0", "--out", str(Path(d) / "bad"), "--device", "cpu", *bad])
                raise AssertionError(f"{bad} must be rejected")
            except SystemExit as e:
                assert e.code == 2


# --- avoid-v2: scans at t, t-0.3, t-0.6 ----------------------------------------------------------------------------

V2 = E.OBS_VERSION_V2


def test_v2_layout_and_v1_default():
    assert E.EnvConfig().obs_version == "avoid-v1" and E.OBS_DIM == 134
    env = E.AvoidEnv(4, CPU, seed=0, cfg=E.preset("hard", obs_version=V2))
    obs = env.reset()
    assert obs.shape == (4, E.OBS_DIM_V2) == (4, 198) and env.obs_dim == 198
    assert torch.equal(obs[:, 0:64], obs[:, 64:128]) and torch.equal(obs[:, 0:64], obs[:, 128:192])  # first step
    lay = E.OBS_LAYOUT_V2
    assert lay["vel"] == [195, 198] and lay["goal_dist"] == [192, 193] and lay["scan_lags_s"] == [0.3, 0.6]


def test_v2_env_history_follows_the_scan_record():
    """Every v2 observation's lag channels are the scan_now of 3 and 6 steps earlier in the same episode (the
    episode's first scan while the history is shorter), across auto-resets, and in final_obs too."""
    n = 16
    env = E.AvoidEnv(n, CPU, seed=1, cfg=E.preset("hard", obs_version=V2))
    obs = env.reset()
    rec = [[obs[e, :64].clone()] for e in range(n)]
    g = torch.Generator().manual_seed(2)
    resets = 0

    def lag(r, k):
        return r[max(0, len(r) - 1 - k)]

    for _ in range(90):
        obs, _, done, info = env.step(torch.rand(n, 3, generator=g) * 2 - 1)
        fin = info["final_obs"]
        for e in range(n):
            rec[e].append(fin[e, :64].clone())
            assert torch.equal(fin[e, 64:128], lag(rec[e], 3)) and torch.equal(fin[e, 128:192], lag(rec[e], 6))
            if done[e]:
                resets += 1
                rec[e] = [obs[e, :64].clone()]
            assert torch.equal(obs[e, 64:128], lag(rec[e], 3)) and torch.equal(obs[e, 128:192], lag(rec[e], 6))
    assert resets > 0


def _v2_rollout_parity(jitter: float, seed: int = 0):
    """Drive a 1-env v2 env through a moving scene; feed the controller the same states as sim2d scans at
    timestamps 0.1 k (+- jitter). Returns the worst |env obs - controller obs|."""
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as d:
        ctrl = RLController(_tiny_ckpt(Path(d), V2))
    assert ctrl.obs_version == V2
    env = E.AvoidEnv(1, CPU, seed=0, cfg=E.preset("legacy", obs_version=V2))
    env.reset()
    people = [(6.0, 2.5), (9.0, -3.0), (4.0, -2.0), (12.0, 1.0)]
    pvel = [(-0.9, -0.2), (-0.5, 0.6), (0.3, 0.8), (-1.2, 0.0)]
    segs = [(-3.0, 4.0, 15.0, 4.0), (14.0, -5.0, 14.0, 3.0)]
    goal = (20.0, 0.0)
    _set_scene(env, (0.0, 0.0, 0.2), people, segs, vel=(0.3, 0.0, 0.0), goal=goal, person_vel=pvel)
    env.scan_hist[:] = env._sectors()[:, None]

    def scene_scan():
        pp, act = env.people()
        pts = pp[0][act[0]].double().numpy()
        wall = env.segs[0][env.sactive[0]].double().numpy()
        pose = (float(env.pos[0, 0]), float(env.pos[0, 1]), float(env.yaw[0]))
        return pose, sim2d.lidar(pose, wall.reshape(-1, 4), pts.reshape(-1, 2))

    want = env._obs(env.scan_hist, env._goal_feat(), env.vel)[0].numpy()
    pose, scan = scene_scan()
    ctrl.vel = env.vel[0].double().numpy()
    got, _ = ctrl.observe(0.0, pose, goal, scan)
    worst = float(np.abs(got - want).max())
    g = torch.Generator().manual_seed(seed)
    moved = 0.0
    for k in range(1, 40):
        a = torch.tensor([[-0.2, 0.0, 0.0]]) + 0.3 * (torch.rand(1, 3, generator=g) - 0.5)
        obs, _, done, _ = env.step(a)
        assert not done.item(), k
        pose, scan = scene_scan()
        ctrl.vel = env.vel[0].double().numpy()
        t = round(0.1 * k, 10) + (rng.uniform(-jitter, jitter) if jitter else 0.0)
        got, _ = ctrl.observe(t, pose, goal, scan)
        worst = max(worst, float(np.abs(got - obs[0].numpy()).max()))
        moved = max(moved, float(np.abs(got[128:192] - got[:64]).max()))
        if k % 7 == 0:  # a repeated t (sim2d arrival step / Isaac duplicate tick) changes nothing
            again, _ = ctrl.observe(t, pose, goal, scan)
            assert np.array_equal(again, got)
    assert moved > 0.02  # the lag channels carry real motion, not copies of scan_now
    return worst


def test_v2_observation_parity_env_vs_controller():
    assert _v2_rollout_parity(0.0) < 1e-4


def test_v2_observation_parity_with_jittered_timestamps():
    """Isaac-like ticks: 0.1 s +- 20 ms. Up to +-25 ms of jitter the scan closest to t - 0.3 / t - 0.6 is always the
    one 3 / 6 ticks back, so the controller reproduces the env exactly. (With more jitter it still takes the scan
    closest in time, which is the intended meaning; it just no longer equals the tick count.)"""
    for seed in range(3):
        assert _v2_rollout_parity(0.02, seed) < 1e-4


def test_scan_history_picks_by_timestamp():
    from benchmarks.avoidance.rl.controller import ScanHistory
    h = ScanHistory(keep_s=1.1)
    ts = [0.0, 0.1, 0.22, 0.29, 0.41, 0.5, 0.63, 0.7]
    for i, t in enumerate(ts):
        h.push(t, np.full(2, float(i)))
    h.push(0.7, np.full(2, 99.0))  # repeated t: ignored (the first scan at a t is kept)
    assert h.at(0.7)[0] == 7
    assert h.at(0.7 - 0.3)[0] == 4  # 0.41 is closest to 0.4
    assert h.at(0.7 - 0.6)[0] == 1  # 0.1
    assert h.at(-5.0)[0] == 0  # not enough history: the oldest
    tie = ScanHistory(keep_s=1.1)
    for i, t in enumerate((0.0, 0.25, 0.75)):
        tie.push(t, np.full(2, float(i)))
    assert tie.at(0.5)[0] == 1  # 0.25 and 0.75 are equally close: the older wins
    for t in np.arange(0.8, 3.0, 0.1):
        h.push(float(t), np.full(2, float(t)))
    assert h.ts[0] > 1.6 and len(h.ts) <= 14  # bounded: nothing older than needed for a 0.6 s lag is kept
    assert abs(h.at(2.9 - 0.6)[0] - 2.3) < 1e-9
    h.push(0.05, np.full(2, -1.0))  # time went backwards: a new run
    assert h.ts == [0.05] and h.at(-0.55)[0] == -1.0


def test_checkpoints_select_their_obs_builder():
    from benchmarks.avoidance.rl.controller import load_policy, register_rl
    pose, people, segs = SCENES[0]
    _, scan = _sim2d_sectors(pose, people, segs)
    with tempfile.TemporaryDirectory() as d:
        p1, p2 = _tiny_ckpt(Path(d)), _tiny_ckpt(Path(d), V2)
        c1, c2 = RLController(p1), RLController(p2)
        assert (c1.obs_version, c2.obs_version) == ("avoid-v1", V2)
        assert c1.observe(0.0, pose, (15.0, 0.0), scan)[0].shape == (134,)
        assert c2.observe(0.0, pose, (15.0, 0.0), scan)[0].shape == (198,)
        assert len(c2.step(0.1, pose, (15.0, 0.0), scan)) == 4
        # a shared model without a tag: the input width decides
        m2, ck2 = load_policy(p2)
        assert ck2["obs_version"] == V2 and ck2["obs_layout"] == E.OBS_LAYOUT_V2
        assert RLController(None, model=m2).obs_version == V2
        register_rl(p2, name="rl_v2_test")
        assert sim2d.CONTROLLERS.pop("rl_v2_test")().obs_version == V2
        bad = torch.load(p2, weights_only=False)
        bad["obs_dim"] = 134
        torch.save(bad, Path(d) / "bad.pt")
        try:
            load_policy(Path(d) / "bad.pt")
            raise AssertionError("mismatched obs_dim must not load")
        except ValueError:
            pass


def test_rl_trace_is_opt_in_and_command_velocity_matches_sim2d():
    """sim2d's opt-in trace changes nothing about an RL episode, and the controller's command-integrated velocity
    (velocity_source="command", what rl/eval.py uses) equals sim2d's own velocity at every next step."""
    import json
    from benchmarks.avoidance.crowd import generate_live, to_scenario_file
    from benchmarks.avoidance.rl.controller import register_rl
    to_scenario_file(generate_live(121, "hard"), sim2d.REPO / "data/scenarios/crowd")
    with tempfile.TemporaryDirectory() as d:
        register_rl(_tiny_ckpt(Path(d), V2), name="rl_trace_test")
        try:
            plain = sim2d.run_episode(121, "rl_trace_test", max_s=20.0, live="hard")
            tp = Path(d) / "tr.npz"
            traced = sim2d.run_episode(121, "rl_trace_test", max_s=20.0, live="hard", trace_path=str(tp))
            z = np.load(tp)
        finally:
            sim2d.CONTROLLERS.pop("rl_trace_test")
    untimed = lambda r: json.dumps({k: v for k, v in r.items() if k != "ms_per_decision"}, sort_keys=True)  # noqa: E731
    assert untimed(plain) == untimed(traced)
    cols = {c: i for i, c in enumerate(json.loads(str(z["meta"]))["columns"])}
    rob = z["robot"]
    moving = rob[:-1, cols["status"]] != 2  # after an arrival step sim2d applies no command
    sim_next = rob[1:, [cols["vx"], cols["vy"], cols["wz"]]][moving]
    ctrl_est = rob[:-1, [cols["ctrl_vx"], cols["ctrl_vy"], cols["ctrl_wz"]]][moving]
    assert len(sim_next) > 100 and np.abs(sim_next - ctrl_est).max() < 1e-9


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
