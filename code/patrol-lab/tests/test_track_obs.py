"""Clone probe (D47): planner-derived observation features (rl/track_features.py, obs avoid-v2t / avoid-v2ts).

Needs torch (skipped otherwise). Runs under pytest, or as a plain script where pytest is missing:
    PYTHONPATH=. /mnt/weights/ai/isaac/IsaacLab/.venv/bin/python tests/test_track_obs.py
"""

import math
import time

import numpy as np

try:
    import pytest
    pytest.importorskip("torch")
except ImportError:  # plain-script mode
    pytest = None

from benchmarks.avoidance import sim2d
from benchmarks.avoidance.crowd import generate_live, to_scenario_file
from benchmarks.avoidance.rl import env as E
from benchmarks.avoidance.rl.controller import RLController
from benchmarks.avoidance.rl.ppo import ActorCritic
from benchmarks.avoidance.rl.track_features import TRACK_VEL_SCALE, sector_velocities

SEED, TIER, MAX_S = 121, "hard", 25.0


def _rl(version):
    return RLController(None, "command", model=ActorCritic(E.obs_spec(version)["dim"], 3, (8,)), obs_version=version)


def test_sector_velocities_take_the_nearest_return_in_each_sector():
    far, near = (6.0, 0.0), (2.0, 0.01)  # same ray, same sector
    f = sector_velocities([far, near, (0.0, 3.0)], [(1.0, 0.0), (0.0, -1.0), (0.4, 0.2)])
    k_front = (90 * E.SECTORS) // E.RAYS  # ray 90 = straight ahead
    k_left = (int(round((math.pi / 2 + E.FOV / 2) / E.RAY_STEP)) * E.SECTORS) // E.RAYS
    assert f.shape == (128,)
    assert (f[k_front], f[64 + k_front]) == (0.0, -1.0 / TRACK_VEL_SCALE)
    assert np.allclose((f[k_left], f[64 + k_left]), (0.4 / TRACK_VEL_SCALE, 0.2 / TRACK_VEL_SCALE))
    assert np.count_nonzero(f) == 3
    assert not sector_velocities([], []).any()


def test_training_env_refuses_sim2d_only_versions():
    try:
        E.AvoidEnv(2, device="cpu", cfg=E.preset("hard", obs_version="avoid-v2ts"))
    except ValueError as e:
        assert "sim2d only" in str(e)
    else:
        raise AssertionError("AvoidEnv accepted a sim2d-only obs version")


def test_track_obs_extends_avoid_v2_and_mirrors_the_planner_state():
    """In a live sim2d patrol driven by the planner: the first 198 features are exactly avoid-v2's, and the appended
    commitment is the driving planner's own (same inputs, and its state never depends on its commands)."""
    to_scenario_file(generate_live(SEED, TIER), sim2d.REPO / "data/scenarios/crowd")
    seen = {"steps": 0, "detours": 0, "moving": 0}

    class Spy:
        name = "spy"

        def __init__(self):
            self.teacher, self.v2, self.v2ts = sim2d.Heuristic(), _rl("avoid-v2"), _rl("avoid-v2ts")
            self.v2t = _rl("avoid-v2t")  # the tracker alone (no detour search): the same velocity features

        def step(self, t, pose, goal, scan):
            o2, _ = self.v2.observe(t, pose, goal, scan)
            o, _ = self.v2ts.observe(t, pose, goal, scan)
            ot, _ = self.v2t.observe(t, pose, goal, scan)
            assert ot.shape == (326,) and np.array_equal(ot, o[:326])
            cmd = self.teacher.step(t, pose, goal, scan)
            p = self.teacher.planner
            assert o.shape == (330,) and np.array_equal(o[:198], o2)
            assert o[326] == p._side and o[327] == float(p.subgoal is not None)
            seen["steps"] += 1
            seen["detours"] += int(o[327])
            seen["moving"] += int(np.abs(o[198:326]).max() > 0)
            return cmd

    sim2d.register("spy", Spy)
    sim2d.run_episode(SEED, "spy", max_s=MAX_S, live=TIER)
    assert seen["steps"] > 100 and seen["detours"] > 0 and seen["moving"] > 0


def test_env_avoid_v2t_extends_avoid_v2_bit_for_bit():
    """Same seed, same actions: avoid-v2t's first 198 features are avoid-v2's (the tracker draws no randomness at
    noise 0), so the scan, goal and velocity channels are untouched by the extra features."""
    import torch
    envs = [E.AvoidEnv(16, device="cpu", seed=3, cfg=E.preset("hard-dense", obs_version=v)) for v in ("avoid-v2", "avoid-v2t")]
    o2, ot = envs[0].reset(), envs[1].reset()
    gen = torch.Generator().manual_seed(0)
    moving = 0
    for _ in range(120):
        assert ot.shape == (16, 326) and torch.equal(ot[:, :198], o2)
        moving += int((ot[:, 198:] != 0).any())
        a = torch.rand(16, 3, generator=gen) * 2 - 1
        o2, r2, d2, _ = envs[0].step(a)
        ot, rt, dt, _ = envs[1].step(a)
        assert torch.equal(r2, rt) and torch.equal(d2, dt)
    assert moving > 0


def test_env_tracker_emulation_on_a_crossing_walker():
    """One walker crossing 4 m ahead at 1.2 m/s (+y): nothing for two scans, then the smoothed velocity (0.5, then
    0.75, 0.875 ... of the true one) in the sectors whose nearest return is the walker, in robot-frame axes."""
    import torch
    env = E.AvoidEnv(1, device="cpu", seed=0, cfg=E.preset("legacy", obs_version="avoid-v2t"))
    env.reset()
    env.pos[:], env.yaw[:], env.vel[:], env.goal[:] = 0.0, math.pi / 2, 0.0, torch.tensor([[0.0, 20.0]])
    env.segs[:], env.sactive[:] = 0.0, False
    env.pactive[:] = False
    env.pactive[0, 0] = True
    env.p0[0, 0], env.pvel[0, 0], env.t[:] = torch.tensor([-4.0, 4.0]), torch.tensor([1.2, 0.0]), 0.0
    env.tr_n[:], env.tr_v[:] = 0, 0.0
    seen = []
    for k in range(5):
        _, f = env._sense()
        f = f[0].view(2, 64) * E.TRACK_VEL_SCALE  # robot frame: heading +y, so world +x is robot -y (to the right)
        nz = (f.abs().sum(0) > 0).nonzero().flatten()
        seen.append((len(nz), (float(f[0, nz].abs().max()) if len(nz) else 0.0),
                     (float(f[1, nz].min()) if len(nz) else 0.0)))
        env.t += E.DT
    assert seen[0][0] == 0 and seen[1][0] == 0  # first sighting, first match: not trusted yet
    for k, frac in ((2, 0.75), (3, 0.875), (4, 0.9375)):
        n, vx_robot, vy_robot = seen[k]
        assert n >= 2 and vx_robot < 1e-5 and abs(vy_robot + 1.2 * frac) < 1e-4, (k, seen[k])


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.2f}s)")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
