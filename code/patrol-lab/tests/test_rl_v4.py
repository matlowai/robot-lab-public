"""RL v4 pieces (D46): time-to-collision term, clipped-Gaussian policy, bounds loss, warm start, BC teacher actions.

Needs torch (skipped otherwise). Runs under pytest, or as a plain script where pytest is missing:
    PYTHONPATH=. /mnt/weights/ai/isaac/IsaacLab/.venv/bin/python tests/test_rl_v4.py
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    import pytest
    torch = pytest.importorskip("torch")
except ImportError:  # plain-script mode
    pytest = None
    import torch

from benchmarks.avoidance.rl import env as E
from benchmarks.avoidance.rl.controller import RLController, load_policy
from benchmarks.avoidance.rl.ppo import ActorCritic, PPOConfig, ppo_update


def _ttc(pos, vel, ppl, pvel, active=None, r=1.0):
    t = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
    act = torch.ones(1, len(ppl), dtype=torch.bool) if active is None else torch.tensor([active])
    return float(E.min_time_to_collision(t([pos]), t([vel]), t([ppl]), t([pvel]), act, r)[0])


def test_ttc_analytic_cases():
    # head-on: 10 m apart, closing at 2 m/s, radius 1 -> 4.5 s
    assert abs(_ttc([0, 0], [1, 0], [[10, 0]], [[-1, 0]]) - 4.5) < 1e-5
    # already inside -> 0
    assert _ttc([0, 0], [0, 0], [[0.5, 0]], [[0, 0]]) == 0.0
    # parallel, same velocity -> never
    assert math.isinf(_ttc([0, 0], [1, 0], [[0, 3]], [[1, 0]]))
    # moving apart -> never
    assert math.isinf(_ttc([0, 0], [0, 0], [[3, 0]], [[1, 0]]))
    # miss by 2 m with radius 1 -> never
    assert math.isinf(_ttc([0, 0], [1, 0], [[10, 2]], [[0, 0]]))
    # inactive people are ignored; the nearest-in-time active one counts
    assert abs(_ttc([0, 0], [1, 0], [[3, 0], [10, 0]], [[0, 0], [0, 0]], active=[False, True]) - 9.0) < 1e-5
    # a person walking into a stopped robot still has a finite TTC (the robot is expected to move away)
    assert abs(_ttc([0, 0], [0, 0], [[5, 0]], [[-1, 0]]) - 4.0) < 1e-5


def test_ttc_penalty_shape():
    t = torch.tensor([0.0, 1.5, 3.0, 5.0, float("inf")])
    p = E.ttc_penalty(t, 3.0, 0.1)
    assert torch.allclose(p, torch.tensor([0.1, 0.025, 0.0, 0.0, 0.0]))
    assert torch.equal(E.ttc_penalty(t, 3.0, 0.0), torch.zeros(5))


def test_ttc_off_is_bit_identical_and_on_only_subtracts():
    torch.manual_seed(0)
    a_seq = [torch.rand(64, 3) * 2 - 1 for _ in range(60)]
    def roll(**over):
        env = E.AvoidEnv(64, "cpu", seed=3, cfg=E.preset("hard", obs_version="avoid-v2", **over))
        env.reset()
        out = []
        for a in a_seq:
            obs, r, done, info = env.step(a)
            out.append((obs.clone(), r.clone(), done.clone()))
        return out, info
    base, _ = roll()
    on, info = roll(w_ttc=0.1)
    for (o0, r0, d0), (o1, r1, d1) in zip(base, on):
        assert torch.equal(o0, o1) and torch.equal(d0, d1)  # the term changes rewards only
        assert bool((r1 <= r0 + 1e-6).all())
    assert any(bool((r1 < r0 - 1e-6).any()) for (_, r0, _), (_, r1, _) in zip(base, on))
    assert "ttc_frac" in info["episodes"] and "r_ttc" in info["episodes"]


def test_clip_policy_env_action_and_deterministic():
    m = ActorCritic(8, 3, (16, 16), squash="clip", log_std_max=0.0)
    u = torch.tensor([[-3.0, 0.2, 5.0]])
    assert torch.equal(m.env_action(u), torch.tensor([[-1.0, 0.2, 1.0]]))
    with torch.no_grad():
        m.log_std.fill_(2.0)
    assert float(m.dist(torch.zeros(1, 8)).stddev.max()) == 1.0  # clamped at exp(log_std_max)
    t = ActorCritic(8, 3, (16, 16))
    assert t.squash == "tanh" and torch.allclose(t.env_action(u), torch.tanh(u))


def test_bound_loss_pulls_mean_back_inside():
    torch.manual_seed(0)
    m = ActorCritic(4, 3, (16,), squash="clip")
    with torch.no_grad():
        m.actor[-1].bias.fill_(3.0)  # every mean far outside [-1, 1]
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    B = 256
    nobs = torch.randn(B, 4)
    with torch.no_grad():
        d = m.dist(nobs)
        u = d.sample()
        logp = d.log_prob(u).sum(-1)
    before = float(m.dist(nobs).mean.mean())
    cfg = PPOConfig(bound_coef=1.0, ent_coef=0.0, epochs=4, minibatches=2, target_kl=1e9)
    ppo_update(m, opt, cfg, nobs, u, logp, torch.zeros(B), torch.zeros(B), torch.zeros(B))
    assert float(m.dist(nobs).mean.mean()) < before - 0.05


def test_critic_only_update_leaves_actor_untouched():
    torch.manual_seed(1)
    m = ActorCritic(4, 3, (16,), squash="clip")
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    B = 128
    nobs = torch.randn(B, 4)
    with torch.no_grad():
        d = m.dist(nobs)
        u = d.sample()
        logp = d.log_prob(u).sum(-1)
    actor0 = [p.clone() for p in m.actor.parameters()] + [m.log_std.clone()]
    critic0 = [p.clone() for p in m.critic.parameters()]
    ppo_update(m, opt, PPOConfig(), nobs, u, logp, torch.zeros(B), torch.randn(B), torch.randn(B), train_actor=False)
    assert all(torch.equal(a, b) for a, b in zip(actor0, list(m.actor.parameters()) + [m.log_std]))
    assert any(not torch.equal(a, b) for a, b in zip(critic0, m.critic.parameters()))


def test_controller_honours_clip_checkpoint():
    from benchmarks.avoidance.rl.train import save_ckpt
    m = ActorCritic(E.OBS_DIM_V2, 3, (256, 256), squash="clip", log_std_max=0.0)
    with torch.no_grad():
        m.actor[-1].bias.copy_(torch.tensor([5.0, -0.3, 0.0]))  # vx mean far past the limit
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.pt"
        save_ckpt(p, m, torch.optim.Adam(m.parameters()), 1, 1, E.preset("hard", obs_version="avoid-v2"), PPOConfig(),
                  (256, 256))
        model, ck = load_policy(p)
        assert ck["squash"] == "clip" and model.squash == "clip" and model.log_std_max == 0.0
        ctrl = RLController(p)
        vx, vy, wz, _ = ctrl.step(0.0, (0.0, 0.0, 0.0), (10.0, 0.0), [])
        assert abs(vx - 1.0) < 1e-6  # clamp(5) = 1 -> full speed
        assert abs(vy - E.scale_action(np.array([1.0, -0.3, 0.0]))[1]) < 0.05  # bias dominates a fresh net


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
