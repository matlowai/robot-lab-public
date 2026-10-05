"""D49: the ambient conflict cost (rl/env.py::ambient_cost) and PPO-Lagrangian training (train.py --cost-limit).

Needs torch (skipped otherwise). Runs under pytest, or as a plain script where pytest is missing:
    PYTHONPATH=. /mnt/weights/ai/isaac/IsaacLab/.venv/bin/python tests/test_rl_lagrange.py
"""

from __future__ import annotations

import csv
import tempfile
import time
from pathlib import Path

try:
    import pytest
    torch = pytest.importorskip("torch")
except ImportError:  # plain-script mode
    pytest = None
    import torch

from benchmarks.avoidance.rl import env as E
from benchmarks.avoidance.rl.controller import load_policy


def _cost(people, pvel, ambient=None, wvel=(1.0, 0.0)):
    t = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
    amb = torch.ones(1, len(people), dtype=torch.bool) if ambient is None else torch.tensor([ambient])
    c, contact = E.ambient_cost(t([[0.0, 0.0]]), t([wvel]), t([people]), t([pvel]), amb, 5, 0.4, 0.3, 0.4)
    return float(c[0]), bool(contact[0])


def test_ambient_cost_cases():
    # nobody around, or a person walking away behind the robot: no cost
    assert _cost([[-5.0, 0.0]], [[-1.0, 0.0]]) == (0.0, False)
    # head-on 2 m ahead, closing at 2 m/s: full-depth intrusion predicted at 0.8 and 1.2 s, nothing yet at 0.4 s:
    # (1/4 + 1/8) / (31/32) = 0.387; not yet close, no contact
    c, contact = _cost([[2.0, 0.0]], [[-1.0, 0.0]])
    assert abs(c - 0.375 / 0.96875) < 1e-4 and not contact
    # the same person, but locked on (not ambient): no cost
    assert _cost([[2.0, 0.0]], [[-1.0, 0.0]], ambient=[False]) == (0.0, False)
    # standing still in the path 2.5 m ahead: walking at 1 m/s predicts an intrusion within 2 s, 0.3 m/s does not
    fast, _ = _cost([[2.5, 0.0]], [[0.0, 0.0]], wvel=(1.0, 0.0))
    slow, _ = _cost([[2.5, 0.0]], [[0.0, 0.0]], wvel=(0.3, 0.0))
    assert fast > slow == 0.0
    # touching (centres 0.8 m apart < 0.85): contact counts 1, plus the closing-speed term
    c, contact = _cost([[0.8, 0.0]], [[-1.0, 0.0]])
    assert contact and c > 2.0


def test_cost_never_touches_the_reward():
    """ambient_cost on vs off, same seed and actions: identical observations, rewards and dones; the cost is only in
    info["cost"] (and is non-zero somewhere in a hard-dense rollout)."""
    envs = [E.AvoidEnv(32, device="cpu", seed=5, cfg=E.preset("hard-dense", obs_version="avoid-v2t", ambient_cost=on))
            for on in (False, True)]
    o0, o1 = envs[0].reset(), envs[1].reset()
    gen = torch.Generator().manual_seed(1)
    total = 0.0
    for _ in range(150):
        a = torch.rand(32, 3, generator=gen) * 2 - 1
        o0, r0, d0, i0 = envs[0].step(a)
        o1, r1, d1, i1 = envs[1].step(a)
        assert torch.equal(o0, o1) and torch.equal(r0, r1) and torch.equal(d0, d1)
        assert not i0["cost"].any()
        total += float(i1["cost"].sum())
    assert total > 0


def test_lagrangian_training_smoke():
    from benchmarks.avoidance.rl import train
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run"
        train.main(["--envs", "64", "--hours", "0.004", "--out", str(out), "--device", "cpu", "--rollout", "8",
                    "--ckpt-min", "100", "--preset", "hard-dense", "--obs", "avoid-v2t", "--set", "ambient_cost=true",
                    "--squash", "clip", "--log-std-max", "0", "--cost-limit", "1e-4", "--lambda-lr", "0.5"])
        rows = list(csv.DictReader(open(out / "train.csv")))
        assert len(rows) >= 3
        lams = [float(r["lambda"]) for r in rows]
        costs = [float(r["cost_step"]) for r in rows]
        assert max(lams) > 1.0  # the cost sits far above a tiny limit, so the multiplier climbs
        for prev, lam, c in zip(lams, lams[1:], costs[1:]):  # and every move follows the sign of (cost - limit)
            assert (lam >= prev) if c > 1e-4 else (lam <= prev), (prev, lam, c)
        ck = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        assert "cost_critic" in ck and ck["lagrange"]["lambda"] == lams[-1]
        model, meta = load_policy(out / "latest.pt")  # deployment ignores the cost critic
        assert meta["obs_version"] == "avoid-v2t"


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
