"""CPU tests for the pure parts (reward, PPO, seeds, units). No Isaac app, no GR00T.

  /mnt/weights/ai/isaac/IsaacLab/.venv/bin/python /mnt/work/AI/robot-lab/rl/tests/test_core.py
(no pytest in either venv; this file is its own runner and prints every check with its actual values)
"""

import math
import sys

import numpy as np
import torch

sys.path.insert(0, "/mnt/work/AI/robot-lab/rl")
from gr00t_rl import seeds  # noqa: E402
from gr00t_rl.ppo import PPOConfig, ResidualActorCritic, gae, ppo_update  # noqa: E402
from gr00t_rl.reward import BOWL_RADIUS, Reading, RewardTracker, RewardWeights  # noqa: E402
from gr00t_rl.units import Units  # noqa: E402

N = 1


def rd(obj, bowl=(0.25, -0.08, 0.0), tcp=None, jaw=0.0, tilt=0.0, vel=0.0, speed=0.0):
    obj = torch.tensor([obj], dtype=torch.float32)
    tcp = obj.clone() if tcp is None else torch.tensor([tcp], dtype=torch.float32)
    return Reading(obj=obj, bowl=torch.tensor([bowl], dtype=torch.float32), bowl_tilt_deg=torch.tensor([tilt]),
                   tcp=tcp, jaw=torch.tensor([jaw]), joint_vel_max=torch.tensor([vel]), obj_speed=torch.tensor([speed]))


F = torch.tensor([False])
T_ = torch.tensor([True])


def run(tr, readings, term_at=None, success=False, dropped=False):
    total = {k: 0.0 for k in RewardTracker.KEYS}
    for t, r in enumerate(readings):
        term = T_ if t == term_at else F
        out = tr.step(r, term, T_ if (success and t == term_at) else F, T_ if (dropped and t == term_at) else F, t)
        for k, v in out.items():
            total[k] += float(v[0])
    return total


def test_good_episode_strict():
    """reach -> lift -> carry over bowl -> open -> success term: all milestones once + strict, positive shaping."""
    tr = RewardTracker(N, "cpu")
    start = rd((0.18, 0.08, 0.02), tcp=(0.18, 0.08, 0.10))
    tr.start(start)
    seq = [rd((0.18, 0.08, 0.02), tcp=(0.18, 0.08, 0.03))]                       # reach (1 cm)
    seq += [rd((0.18, 0.08, 0.02 + 0.01 * k)) for k in range(1, 7)]               # lift to +6 cm, held
    seq += [rd((0.18 + 0.07 * a / 10, 0.08 - 0.16 * a / 10, 0.08)) for a in range(1, 11)]  # carry to bowl
    seq += [rd((0.25, -0.08, 0.05), jaw=0.6, tcp=(0.25, -0.08, 0.09))]            # open over bowl
    seq += [rd((0.25, -0.08, 0.03), jaw=0.7, tcp=(0.25, -0.08, 0.09))]            # terminal tick (reading ignored)
    tot = run(tr, seq, term_at=len(seq) - 1, success=True)
    print("  good episode components:", {k: round(v, 3) for k, v in tot.items()})
    assert tot["reach"] == 0.25 and tot["lift"] == 1.0 and tot["over_bowl"] == 1.0 and tot["release"] == 2.0, tot
    assert tot["strict"] == 10.0 and bool(tr.strict[0]), tot
    assert tot["shaping"] > 0.3, tot["shaping"]


def test_push_without_lift_pays_nothing():
    """Object slid into the bowl on the table, jaw opened, success term fires: no lift => no strict, no milestones."""
    tr = RewardTracker(N, "cpu")
    tr.start(rd((0.18, 0.08, 0.02), tcp=(0.18, 0.08, 0.15)))
    seq = [rd((0.18 + 0.07 * a / 10, 0.08 - 0.16 * a / 10, 0.02), tcp=(0.30, 0.0, 0.10)) for a in range(1, 11)]
    seq += [rd((0.25, -0.08, 0.02), jaw=0.7, tcp=(0.30, 0.0, 0.10)), rd((0.25, -0.08, 0.02), jaw=0.7)]
    tot = run(tr, seq, term_at=len(seq) - 1, success=True)
    print("  push episode components:", {k: round(v, 3) for k, v in tot.items()}, "strict", bool(tr.strict[0]))
    assert tot["strict"] == 0.0 and not bool(tr.strict[0]) and tot["release"] == 0.0 and tot["over_bowl"] == 0.0
    assert abs(tot["shaping"]) < 1e-3, tot["shaping"]


def test_hover_and_regrasp_cannot_farm():
    """Lift, hover, drop outside, re-lift repeatedly: milestones are paid once; shaping telescopes (<= 0 net)."""
    tr = RewardTracker(N, "cpu")
    tr.start(rd((0.18, 0.08, 0.02), tcp=(0.18, 0.08, 0.10)))
    seq = []
    for _ in range(5):
        seq += [rd((0.18, 0.08, 0.02 + 0.01 * k)) for k in range(1, 7)]          # lift (held)
        seq += [rd((0.18, 0.08, 0.08))] * 20                                       # hover
        seq += [rd((0.18, 0.08, 0.02), tcp=(0.18, 0.08, 0.12), jaw=0.7)]          # dropped outside
    tot = run(tr, seq)
    print("  farm attempt components:", {k: round(v, 3) for k, v in tot.items()})
    assert tot["lift"] == 1.0 and tot["reach"] == 0.25 and tot["release"] == 0.0
    assert tot["shaping"] <= 1e-6, tot["shaping"]


def test_bowl_knock_and_blowup():
    tr = RewardTracker(N, "cpu")
    tr.start(rd((0.18, 0.08, 0.02), tcp=(0.18, 0.08, 0.10)))
    tot = run(tr, [rd((0.18, 0.08, 0.02), bowl=(0.29, -0.08, 0.0), tcp=(0.18, 0.08, 0.10), tilt=30.0),
                   rd((0.18, 0.08, 0.02), bowl=(0.29, -0.08, 0.0), tcp=(0.18, 0.08, 0.10), tilt=30.0, vel=50.0)])
    print("  knock+blowup components:", {k: round(v, 3) for k, v in tot.items()}, "done", bool(tr.done[0]))
    assert tot["bowl_bad"] == -2.0 and tot["blowup"] == -2.0 and bool(tr.done[0]) and bool(tr.blowup[0])


def test_strict_uses_last_valid_bowl_reading():
    """Success fires but the bowl was displaced 4 cm on the last valid tick -> task yes, strict no."""
    tr = RewardTracker(N, "cpu")
    tr.start(rd((0.18, 0.08, 0.02), tcp=(0.18, 0.08, 0.10)))
    seq = [rd((0.18, 0.08, 0.02 + 0.01 * k)) for k in range(1, 6)]
    seq += [rd((0.29, -0.08, 0.04), bowl=(0.29, -0.08, 0.0), jaw=0.7), rd((0.29, -0.08, 0.03), jaw=0.7)]
    run(tr, seq, term_at=len(seq) - 1, success=True)
    print("  displaced bowl: task", bool(tr.task[0]), "strict", bool(tr.strict[0]),
          "last shift", round(float(tr.bowl_last_shift[0]), 3))
    assert bool(tr.task[0]) and not bool(tr.strict[0])


def test_init_policy_is_base():
    torch.manual_seed(0)
    ac = ResidualActorCritic(72, 89, 2048)
    a, c, e = torch.randn(64, 72), torch.randn(64, 89), torch.randn(64, 2048)
    u, rel, logp, v, p = ac.act(a, e, c, deterministic=True)
    print(f"  init deterministic: max|u|={u.abs().max():.2e} releases={int(rel.sum())} p_release={float(p.mean()):.4f}")
    assert float(u.abs().max()) == 0.0 and int(rel.sum()) == 0 and abs(float(p.mean()) - 1 / (1 + math.e ** 3)) < 1e-6
    u, rel, *_ = ac.act(a, e, c)
    print(f"  init stochastic: std(u)={float(u.std()):.3f} (expect ~0.3) release frac={float(rel.mean()):.3f}")
    assert 0.2 < float(u.std()) < 0.4


def test_gae_terminal_and_ppo_learns_release():
    """Bandit sanity: release=1 gets +1, release=0 gets 0 -> PPO raises p(release)."""
    r = torch.tensor([[1.0], [1.0], [1.0]])
    v = torch.zeros(3, 1)
    d = torch.tensor([[0.0], [1.0], [0.0]])
    adv, ret = gae(r, v, d, torch.zeros(1), 0.5, 1.0)
    print("  gae adv:", adv.flatten().tolist())
    assert torch.allclose(adv.flatten(), torch.tensor([1.5, 1.0, 1.0]))
    torch.manual_seed(0)
    ac = ResidualActorCritic(8, 8, 0, release_bias=-1.0, critic_uses_emb=False)
    opt = torch.optim.Adam(ac.parameters(), lr=3e-3)
    p0 = None
    for it in range(30):
        x = torch.randn(512, 8)
        u, rel, logp, val, p = ac.act(x, None, x)
        p0 = float(p.mean()) if p0 is None else p0
        rew = rel.clone()
        batch = {"a_low": x, "c_low": x, "emb": None, "u": u, "rel": rel, "logp": logp,
                 "adv": rew - val, "ret": rew}
        ppo_update(ac, opt, batch, PPOConfig(epochs=4, minibatches=4))
    p1 = float(ac.act(torch.randn(512, 8), None, torch.randn(512, 8))[4].mean())
    print(f"  PPO bandit: p(release) {p0:.3f} -> {p1:.3f}")
    assert p1 > 0.8 > p0


def test_seeds_disjoint():
    tr = {seeds.train_seed(r, w) for r in range(0, 200) for w in range(0, 100)}
    ev = {seeds.eval_seed(o, w) for o in seeds.EVAL_OBJECT_ORDER for w in range(100)}
    print(f"  {len(tr)} training seeds, {len(ev)} eval seeds, overlap {len(tr & ev)}; "
          f"any train seed in eval range: {any(seeds.is_eval_seed(s) for s in tr)}")
    assert not (tr & ev) and not any(seeds.is_eval_seed(s) for s in tr) and all(seeds.is_eval_seed(s) for s in ev)
    a, b = seeds.mix_seed(1, 2, 3), seeds.mix_seed(1, 2, 4)
    assert a != b and a == seeds.mix_seed(1, 2, 3) and 0 <= a < 2 ** 63


def test_units_roundtrip():
    U = Units()
    q = np.array([[0.1, -0.5, 0.7, 0.2, -0.3, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.785]], np.float32)
    back = U.from_units(U.to_units(q))
    print(f"  units roundtrip max err {np.abs(back - q).max():.2e}; jaw 0.785 rad = {U.to_units(q)[1, 5]:.1f} units; "
          f"0.5 rad = {U.jaw_rad_to_units(0.5):.1f} units")
    assert np.abs(back - q).max() < 1e-5 and abs(U.to_units(q)[1, 5] - 50.0) < 0.2


if __name__ == "__main__":
    fails = 0
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
