"""avoid-v2t parity (D48): the training env's emulated tracker vs the real ObstacleTracker on the same scans.

Drives AvoidEnv (avoid-v2t) with a go-to-goal driver. Every step, each env's scan is rebuilt as robot-frame hit points
(cast_rays: exactly sim2d.lidar's points) and fed to a real robots.spot.local_planner.ObstacleTracker, and its
per-sector features (rl/track_features.sector_velocities) are compared with the env's emulated ones. A fresh tracker
starts with every episode. CPU only.

    PYTHONPATH=. python tools/track_parity.py --preset hard-dense --envs 32 --steps 1500 [--noise 0.0]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from benchmarks.avoidance.rl import env as E
from benchmarks.avoidance.rl.track_features import sector_velocities
from robots.spot.local_planner import ObstacleTracker


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="hard-dense")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.set_num_threads(8)
    env = E.AvoidEnv(a.envs, device="cpu", seed=a.seed,
                     cfg=E.preset(a.preset, obs_version="avoid-v2t", w_space=0.0, track_noise_mps=a.noise))
    obs = env.reset()
    trackers = [ObstacleTracker() for _ in range(a.envs)]
    rel = E.RAY_REL
    emu_all, real_all = [], []

    def real_features(i):
        pp, act = env.people()
        r = E.cast_rays(env.pos[i:i + 1], env.yaw[i:i + 1], pp[i:i + 1], act[i:i + 1], env.segs[i:i + 1],
                        env.sactive[i:i + 1])[0].numpy().astype(np.float64)
        hit = r < E.MAX_RANGE
        scan = [(float(rr * np.cos(q)), float(rr * np.sin(q))) for rr, q in zip(r[hit], rel[hit])]
        pose = (float(env.pos[i, 0]), float(env.pos[i, 1]), float(env.yaw[i]))
        return sector_velocities(scan, trackers[i].update(float(env.t[i]), pose, scan))

    for i in range(a.envs):
        real_features(i)
    for _ in range(a.steps):
        g = obs[:, 192:195]
        bearing = torch.atan2(g[:, 2], g[:, 1])
        act = torch.stack([torch.ones_like(bearing), torch.zeros_like(bearing), bearing.clamp(-1, 1)], -1)
        obs, _, done, _ = env.step(act)
        for i in range(a.envs):
            if done[i]:
                trackers[i] = ObstacleTracker()
            emu_all.append(obs[i, 198:326].numpy().astype(np.float64))
            real_all.append(real_features(i))
    emu = np.stack(emu_all).reshape(-1, 2, 64).transpose(0, 2, 1).reshape(-1, 2) * E.TRACK_VEL_SCALE
    real = np.stack(real_all).reshape(-1, 2, 64).transpose(0, 2, 1).reshape(-1, 2) * E.TRACK_VEL_SCALE
    me, mr = np.abs(emu).sum(-1) > 0, np.abs(real).sum(-1) > 0
    both = me & mr
    err = np.linalg.norm(emu[both] - real[both], axis=-1)
    cos = (emu[both] * real[both]).sum(-1) / (np.linalg.norm(emu[both], axis=-1) * np.linalg.norm(real[both], axis=-1))
    out = {
        "preset": a.preset, "noise_mps": a.noise, "sector_steps": int(len(emu)),
        "moving_share_emulated": round(float(me.mean()), 4), "moving_share_real": round(float(mr.mean()), 4),
        "agree_moving_or_not": round(float((me == mr).mean()), 4),
        "real_only_share_of_real": round(float((mr & ~me).sum() / max(mr.sum(), 1)), 4),
        "emulated_only_share_of_emulated": round(float((me & ~mr).sum() / max(me.sum(), 1)), 4),
        "both": int(both.sum()),
        "err_mps_median": round(float(np.median(err)), 3) if len(err) else None,
        "err_mps_p90": round(float(np.percentile(err, 90)), 3) if len(err) else None,
        "direction_cos_median": round(float(np.median(cos)), 3) if len(cos) else None,
        "real_speed_median": round(float(np.median(np.linalg.norm(real[mr], axis=-1))), 3) if mr.any() else None,
        "emulated_speed_median": round(float(np.median(np.linalg.norm(emu[me], axis=-1))), 3) if me.any() else None,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
