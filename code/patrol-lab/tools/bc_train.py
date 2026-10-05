"""Behaviour cloning of the heuristic planner into the PPO policy's actor (D46, RL v4 arm A3 warm start).

Fits ActorCritic.actor's mean to the teacher's actions (tools/collect_teacher.py) by MSE, with the observation
normaliser set from the teacher data, and saves a checkpoint in train.py's format (squash="clip"; critic untrained),
ready for `train.py --init-ckpt`. 10 % of the episodes (by seed) are held out for the validation loss.

    PYTHONPATH=. python tools/bc_train.py --data <teacher dir> --out <dir>/bc.pt --epochs 12
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.avoidance.rl.env import OBS_SPECS, preset
from benchmarks.avoidance.rl.ppo import ActorCritic, PPOConfig
from benchmarks.avoidance.rl.train import save_ckpt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init-log-std", type=float, default=-1.0, help="exploration std for the PPO that follows")
    ap.add_argument("--log-std-max", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    files = sorted(f for d in a.data.split(",") for f in glob.glob(str(Path(d) / "ep_*.npz")))  # DAgger: dirs joined
    if not files:
        raise SystemExit(f"no ep_*.npz in {a.data}")
    rng = np.random.default_rng(a.seed)
    val_files = set(rng.choice(files, size=max(1, len(files) // 10), replace=False).tolist())
    def load(fs):  # preallocated: DAgger round 3 is ~9 M steps, and a list + concatenate would hold it twice
        n, width = 0, None
        for f in fs:
            with np.load(f) as d:
                n, width = n + len(d["act"]), d["obs"].shape[1]
        o, ac, i = np.empty((n, width), np.float32), np.empty((n, 3), np.float32), 0
        for f in fs:
            with np.load(f) as d:
                k = len(d["act"])
                o[i:i + k], ac[i:i + k] = d["obs"], d["act"]
                i += k
        return torch.from_numpy(o), torch.from_numpy(ac)
    tr_o, tr_a = load([f for f in files if f not in val_files])
    va_o, va_a = load(sorted(val_files))
    versions = [v for v, sp in OBS_SPECS.items() if sp["dim"] == tr_o.shape[1]]
    if not versions:
        raise SystemExit(f"obs width {tr_o.shape[1]} matches no obs version")
    version, dim = versions[0], tr_o.shape[1]
    dev = torch.device(a.device)
    model = ActorCritic(dim, 3, (256, 256), init_log_std=a.init_log_std, squash="clip",
                        log_std_max=a.log_std_max).to(dev)
    for i in range(0, len(tr_o), 1 << 20):  # exact moments, merged chunk by chunk (no float64 copy of all of it)
        model.obs_rms.update(tr_o[i:i + (1 << 20)].to(dev))
    tr_o, tr_a, va_o, va_a = tr_o.to(dev), tr_a.to(dev), va_o.to(dev), va_a.to(dev)
    opt = torch.optim.Adam(model.actor.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    hist = []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr_o), device=dev)
        tot, n = 0.0, 0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            mu = model.actor(model.obs_rms(tr_o[idx]))
            loss = ((mu - tr_a[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            n += len(idx)
        sched.step()
        with torch.no_grad():
            vmu = torch.cat([model.actor(model.obs_rms(va_o[i:i + 65536])) for i in range(0, len(va_o), 65536)])
            vloss = float(((vmu - va_a) ** 2).mean())
            per_dim = ((vmu - va_a) ** 2).mean(0).tolist()
            vmid = float(((vmu[:, 0].clamp(-1, 1) > -0.8) & (vmu[:, 0].clamp(-1, 1) < 0.8)).float().mean())
        hist.append({"epoch": ep + 1, "train_mse": round(tot / n, 5), "val_mse": round(vloss, 5),
                     "val_mse_per_dim": [round(x, 5) for x in per_dim], "val_vx_mid_frac": round(vmid, 4)})
        print(json.dumps(hist[-1]), flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env_cfg = preset("hard-dense", obs_version=version, w_space=0.0)  # metadata only: the A3 training env
    save_ckpt(out, model.cpu(), torch.optim.Adam(model.parameters()), 0, 0, env_cfg, PPOConfig(), (256, 256))
    (out.with_suffix(".json")).write_text(json.dumps({
        "data": a.data, "train_samples": len(tr_o), "val_samples": len(va_o), "val_episodes": len(val_files),
        "epochs": a.epochs, "history": hist, "wall_s": round(time.time() - t0, 1), "init_log_std": a.init_log_std,
    }, indent=2))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
