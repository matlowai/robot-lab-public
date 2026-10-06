"""Residual PPO on top of a frozen GR00T N1.7 SO-101 policy -- one training ROUND (one object, one Isaac process).

  cd /mnt/weights/ai/isaac/IsaacLab && CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES uv run --extra teleop python \
      /mnt/work/AI/robot-lab/rl/train.py --object mug --num_envs 32 --waves 20 --round 0 --run_dir <dir> \
      [--ckpt_in <dir>/ckpt/latest.pt] [--port 6190] [--deadline <unix s>]

Isaac Lab 3.0 EA clones ONE object type per process (so101_pick_place.set_object), so scripts/run_rl.sh cycles
rounds over the four training objects, each round resuming <run_dir>/ckpt/latest.pt. Held-out objects are refused.
Every wave appends one JSON line to <run_dir>/train_log.jsonl (strict / task / lifted / released counts, reward
components, PPO stats, timings, RAM / VRAM). Exit codes: 0 done, 3 RAM guard tripped (see gr00t_rl/common.py).
"""

import argparse
import json
import sys
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--object", required=True)
ap.add_argument("--num_envs", type=int, default=32)
ap.add_argument("--waves", type=int, default=20)
ap.add_argument("--round", type=int, default=0)
ap.add_argument("--run_dir", required=True)
ap.add_argument("--ckpt_in", default="")
ap.add_argument("--port", type=int, default=6190)
ap.add_argument("--deadline", type=float, default=0.0, help="unix time; stop starting new waves after it")
ap.add_argument("--units", default="/mnt/weights/ai/robot-lab-data/overnight/full-20260923-2207/lerobot_ds/units.json")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--rms_waves", type=int, default=8, help="update obs normalisers for the first N global waves only")
ap.add_argument("--actor_priv", action="store_true", help="ABLATION: actor also sees privileged sim state")
ap.add_argument("--video_every", type=int, default=0, help="save env-0 video every N global waves (0 = never)")
args = ap.parse_args()
app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app

sys.path.insert(0, "/mnt/work/AI/robot-lab")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # frozen copies import their own gr00t_rl
from multiprocessing.connection import Client  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gr00t_rl import seeds  # noqa: E402
from gr00t_rl.common import atomic_torch_save, gpu_used_mib, mem_available_gb, ram_guard, rss_gb  # noqa: E402
from gr00t_rl.isaac_env import (ACTOR_LOW_DIM, CRITIC_LOW_DIM, TRAIN_NAMES, ChunkEnv, run_wave)  # noqa: E402
from gr00t_rl.ppo import PPOConfig, ResidualActorCritic, gae, ppo_update  # noqa: E402
from gr00t_rl.video import write_mp4  # noqa: E402

if args.object not in TRAIN_NAMES:
    raise SystemExit(f"refusing to train on {args.object!r}: RL trains on {TRAIN_NAMES} only (held-out stay untouched)")
RUN = Path(args.run_dir)
(RUN / "ckpt").mkdir(parents=True, exist_ok=True)
(RUN / "videos").mkdir(parents=True, exist_ok=True)
cfg = PPOConfig(lr=args.lr)
dev = "cuda:0"

ram_guard("startup")
t_boot = time.time()
cenv = ChunkEnv(args.object, args.num_envs, args.units, seed=seeds.train_seed(args.round, 0))
client = Client(("127.0.0.1", args.port), authkey=b"robot-lab-rl")
client.send({"cmd": "info"})
info = client.recv()
ac = ResidualActorCritic(CRITIC_LOW_DIM if args.actor_priv else ACTOR_LOW_DIM, CRITIC_LOW_DIM, info["emb_dim"]).to(dev)
opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr, eps=1e-5)
meta = {"global_wave": 0, "rounds": [], "actor_priv": args.actor_priv, "policy": info["policy"]}
if args.ckpt_in and Path(args.ckpt_in).exists():
    ck = torch.load(args.ckpt_in, map_location=dev, weights_only=False)
    ac.load_state_dict(ck["ac"])
    opt.load_state_dict(ck["opt"])
    meta = ck["meta"]
    assert meta["actor_priv"] == args.actor_priv, "actor_priv mismatch with checkpoint"
print(f"BOOT object={args.object} n={args.num_envs} round={args.round} global_wave={meta['global_wave']} "
      f"policy={info['policy']} boot_s={time.time() - t_boot:.1f} rss={rss_gb():.1f}GB avail={mem_available_gb():.1f}GB",
      flush=True)
log = open(RUN / "train_log.jsonl", "a")

waves_done = 0
for w in range(args.waves):
    if args.deadline and time.time() > args.deadline:
        print(f"DEADLINE reached before wave {w}", flush=True)
        break
    ram_guard(f"round {args.round} wave {w}")
    gw = meta["global_wave"]
    seed = seeds.train_seed(args.round, w)
    vid = (0,) if (args.video_every and gw % args.video_every == 0) else ()
    torch.cuda.reset_peak_memory_stats()
    buf, summ, frames, tm = run_wave(cenv, client, ac, seed, "train", noise_key=seeds.TRAIN_NOISE_KEY,
                                     video_envs=vid, actor_priv=args.actor_priv)
    t0 = time.perf_counter()
    with torch.no_grad():
        adv, ret = gae(buf["rew"], buf["val"], buf["done"], torch.zeros_like(buf["val"][0]), cfg.gamma, cfg.lam)
    m = buf["mask"].bool()
    batch = {k: buf[k][m] for k in ("a_low", "c_low", "u", "rel", "logp")}
    batch["emb"] = buf["emb"][m].float()
    batch["adv"], batch["ret"] = adv[m], ret[m]
    stats = ppo_update(ac, opt, batch, cfg)
    if gw < args.rms_waves:  # after the update, so the update used the normalisation that produced the samples
        ac.actor_enc.update(batch["a_low"], batch["emb"])
        ac.critic_enc.update(batch["c_low"], batch["emb"])
    t_upd = time.perf_counter() - t0
    meta["global_wave"] = gw + 1
    ck = {"ac": ac.state_dict(), "opt": opt.state_dict(), "meta": meta}
    atomic_torch_save(ck, RUN / "ckpt" / "latest.pt")
    v = summ["valid_layout"]
    rec = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "round": args.round, "wave": w, "global_wave": gw,
        "object": args.object, "seed": seed, "n_envs": args.num_envs, "valid_layouts": int(v.sum()),
        **{k: int(summ[k].sum()) for k in ("strict", "task", "lifted", "reached", "over_bowl", "released_over",
                                             "bowl_bad", "blowup", "dropped")},
        "strict_valid": int((summ["strict"] & v).sum()),
        "return_mean": float(summ["return"].mean()),
        "components_mean": {k: round(float(x.mean()), 4) for k, x in summ["components"].items()},
        "transitions": int(m.sum()), "p_release_mean": round(tm["p_release_mean"], 4),
        "p_release_over_bowl": round(tm["p_release_over_bowl"], 4), "n_over_bowl_decisions": tm["n_over_bowl_decisions"],
        "ppo": {k: (round(x, 5) if isinstance(x, float) else x) for k, x in stats.items()},
        "log_std": [round(x, 3) for x in ac.log_std.detach().cpu().tolist()],
        "timing": {"wave_s": round(tm["wave_s"], 1), "plan_s": round(tm["plan"], 1), "sim_s": round(tm["sim"], 1),
                   "actor_s": round(tm["actor"], 2), "update_s": round(t_upd, 1), "plans": tm["plans"],
                   "plan_envs": tm["plan_envs"],
                   "env_ticks_per_s": round(args.num_envs * tm["ticks"] / tm["wave_s"], 1),
                   "rl_steps_per_s": round(int(m.sum()) / tm["wave_s"], 2),
                   "plan_ms_per_env": round(1e3 * tm["plan"] / max(1, tm["plan_envs"]), 2)},
        "mem": {"rss_gb": round(rss_gb(), 2), "avail_gb": round(mem_available_gb(), 1),
                "torch_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "nvidia_smi": gpu_used_mib()},
    }
    log.write(json.dumps(rec) + "\n")
    log.flush()
    print("WAVE", json.dumps(rec), flush=True)
    waves_done += 1
    for i, fr in frames.items():
        if fr:
            tag = "strict" if bool(summ["strict"][i]) else ("task" if bool(summ["task"][i]) else "fail")
            write_mp4(RUN / "videos" / f"train_gw{gw:05d}_{args.object.replace(' ', '_')}_env{i}_{tag}.mp4", fr)

meta["rounds"].append({"round": args.round, "object": args.object, "waves_done": waves_done,
                       "end": time.strftime("%Y-%m-%dT%H:%M:%S")})
atomic_torch_save({"ac": ac.state_dict(), "opt": opt.state_dict(), "meta": meta}, RUN / "ckpt" / "latest.pt")
atomic_torch_save({"ac": ac.state_dict(), "opt": opt.state_dict(), "meta": meta},
                  RUN / "ckpt" / f"round{args.round:03d}.pt")
client.send({"cmd": "close"})
log.close()
cenv.close()
app.close()
