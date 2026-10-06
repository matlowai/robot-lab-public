"""Paired STRICT evaluation: base GR00T vs GR00T + residual RL, on identical layouts and identical GR00T noise.

  cd /mnt/weights/ai/isaac/IsaacLab && CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES uv run --extra teleop python \
      /mnt/work/AI/robot-lab/rl/evaluate.py --object mug --num_envs 32 --waves 2 --ckpt <run>/ckpt/latest.pt \
      --out <run>/eval [--seeds sealed|dev] [--arms base,rl]

Pre-registered protocol (rl/DESIGN.md section 5): seeds = SEALED range (gr00t_rl/seeds.py, never used in training);
per wave both arms run on the same wave seed (same N layouts) with the same GR00T noise seeds per (env, chunk);
the RL arm is DETERMINISTIC (arm = 4 deg * tanh(mean), release = p > 0.5); the checkpoint is the run's FINAL one (no
selection). Headline = STRICT successes over valid layouts (object >= 10 cm from the bowl, bowl upright at start).
Writes results_<object>.json (per-episode records for every arm + paired counts) and videos of the first wave's
envs --video_envs for every arm (scene | wrist), plus 8-frame strips.
"""

import argparse
import json
import sys
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--object", required=True)
ap.add_argument("--num_envs", type=int, default=32)
ap.add_argument("--waves", type=int, default=2)
ap.add_argument("--ckpt", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--arms", default="base,rl")
ap.add_argument("--seeds", default="sealed", choices=["sealed", "dev"])
ap.add_argument("--port", type=int, default=6190)
ap.add_argument("--video_envs", default="0,1,2,3")
ap.add_argument("--rest_ticks", type=int, default=30, help="post-success rest check (0 = off: env auto-resets on success)")
ap.add_argument("--units", default="/mnt/weights/ai/robot-lab-data/overnight/full-20260923-2207/lerobot_ds/units.json")
args = ap.parse_args()
app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app

sys.path.insert(0, "/mnt/work/AI/robot-lab")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # frozen copies import their own gr00t_rl
from multiprocessing.connection import Client  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from gr00t_rl import seeds  # noqa: E402
from gr00t_rl.common import mem_available_gb, ram_guard, rss_gb, wilson  # noqa: E402
from gr00t_rl.isaac_env import ACTOR_LOW_DIM, CRITIC_LOW_DIM, ChunkEnv, run_wave  # noqa: E402
from gr00t_rl.ppo import ResidualActorCritic  # noqa: E402
from gr00t_rl.video import write_mp4, write_strip  # noqa: E402

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
TAG = args.object.replace(" ", "_")
arms = args.arms.split(",")
vids = tuple(int(x) for x in args.video_envs.split(",") if x != "")
dev = "cuda:0"
ram_guard("eval startup")
cenv = ChunkEnv(args.object, args.num_envs, args.units, seed=0, rest_check=args.rest_ticks > 0)
client = Client(("127.0.0.1", args.port), authkey=b"robot-lab-rl")
client.send({"cmd": "info"})
info = client.recv()
ac, meta = None, {}
if "rl" in arms:
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    meta = ck["meta"]
    ac = ResidualActorCritic(CRITIC_LOW_DIM if meta.get("actor_priv") else ACTOR_LOW_DIM, CRITIC_LOW_DIM,
                             info["emb_dim"]).to(dev)
    ac.load_state_dict(ck["ac"])
    ac.eval()

episodes = {a: [] for a in arms}
timing = {a: [] for a in arms}
for w in range(args.waves):
    seed = seeds.eval_seed(args.object, w) if args.seeds == "sealed" else seeds.dev_seed(500_000 + w)
    for arm in arms:
        ram_guard(f"eval {args.object} wave {w} {arm}")
        _, summ, frames, tm = run_wave(cenv, client, ac, seed, arm, noise_key=seeds.EVAL_NOISE_KEY,
                                       video_envs=vids if w == 0 else (), actor_priv=bool(meta.get("actor_priv")),
                                       rest_ticks=args.rest_ticks)
        timing[arm].append({k: (round(v, 2) if isinstance(v, float) else v) for k, v in tm.items()})
        for i in range(args.num_envs):
            rec = {"wave": w, "seed": seed, "env": i, "valid_layout": bool(summ["valid_layout"][i])}
            for k in ("strict", "task", "lifted", "reached", "over_bowl", "released_over", "bowl_bad", "blowup",
                      "dropped"):
                rec[k] = bool(summ[k][i])
            rec["strict_rested"] = rec["strict"] and bool(summ["rested"][i])
            rec["max_rise_m"] = round(float(summ["max_rise"][i]), 4)
            rec["end_tick"] = int(summ["end_step"][i])
            rec["return"] = round(float(summ["return"][i]), 3)
            rec["n_release_chunks"] = int(summ["n_release_chunks"][i])
            rec["first_release_chunk"] = int(summ["first_release_chunk"][i])
            episodes[arm].append(rec)
        for i, fr in frames.items():
            if fr:
                r = episodes[arm][-args.num_envs + i]
                st = "strict" if r["strict"] else ("task" if r["task"] else "fail")
                write_mp4(OUT / f"{TAG}_{arm}_w{w}_env{i}_{st}.mp4", fr)
                write_strip(OUT / f"{TAG}_{arm}_w{w}_env{i}_{st}_strip.png", fr)
        print(f"EVAL object={args.object} wave={w} arm={arm} strict={int(summ['strict'][summ['valid_layout']].sum())}/{int(summ['valid_layout'].sum())}"
              f" strict_rested={int((summ['strict'] & summ['rested'])[summ['valid_layout']].sum())}"
              f"/{int(summ['valid_layout'].sum())} wave_s={tm['wave_s']:.0f}", flush=True)


if "base" in arms and "rl" in arms:  # a layout counts only if it is valid in BOTH arms (GPU physics is not bitwise
    for b, r in zip(episodes["base"], episodes["rl"]):  # reproducible, so the post-settle check can differ by an env)
        assert (b["wave"], b["env"], b["seed"]) == (r["wave"], r["env"], r["seed"])
        joint = b["valid_layout"] and r["valid_layout"]
        for x in (b, r):
            x["valid_layout_raw"], x["valid_layout"] = x["valid_layout"], joint


def tally(eps):
    v = [e for e in eps if e["valid_layout"]]
    k = sum(e["strict"] for e in v)
    lo, hi = wilson(k, len(v))
    return {"valid": len(v), "strict": k, "strict_rate": k / max(1, len(v)), "strict_ci95": [lo, hi],
            "strict_rested": sum(e["strict_rested"] for e in v),
            "task": sum(e["task"] for e in v), "lifted": sum(e["lifted"] for e in v),
            "over_bowl": sum(e["over_bowl"] for e in v), "released_over": sum(e["released_over"] for e in v),
            "bowl_bad": sum(e["bowl_bad"] for e in v), "blowup": sum(e["blowup"] for e in v),
            "invalid_layouts": len(eps) - len(v)}


summary = {"object": args.object, "seeds": args.seeds, "rest_ticks": args.rest_ticks, "waves": args.waves, "num_envs": args.num_envs,
           "ckpt": args.ckpt, "ckpt_meta": {k: meta.get(k) for k in ("global_wave", "actor_priv", "policy")},
           "policy": info["policy"], "arms": {a: tally(episodes[a]) for a in arms}}
if "base" in arms and "rl" in arms:
    pairs = [(b, r) for b, r in zip(episodes["base"], episodes["rl"]) if b["valid_layout"]]
    summary["paired"] = {"both": sum(b["strict"] and r["strict"] for b, r in pairs),
                         "base_only": sum(b["strict"] and not r["strict"] for b, r in pairs),
                         "rl_only": sum(r["strict"] and not b["strict"] for b, r in pairs),
                         "neither": sum(not b["strict"] and not r["strict"] for b, r in pairs)}
summary["mem"] = {"rss_gb": round(rss_gb(), 2), "avail_gb": round(mem_available_gb(), 1)}
(OUT / f"results_{TAG}.json").write_text(json.dumps({"summary": summary, "episodes": episodes, "timing": timing},
                                                    indent=1))
print("SUMMARY", json.dumps(summary), flush=True)
client.send({"cmd": "close"})
cenv.close()
app.close()
