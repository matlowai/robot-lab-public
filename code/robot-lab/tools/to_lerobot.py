"""Raw sim episodes (npz from record_demos.py) -> LeRobot v3 dataset for FLUX 3 Action SO-101 LoRA.

Run with the LeRobot venv:
  /mnt/weights/ai/lerobot/lerobot/.venv/bin/python tools/to_lerobot.py \
      --raw /mnt/weights/ai/robot-lab-data/raw/mug ... --out /mnt/weights/ai/robot-lab-data/lerobot/so101_pickplace

Units (the LeRobot FLUX 3 docs: "The base's saved processors own normalization statistics. Feed raw states/actions";
the SO-101 package's quantiles assume its own calibration):
  * arm joints: sim radians -> degrees (true scale, so per-step deltas match the package's delta statistics),
    USD sign convention (validated sim-to-real by NVIDIA's Sim-to-Real SO-101 workshop), plus a per-joint offset that
    centers our motion range inside the package's state q01..q99 band
  * gripper: USD degrees [-10, 100] -> [0, 100] (the workshop's LeRobot gripper mapping; package stats 0.5..54)
The mapping is written to units.json; the eval harness applies its exact inverse to commanded actions.
"""

import argparse
import time
import glob
import json
import shutil
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PKG = Path("/mnt/weights/ai/flux3-action/flux-3-action-so101/dataset_statistics.json")
GRIP_MIN_DEG, GRIP_MAX_DEG = -10.0, 100.0

p = argparse.ArgumentParser()
p.add_argument("--raw", nargs="+", required=True, help="raw episode dirs (ep_*.npz)")
p.add_argument("--out", required=True)
p.add_argument("--repo_id", default="local/so101_pickplace")
p.add_argument("--max_per_dir", type=int, default=0, help="cap successful episodes per dir (0 = all)")
args = p.parse_args()

files = []
for d in args.raw:
    fs = [f for f in sorted(glob.glob(f"{d}/ep_*.npz")) if bool(np.load(f)["success"])]
    files += fs[: args.max_per_dir] if args.max_per_dir else fs
print(f"CONVERT {len(files)} successful episodes from {len(args.raw)} dirs", flush=True)
T0 = time.time()
if not files:
    raise SystemExit("no successful episodes")

pkg = json.loads(PKG.read_text())
q01, q99 = np.array(pkg["state"]["q01"]), np.array(pkg["state"]["q99"])


def arm_deg(x):
    return np.degrees(x[..., :5])


# per-joint offset: center our median arm pose inside the package's state band
allstate = np.concatenate([np.load(f)["state"] for f in files])
offsets = (q01[:5] + q99[:5]) / 2 - np.median(arm_deg(allstate), axis=0)


def to_units(x):
    """(T, 6) sim radians -> dataset units."""
    out = np.empty_like(x, dtype=np.float32)
    out[:, :5] = arm_deg(x) + offsets
    g = np.degrees(x[:, 5])
    out[:, 5] = (g - GRIP_MIN_DEG) / (GRIP_MAX_DEG - GRIP_MIN_DEG) * 100.0
    return out


from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

out = Path(args.out)
if out.exists():
    shutil.rmtree(out)
features = {
    "observation.images.scene": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
    "observation.images.wrist": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"{j}.pos" for j in JOINTS]},
    "action": {"dtype": "float32", "shape": (6,), "names": [f"{j}.pos" for j in JOINTS]},
}
from lerobot.configs.video import RGBEncoderConfig  # noqa: E402

# LeRobot's default is AV1 (libsvtav1): ~35 s/episode here, too slow for an overnight dataset. H.264 veryfast with the
# same 2-frame GOP keeps random-access decoding cheap for training; streaming encodes during add_frame (no PNGs).
ds = LeRobotDataset.create(repo_id=args.repo_id, fps=30, features=features, root=out, robot_type="so101_follower",
                           use_videos=True, streaming_encoding=True,
                           rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="veryfast", crf=23, g=2))
states_u, actions_u, tasks = [], [], {}
for f in files:
    d = np.load(f)
    st, ac = to_units(d["state"]), to_units(d["action"])
    task = str(d["task"]) if "task" in d.files else "put the mug in the yellow bowl"
    tasks[task] = tasks.get(task, 0) + 1
    # Load each array ONCE: indexing an NpzFile key (d["scene"][t]) re-decompresses the whole array per access.
    scene, wrist = d["scene"], d["wrist"]
    for t in range(len(st)):
        ds.add_frame({"observation.images.scene": scene[t], "observation.images.wrist": wrist[t],
                      "observation.state": st[t], "action": ac[t], "task": task})
    ds.save_episode()
    states_u.append(st)
    actions_u.append(ac)
ds.finalize()
print(f"TIMING {len(files)} episodes in {time.time() - T0:.1f} s ({(time.time() - T0) / len(files):.2f} s/episode)", flush=True)

S, A = np.concatenate(states_u), np.concatenate(actions_u)
dA = np.concatenate([np.diff(a, axis=0) for a in actions_u])
units = {"joints": JOINTS, "arm": "degrees(sim rad) + offset", "arm_offsets_deg": offsets.round(4).tolist(),
         "gripper": {"sim_deg_range": [GRIP_MIN_DEG, GRIP_MAX_DEG], "dataset_range": [0, 100]},
         "episodes": len(files), "frames": int(len(S)), "tasks": tasks}
(out / "units.json").write_text(json.dumps(units, indent=1))
print("UNITS", json.dumps(units))
print("STATE ours q01", np.percentile(S, 1, 0).round(1).tolist(), " pkg", q01.round(1).tolist())
print("STATE ours q99", np.percentile(S, 99, 0).round(1).tolist(), " pkg", q99.round(1).tolist())
print("DELTA ours q01", np.percentile(dA[:, :5], 1, 0).round(2).tolist(), " pkg", np.array(pkg["action"]["q01"][:5]).round(2).tolist())
print("DELTA ours q99", np.percentile(dA[:, :5], 99, 0).round(2).tolist(), " pkg", np.array(pkg["action"]["q99"][:5]).round(2).tolist())
print("GRIP  ours min/max", A[:, 5].min().round(1), A[:, 5].max().round(1), " pkg action q01/q99",
      round(pkg["action"]["q01"][5], 1), round(pkg["action"]["q99"][5], 1))
# read-back check
rd = LeRobotDataset(args.repo_id, root=out)
s = rd[0]
print("READBACK", len(rd), "frames;", {k: tuple(v.shape) for k, v in s.items() if hasattr(v, "shape")}, "task:", s.get("task"))
