"""Raw demo npz (tools/record_demos.py) -> Isaac-GR00T LeRobot v2.1 dataset, directly (GR00T night 2).

Night 1 went raw -> LeRobot v3 (tools/to_lerobot.py, 2.6 s/episode single-threaded) -> v2.1 (tools/to_gr00t.py).
This writes v2.1 in one parallel pass with the SAME unit mapping, video settings and checks:
  * units: the night-1 units.json (arm deg + fixed per-joint offsets, gripper USD deg [-10, 100] -> [0, 100]), so
    night-2 states/actions are in exactly night-1's units and the eval harness converts identically
  * video: H.264 yuv420p, crf 23, preset veryfast, GOP 2 (the night-1 dataset's settings), via FFmpeg 7.1
  * checks: packet count == episode length for EVERY mp4; for --check episodes (first and last of each object) the
    decoded video is best aligned with the recorded RGB at shift 0 (whole-episode mean PSNR over shifts -2..2) and
    the parquet state/action equal the unit-converted npz; EVERY episode's last action step must not jump
    (the night-1 post-reset label bug: >= 33 units); release-frame statistics are written to release_stats.json.
Keeps only episodes whose npz has success=True (strict, judged by the recorder).

  /mnt/weights/ai/nvidia-action/Isaac-GR00T/.venv/bin/python tools/gr00t_raw_to_ds.py \
      --raw <run>/raw/mug <run>/raw/blue_block ... --out <run>/gr00t_ds [--units <night-1 units.json>] [--check 8]
"""

import argparse
import concurrent.futures as cf
import glob
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

FF = "/mnt/weights/ai/nvidia-action/ffmpeg7/ffmpeg-n7.1.5-12-g1fdbca85aa-linux64-gpl-shared-7.1"
FFENV = {**os.environ, "LD_LIBRARY_PATH": f"{FF}/lib"}
NIGHT1_DS = "/mnt/weights/ai/robot-lab-data/overnight-gr00t/full-20261004-2334/gr00t_ds"
CAMS = {"observation.images.scene": "scene", "observation.images.wrist": "wrist"}
V21_DATA = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
OPEN_UNITS = 30.0  # gripper command above this (dataset units) = open; closed/holding ~9, pre-shape/release ~50

p = argparse.ArgumentParser()
p.add_argument("--raw", nargs="+", required=True)
p.add_argument("--out", required=True)
p.add_argument("--units", default=f"{NIGHT1_DS}/units.json")
p.add_argument("--info_template", default=f"{NIGHT1_DS}/meta/info.json")
p.add_argument("--max_per_dir", type=int, default=0)
p.add_argument("--check", type=int, default=8)
p.add_argument("--workers", type=int, default=24)
p.add_argument("--max_last_jump", type=float, default=5.0, help="units; the night-1 bogus label jumped >= 33")
args = p.parse_args()
T0 = time.time()
units = json.loads(Path(args.units).read_text())
OFF = np.array(units["arm_offsets_deg"], np.float32)
GMIN, GMAX = units["gripper"]["sim_deg_range"]


def to_units(x):
    out = np.empty_like(x, dtype=np.float32)
    out[:, :5] = np.degrees(x[:, :5]) + OFF
    out[:, 5] = (np.degrees(x[:, 5]) - GMIN) / (GMAX - GMIN) * 100.0
    return out


files = []
for d in args.raw:
    fs = [f for f in sorted(glob.glob(f"{d}/ep_*.npz")) if bool(np.load(f)["success"])]
    files += fs[: args.max_per_dir] if args.max_per_dir else fs
if not files:
    raise SystemExit("no successful episodes")
out = Path(args.out)
if out.exists():
    shutil.rmtree(out)
(out / "meta").mkdir(parents=True)
tasks = list(dict.fromkeys(str(np.load(f)["task"]) for f in files))
task_idx = {t: i for i, t in enumerate(tasks)}
print(f"CONVERT {len(files)} strict-successful episodes from {len(args.raw)} dirs, {len(tasks)} tasks", flush=True)


def encode(frames, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    cmd = [f"{FF}/bin/ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-g", "2",
           "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True, env=FFENV)
    got = subprocess.run([f"{FF}/bin/ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(dst)],
                         check=True, env=FFENV, capture_output=True, text=True).stdout.strip()
    if int(got) != len(frames):
        raise RuntimeError(f"{dst}: {got} packets, expected {len(frames)}")


def release_frames(act_u):
    g = act_u[:, 5]
    closed = np.where(g < OPEN_UNITS)[0]
    return int((g[closed.max() + 1:] > OPEN_UNITS).sum()) if len(closed) else 0


def work(job):
    i, f, start = job
    d = np.load(f)
    st, ac = to_units(d["state"]), to_units(d["action"])
    n = len(st)
    last_jump = float(np.abs(ac[-1] - ac[-2]).max())
    df = pd.DataFrame({"observation.state": list(st), "action": list(ac),
                       "timestamp": (np.arange(n) / 30.0).astype(np.float32), "frame_index": np.arange(n),
                       "episode_index": np.full(n, i), "index": np.arange(start, start + n),
                       "task_index": np.full(n, task_idx[str(d["task"])])})
    dst = out / V21_DATA.format(episode_chunk=i // 1000, episode_index=i)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst)
    for key, raw_key in CAMS.items():
        encode(d[raw_key], out / V21_VIDEO.format(episode_chunk=i // 1000, video_key=key, episode_index=i))
    return {"episode_index": i, "tasks": [str(d["task"])], "length": n, "source": f, "object": str(d["object"]),
            "release_frames": release_frames(ac), "last_jump_units": round(last_jump, 3),
            "strict": json.loads(str(d["strict"])) if "strict" in d.files else None}


lengths = [len(np.load(f)["state"]) for f in files]
starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
with cf.ProcessPoolExecutor(args.workers) as ex:
    eps = list(ex.map(work, [(i, f, int(s)) for i, (f, s) in enumerate(zip(files, starts))]))
frames = int(sum(lengths))
print(f"VIDEOS {2 * len(eps)} files encoded and packet-count verified", flush=True)

bad = [e for e in eps if e["last_jump_units"] > args.max_last_jump]
if bad:
    raise SystemExit(f"LABELCHECK FAILED: {len(bad)} episodes end with an action jump > {args.max_last_jump} units, "
                     f"e.g. {bad[:3]}")
print(f"LABELCHECK OK: max final-step action jump {max(e['last_jump_units'] for e in eps):.3f} units over "
      f"{len(eps)} episodes (night-1 bogus label: >= 33)", flush=True)

info = json.loads(Path(args.info_template).read_text())
info.update({"total_episodes": len(eps), "total_frames": frames, "total_tasks": len(tasks),
             "total_videos": 2 * len(eps), "total_chunks": (len(eps) - 1) // 1000 + 1,
             "splits": {"train": f"0:{len(eps)}"}, "data_path": V21_DATA, "video_path": V21_VIDEO})
(out / "meta/info.json").write_text(json.dumps(info, indent=2))
(out / "meta/episodes.jsonl").write_text("".join(
    json.dumps({"episode_index": e["episode_index"], "tasks": e["tasks"], "length": e["length"]}) + "\n" for e in eps))
(out / "meta/tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(tasks)))
shutil.copy(Path(args.info_template).parent / "modality.json", out / "meta/modality.json")
shutil.copy(args.units, out / "units.json")

# ---- release statistics ----
rel = {}
for e in eps:
    rel.setdefault(e["object"], []).append(e["release_frames"])
allr = [e["release_frames"] for e in eps]
stats = {"open_units_threshold": OPEN_UNITS, "definition": "frames after the last closed command whose gripper "
         "command is open (> threshold)", "all": {"median": float(np.median(allr)), "min": int(min(allr)),
         "max": int(max(allr)), "episodes": len(allr)},
         "per_object": {k: {"median": float(np.median(v)), "min": int(min(v)), "max": int(max(v)), "episodes": len(v)}
                        for k, v in rel.items()},
         "episode_length_median": float(np.median(lengths)), "frames": frames}
(out / "release_stats.json").write_text(json.dumps(stats, indent=1))
(out / "episodes_detail.json").write_text(json.dumps(eps, indent=0))
print("RELEASE", json.dumps(stats["all"]), "per object", json.dumps({k: v["median"] for k, v in stats["per_object"].items()}),
      flush=True)


# ---- alignment + value check against the recorded npz ----
def decode(path):
    raw = subprocess.run([f"{FF}/bin/ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, env=FFENV, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 256, 256, 3)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


by_obj = {}
for e in eps:
    by_obj.setdefault(e["object"], []).append(e)
per = max(1, args.check // max(1, len(by_obj)))
pick = []
for v in by_obj.values():
    pick += v[: max(1, per - 1)] + [v[-1]]
worst = 99.0
for e in pick:
    i = e["episode_index"]
    d = np.load(e["source"])
    dfp = pd.read_parquet(out / V21_DATA.format(episode_chunk=i // 1000, episode_index=i))
    ds_s, ds_a = np.stack(dfp["observation.state"].to_numpy()), np.stack(dfp["action"].to_numpy())
    assert np.abs(ds_s - to_units(d["state"])).max() < 1e-4 and np.abs(ds_a - to_units(d["action"])).max() < 1e-4, i
    for key, raw_key in CAMS.items():
        vid, ref = decode(out / V21_VIDEO.format(episode_chunk=i // 1000, video_key=key, episode_index=i)), d[raw_key]
        assert len(vid) == len(ref), (i, key, len(vid), len(ref))
        score = {s: float(np.mean([psnr(vid[t], ref[t + s]) for t in range(2, len(ref) - 2)])) for s in range(-2, 3)}
        if max(score, key=score.get) != 0:
            raise SystemExit(f"frame misalignment ep {i} {key}: {score}")
        worst = min(worst, score[0])
print(f"RAWCHECK {len(pick)} episodes OK, worst frame PSNR vs recorded RGB {worst:.1f} dB", flush=True)
print(f"TIMING {len(eps)} episodes, {frames} frames in {time.time() - T0:.1f} s", flush=True)
print("DONE", out, flush=True)
