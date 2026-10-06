"""LeRobot v3 dataset (tools/to_lerobot.py output) -> Isaac-GR00T LeRobot v2.1 layout for N1.7 post-training.

Isaac-GR00T's loader (gr00t/data/dataset/lerobot_episode_loader.py) reads the v2.1 layout: one parquet and one mp4
per episode per camera, meta/episodes.jsonl, meta/tasks.jsonl, meta/modality.json and a GR00T meta/stats.json
(generated afterwards by `gr00t/data/stats.py`). Our v3 dataset concatenates episodes into shared files, so this
script splits them back out. Nothing is resampled or re-labelled: same frames, same state/action values in the same
dataset units (units.json is copied so the eval harness converts exactly as for FLUX), minus each episode's final
frame (see "Label fix" below).

  <gr00t venv>/bin/python tools/to_gr00t.py --src <run>/lerobot_ds --out <dir> \
      [--per_task N] [--raw_root <run>/raw] [--check 12] [--workers 16]

Label fix (default on): every source episode's FINAL action label is bogus. record_demos.py reads
robot.data.joint_pos_target AFTER env.step, and on the success step the env has already auto-reset, so the last label
is the reset pose (jaw 9 = closed, arm jumps 33-100 units) instead of the expert's RELEASE command (jaw 50). Measured
on all 909 episodes of full-20260923-2207 (min jump 33 units, median 55). The final frame is dropped (909 frames of
130,980); --keep_last_frame reproduces the FLUX training data exactly.

Video: stream copy by default (lossless). Every source episode starts on a keyframe (per-episode encoding, GOP 2);
the cut is verified by packet count for EVERY episode, and with --raw_root against the original recorded RGB
(PSNR at the true frame index must beat the neighbours, i.e. no off-by-one). --reencode switches to libx264 crf 18.
"""

import argparse
import concurrent.futures as cf
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
CAMS = ["observation.images.scene", "observation.images.wrist"]
V21_DATA = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
MODALITY = {
    "state": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
    "action": {"single_arm": {"start": 0, "end": 5}, "gripper": {"start": 5, "end": 6}},
    "video": {"scene": {"original_key": "observation.images.scene"},
              "wrist": {"original_key": "observation.images.wrist"}},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}

p = argparse.ArgumentParser()
p.add_argument("--src", required=True, help="LeRobot v3 dataset root (has units.json)")
p.add_argument("--out", required=True)
p.add_argument("--per_task", type=int, default=0, help="keep the first N episodes of each task (0 = all)")
p.add_argument("--raw_root", default="", help="raw npz root (<run>/raw) for the pixel/state alignment check")
p.add_argument("--check", type=int, default=12, help="episodes to pixel-check against raw (spread over tasks)")
p.add_argument("--workers", type=int, default=16)
p.add_argument("--reencode", action="store_true")
p.add_argument("--keep_last_frame", action="store_true",
               help="keep each episode's final frame (default: drop it, its action label is the post-reset target)")
args = p.parse_args()
T0 = time.time()
src, out = Path(args.src), Path(args.out)
info = json.loads((src / "meta/info.json").read_text())
assert info["codebase_version"] == "v3.0", info["codebase_version"]
fps = info["fps"]
eps = pd.concat([pd.read_parquet(f) for f in sorted((src / "meta/episodes").glob("chunk-*/file-*.parquet"))])
eps = eps.sort_values("episode_index").reset_index(drop=True)
tasks_df = pd.read_parquet(src / "meta/tasks.parquet")
task_of = {int(i): t for t, i in zip(tasks_df.index, tasks_df["task_index"])}
data = pd.concat([pd.read_parquet(f) for f in sorted((src / "data").glob("chunk-*/file-*.parquet"))]).set_index("index", drop=False)

eps["task"] = [t[0] for t in eps["tasks"]]
DROP = 0 if args.keep_last_frame else 1
eps["src_length"] = eps["length"]
eps["length"] = eps["length"] - DROP
if args.per_task:
    eps = eps.groupby("task", sort=False).head(args.per_task).reset_index(drop=True)
print(f"CONVERT {len(eps)} episodes ({eps['length'].sum()} frames; final frame dropped: {bool(DROP)}) from {src}",
      flush=True)

if out.exists():
    shutil.rmtree(out)
(out / "meta").mkdir(parents=True)


def cut(job):
    """One episode x camera: stream-copy (or re-encode) [from_ts, from_ts + length/fps) into its own mp4."""
    new_i, row, cam = job
    srcf = src / info["video_path"].format(video_key=cam, chunk_index=int(row[f"videos/{cam}/chunk_index"]),
                                           file_index=int(row[f"videos/{cam}/file_index"]))
    dst = out / V21_VIDEO.format(episode_chunk=new_i // 1000, video_key=cam, episode_index=new_i)
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0, n = float(row[f"videos/{cam}/from_timestamp"]), int(row["length"])
    codec = ["-c:v", "libx264", "-crf", "18", "-g", "2", "-pix_fmt", "yuv420p", "-preset", "veryfast"] if args.reencode \
        else ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    cmd = [f"{FF}/bin/ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t0:.6f}", "-i", str(srcf),
           "-frames:v", str(n), "-an", *codec, str(dst)]
    subprocess.run(cmd, check=True, env=FFENV)
    got = subprocess.run([f"{FF}/bin/ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(dst)], check=True, env=FFENV,
                         capture_output=True, text=True).stdout.strip()
    if int(got) != n:
        raise RuntimeError(f"episode {new_i} {cam}: {got} packets, expected {n} ({dst})")
    return new_i


# ---- per-episode parquet + meta ----
task_names = list(dict.fromkeys(eps["task"]))  # contiguous task indices in first-seen order
task_idx = {t: i for i, t in enumerate(task_names)}
episodes_jsonl, frames, jobs = [], 0, []
for new_i, row in eps.iterrows():
    df = data.loc[int(row["dataset_from_index"]): int(row["dataset_to_index"]) - 1].copy()
    assert len(df) == int(row["src_length"]), (new_i, len(df), row["src_length"])
    df = df.iloc[: int(row["length"])]
    assert (df["episode_index"] == int(row["episode_index"])).all()
    assert task_of[int(df["task_index"].iloc[0])] == row["task"]
    df["episode_index"] = new_i
    df["index"] = np.arange(frames, frames + len(df))
    df["frame_index"] = np.arange(len(df))
    df["timestamp"] = (np.arange(len(df)) / fps).astype(np.float32)
    df["task_index"] = task_idx[row["task"]]
    dst = out / V21_DATA.format(episode_chunk=new_i // 1000, episode_index=new_i)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(dst)
    episodes_jsonl.append({"episode_index": new_i, "tasks": [row["task"]], "length": int(row["length"]),
                           "source_episode_index": int(row["episode_index"])})
    frames += len(df)
    jobs += [(new_i, row, cam) for cam in CAMS]

with cf.ThreadPoolExecutor(args.workers) as ex:
    list(ex.map(cut, jobs))
print(f"VIDEOS {len(jobs)} files cut and packet-count verified ({'re-encoded' if args.reencode else 'stream copy'})",
      flush=True)

feats = {k: v for k, v in info["features"].items()}
for cam in CAMS:
    feats[cam] = {**feats[cam], "info": {**feats[cam]["info"], "video.is_depth_map": False}}
n_ep = len(eps)
v21 = {"codebase_version": "v2.1", "robot_type": info["robot_type"], "total_episodes": n_ep, "total_frames": frames,
       "total_tasks": len(task_names), "total_videos": n_ep * len(CAMS), "total_chunks": (n_ep - 1) // 1000 + 1,
       "chunks_size": 1000, "fps": fps, "splits": {"train": f"0:{n_ep}"}, "data_path": V21_DATA,
       "video_path": V21_VIDEO, "features": feats}
(out / "meta/info.json").write_text(json.dumps(v21, indent=2))
(out / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes_jsonl))
(out / "meta/tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(task_names)))
(out / "meta/modality.json").write_text(json.dumps(MODALITY, indent=2))
shutil.copy(src / "units.json", out / "units.json")
print(f"TIMING {n_ep} episodes, {frames} frames in {time.time() - T0:.1f} s", flush=True)


# ---- alignment check against the original recorded frames ----
def decode(path, n):
    raw = subprocess.run([f"{FF}/bin/ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, env=FFENV, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 256, 256, 3)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


if args.raw_root:
    units = json.loads((src / "units.json").read_text())
    off = np.array(units["arm_offsets_deg"], np.float32)
    gmin, gmax = units["gripper"]["sim_deg_range"]
    raw_lists = {}
    worst, checked = 99.0, 0
    eps_src = pd.concat([pd.read_parquet(f) for f in sorted((src / "meta/episodes").glob("chunk-*/file-*.parquet"))])
    per = max(1, args.check // len(task_names))  # first AND last episodes of each task: covers far seeks, file-001
    g = eps.groupby("task", sort=False)
    pick = sorted(set(g.head(max(1, per - 1)).index) | set(g.tail(1).index))
    for new_i in pick:
        row = eps.loc[new_i]
        obj = row["task"].removeprefix("put the ").removesuffix(" in the yellow bowl").replace(" ", "_")
        if obj not in raw_lists:  # k-th successful npz of this object == k-th v3 episode of this task
            fs = sorted((Path(args.raw_root) / obj).glob("ep_*.npz"))
            raw_lists[obj] = [f for f in fs if bool(np.load(f)["success"])]
        src_of_task = eps_src[[t[0] == row["task"] for t in eps_src["tasks"]]].sort_values("episode_index")
        k = list(src_of_task["episode_index"]).index(int(row["episode_index"]))  # row["episode_index"] = source index
        npz = np.load(raw_lists[obj][k])
        st = npz["state"][: int(row["length"])]
        u = np.empty_like(st)
        u[:, :5] = np.degrees(st[:, :5]) + off
        u[:, 5] = (np.degrees(st[:, 5]) - gmin) / (gmax - gmin) * 100.0
        dfp = pd.read_parquet(out / V21_DATA.format(episode_chunk=new_i // 1000, episode_index=new_i))
        ds_state = np.stack(dfp["observation.state"].to_numpy())
        assert ds_state.shape == u.shape, (ds_state.shape, u.shape)
        sdiff = float(np.abs(ds_state - u).max())
        assert sdiff < 1e-2, f"state mismatch {sdiff} for episode {new_i} vs {raw_lists[obj][k]}"
        for cam, key in (("observation.images.scene", "scene"), ("observation.images.wrist", "wrist")):
            vid = decode(out / V21_VIDEO.format(episode_chunk=new_i // 1000, video_key=cam, episode_index=new_i), len(u))
            ref = npz[key][: int(row["length"])]
            assert len(vid) == len(ref), (new_i, cam, len(vid), len(ref))
            # mean PSNR over the whole episode at frame shifts -2..2: shift 0 must win (single frames can tie when
            # the camera view is momentarily static, so compare whole-episode means)
            score = {s: float(np.mean([psnr(vid[t], ref[t + s]) for t in range(2, len(ref) - 2)])) for s in range(-2, 3)}
            best = max(score, key=score.get)
            if best != 0:
                raise RuntimeError(f"frame misalignment ep {new_i} {cam}: best shift {best}, mean PSNR by shift {score}")
            worst = min(worst, score[0])
        checked += 1
        print(f"CHECK ep {new_i} <- {raw_lists[obj][k].name} ({obj}): state max|diff|={sdiff:.2e}, frames aligned", flush=True)
    print(f"RAWCHECK {checked} episodes OK, worst frame PSNR vs recorded RGB {worst:.1f} dB", flush=True)
print("DONE", out, flush=True)
