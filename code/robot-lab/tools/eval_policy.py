"""Closed-loop evaluation of a FLUX SO-101 policy (served by tools/policy_server.py) in Isaac Lab.

  OMNI_KIT_ACCEPT_EULA=YES uv run --extra teleop python /mnt/work/AI/robot-lab/tools/eval_policy.py \
      --object mug --episodes 10 --units <dataset>/units.json --out <dir> [--port 6100] [--seed 1000]

One env, sequential episodes. Every control tick (30 Hz): scene + wrist RGB and the measured joint state (converted to
dataset units with units.json) go to the server; the returned absolute command is converted back to radians and
applied as joint-position targets. Success is the task's own termination (object in bowl, jaw released).
Writes results.json (per-episode success/steps/plan latency) and an mp4 per episode (scene | wrist).
"""

import argparse
import json
import sys
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--object", default="mug")
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--units", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=6100)
ap.add_argument("--seed", type=int, default=1000, help="eval seeds differ from demo seeds")
ap.add_argument("--max_seconds", type=float, default=20.0)
ap.add_argument("--save_video", type=int, default=1)
args = ap.parse_args()
app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app

sys.path.insert(0, "/mnt/work/AI/robot-lab")
from multiprocessing.connection import Client  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import robot_lab.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robot_lab.tasks.so101_mug_bowl import SCENE_CAM_EYE, SCENE_CAM_TARGET  # noqa: E402
from robot_lab.tasks.so101_pick_place import CAPTION, set_object  # noqa: E402

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
units = json.loads(Path(args.units).read_text())
OFF = np.array(units["arm_offsets_deg"], np.float32)
GMIN, GMAX = units["gripper"]["sim_deg_range"]


def to_units(q):  # (6,) rad -> dataset units
    u = np.empty(6, np.float32)
    u[:5] = np.degrees(q[:5]) + OFF
    u[5] = (np.degrees(q[5]) - GMIN) / (GMAX - GMIN) * 100.0
    return u


def from_units(u):  # (6,) dataset units -> rad
    q = np.empty(6, np.float32)
    q[:5] = np.radians(u[:5] - OFF)
    q[5] = np.radians(u[5] / 100.0 * (GMAX - GMIN) + GMIN)
    return q


T = "RobotLab-SO101-PickPlace-Joint-v0"
cfg = set_object(parse_env_cfg(T, device="cuda:0", num_envs=1), args.object)
cfg.seed = args.seed
cfg.episode_length_s = args.max_seconds
env = gym.make(T, cfg=cfg).unwrapped
robot = env.scene["robot"]
jid = robot.find_joints(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"])[0]
task = CAPTION.format(args.object)
conn = Client(("127.0.0.1", args.port), authkey=b"robot-lab")


def jpos():
    q = robot.data.joint_pos
    q = q.torch if hasattr(q, "torch") else q
    return q[0, jid].detach().cpu().numpy()


results = []
for ep in range(args.episodes):
    obs, _ = env.reset(seed=args.seed + ep)
    o = env.scene.env_origins
    env.scene["scene_cam"].set_world_poses_from_view(torch.tensor(SCENE_CAM_EYE, device=env.device) + o,
                                                     torch.tensor(SCENE_CAM_TARGET, device=env.device) + o)
    # settle two frames so the first observation is a fresh render (the first render after reset is stale)
    hold = torch.tensor(jpos(), device=env.device)[None]
    for _ in range(2):
        obs, *_ = env.step(hold)
    conn.send({"cmd": "reset"}); conn.recv()
    frames, plan_ms, success, t0 = [], [], False, time.time()
    for step in range(int(args.max_seconds * 30)):
        scene = obs["rgb_camera"]["scene"][0, ..., :3].to(torch.uint8).cpu().numpy()
        wrist = obs["rgb_camera"]["wrist"][0, ..., :3].to(torch.uint8).cpu().numpy()
        conn.send({"cmd": "act", "scene": scene, "wrist": wrist, "state": to_units(jpos()), "task": task})
        r = conn.recv()
        plan_ms.append(r["ms"])
        q = from_units(np.asarray(r["action"], np.float32))
        obs, rew, term, trunc, info = env.step(torch.tensor(q, device=env.device)[None])
        if args.save_video:
            frames.append(np.concatenate([scene, wrist], 1))
        if bool(env.termination_manager.get_term("success")[0]):
            success = True
            break
        if bool(term[0]) or bool(trunc[0]):
            break
    ms = np.array(plan_ms)
    replans = ms[ms > 50]  # ticks that ran a full denoise (others pop the queued chunk)
    rec = {"episode": ep, "object": args.object, "success": success, "steps": step + 1, "seconds_sim": (step + 1) / 30,
           "wall_s": round(time.time() - t0, 1), "plan_ms_median": float(np.median(replans)) if len(replans) else None,
           "n_plans": int(len(replans))}
    results.append(rec)
    print("EVAL", json.dumps(rec), flush=True)
    if args.save_video and frames:
        import imageio.v3 as iio

        try:
            iio.imwrite(OUT / f"{args.object.replace(' ', '_')}_ep{ep:02d}_{'ok' if success else 'fail'}.mp4",
                        np.stack(frames), fps=30, codec="libx264")
        except Exception:  # noqa: BLE001  (no ffmpeg plugin in the Isaac venv -> keep raw frames)
            np.save(OUT / f"{args.object.replace(' ', '_')}_ep{ep:02d}_{'ok' if success else 'fail'}.npy", np.stack(frames))
summary = {"object": args.object, "episodes": len(results), "successes": sum(r["success"] for r in results),
           "success_rate": sum(r["success"] for r in results) / max(1, len(results)), "task": task}
(OUT / f"results_{args.object.replace(' ', '_')}.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
print("SUMMARY", json.dumps(summary), flush=True)
env.close()
app.close()
