"""STRICT closed-loop evaluation of an SO-101 policy served over the robot-lab socket protocol (GR00T via
tools/gr00t_policy_server.py; FLUX via tools/policy_server.py also works), plus a demo REPLAY mode that checks the
harness itself.

  cd /mnt/weights/ai/isaac/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 uv run --extra teleop python \
      /mnt/work/AI/robot-lab/tools/gr00t_eval.py --object mug --episodes 12 --max_seconds 20 --seed 1000 \
      --units <dataset>/units.json --out <dir> [--port 6100]
  ... --replay <run>/raw/mug   # no server: replay recorded successful demos (layout + joints restored from the npz)
      The replay is the harness check from course Lesson 7.5 step 1: recorded expert actions through the exact eval
      path must mostly come out as STRICT successes, else the units, controller or the strict criteria are wrong.

Why "strict" (course Module 7, "Every success was fake"): the task's own success term (object centre inside the bowl
radius, low, jaw open) fired on episodes where nothing was ever grasped: the bowl bumped onto the object, or a physics
blow-up flung the bowl through the table. Here an episode counts as a STRICT success only if ALL hold:
  1. the task's success term fires (the FLUX-comparable "task success", also reported);
  2. the object was LIFTED: its centre rose >= --lift_rise (3 cm) above its settled start height before that;
  3. the bowl is UPRIGHT (tilt <= 20 deg) and within --bowl_shift (3 cm, xy) of its post-reset position;
  4. no BLOW-UP: |joint vel| <= 30 rad/s, object and bowl above the table, object speed <= 3 m/s (a blow-up ends the
     episode as a failure).
Resets are checked too: a layout whose object starts within 10 cm (xy) of the bowl centre is rejected and re-seeded
(seed + ep + 100000 k); rejections are counted. Per-episode seeds (seed + ep) = the FLUX eval's layouts.
Start state matches the demos: the recorder dropped the first 2 frames after reset, during which the expert held the
arm and opened the jaw toward the 45 deg pre-shape (dataset command 50, first recorded jaw ~16). We do the same before
the first observation, and log the start jaw so the match is checked, not assumed.

Writes results_<object>.json (summary + per-episode: task/strict success and every criterion, latency, layout),
an mp4 per episode (scene | wrist, via the system ffmpeg CLI) and a 6-frame PNG strip for every task-success episode
(so each "success" can be looked at), and diag_<object>_epNN.npz (commanded vs measured joints, TCP-object distance).
"""

import argparse
import json
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--object", default="mug")
ap.add_argument("--episodes", type=int, default=12)
ap.add_argument("--units", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=6100)
ap.add_argument("--seed", type=int, default=1000, help="episode ep uses seed+ep (FLUX eval layouts)")
ap.add_argument("--max_seconds", type=float, default=20.0)
ap.add_argument("--save_video", type=int, default=1)
ap.add_argument("--lift_rise", type=float, default=0.03)
ap.add_argument("--bowl_shift", type=float, default=0.03)
ap.add_argument("--bowl_tilt_deg", type=float, default=20.0)
ap.add_argument("--min_obj_bowl_xy", type=float, default=0.10)
ap.add_argument("--replay", default="", help="raw npz dir: replay recorded successful demos instead of a policy")
ap.add_argument("--hold", action="store_true", help="[NEG-CONTROL] no server: hold the start pose (must never succeed)")
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
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robot_lab.tasks.so101_mug_bowl import SCENE_CAM_EYE, SCENE_CAM_TARGET, _t  # noqa: E402
from robot_lab.tasks.so101_pick_place import CAPTION, object_center_w, set_object  # noqa: E402

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
TAG = args.object.replace(" ", "_")
units = json.loads(Path(args.units).read_text())
OFF = np.array(units["arm_offsets_deg"], np.float32)
GMIN, GMAX = units["gripper"]["sim_deg_range"]
JAW_PRESHAPE_RAD = 0.785  # dataset command 50 = 45 deg: what the expert commands from the first step
VEL_MAX, OBJ_SPEED_MAX, Z_FLOOR = 30.0, 3.0, -0.03


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
cfg.episode_length_s = args.max_seconds + 5.0  # we end episodes ourselves; never let the env time out first
env = gym.make(T, cfg=cfg).unwrapped
dev = env.device
robot, obj_asset, bowl = env.scene["robot"], env.scene["object"], env.scene["bowl"]
jid = robot.find_joints(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"])[0]
ee_idx = robot.find_bodies("gripper")[0][0]
TCP_LOCAL = torch.tensor((0.014, 0.0, -0.085), device=dev)
task = CAPTION.format(args.object)
conn = None if (args.replay or args.hold) else Client(("127.0.0.1", args.port), authkey=b"robot-lab")
origin = env.scene.env_origins[0]


def jpos():
    return _t(robot.data.joint_pos)[0, jid].detach().cpu().numpy()


def obj_c():  # object geometric centre, env frame
    return (object_center_w(env)[0] - origin).cpu().numpy()


def bowl_pos():
    return (_t(bowl.data.root_pos_w)[0] - origin).cpu().numpy()


def bowl_tilt_deg():
    q = _t(bowl.data.root_quat_w)[0:1]  # xyzw
    up = quat_apply(q, torch.tensor([[0.0, 0.0, 1.0]], device=dev))[0]
    return float(torch.rad2deg(torch.arccos(up[2].clamp(-1, 1))))


def set_cam():
    env.scene["scene_cam"].set_world_poses_from_view(torch.tensor(SCENE_CAM_EYE, device=dev) + env.scene.env_origins,
                                                     torch.tensor(SCENE_CAM_TARGET, device=dev) + env.scene.env_origins)


def write_mp4(path, frames):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                          str(path)], stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(np.stack(frames)).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def write_strip(path, frames):
    from PIL import Image

    idx = np.linspace(0, len(frames) - 1, 6).round().astype(int)
    Image.fromarray(np.concatenate([frames[i][:, :256] for i in idx], 1)).save(path)


replay_files = []
if args.replay:
    replay_files = [f for f in sorted(Path(args.replay).glob("ep_*.npz")) if bool(np.load(f)["success"])]
    assert replay_files, f"no successful demos in {args.replay}"


def restore_demo_layout(d):
    """Put robot joints, object and bowl exactly where the recorded demo's first frame had them."""
    q0 = torch.tensor(d["state"][0], device=dev)[None]
    robot.write_joint_position_to_sim_index(position=q0, joint_ids=jid, env_ids=[0])
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0), joint_ids=jid, env_ids=[0])
    for asset, key in ((obj_asset, "object_pose"), (bowl, "bowl_pose")):
        pose = torch.tensor(d[key][0], device=dev)[None].clone()
        pose[:, :3] += origin
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=[0])
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=[0])


results = []
n_ep = min(args.episodes, len(replay_files)) if args.replay else args.episodes
for ep in range(n_ep):
    rejected, k = [], 0
    while True:  # reset, settle like the demos, check the layout
        seed = args.seed + ep + 100000 * k
        obs, _ = env.reset(seed=seed)
        set_cam()
        demo = np.load(replay_files[ep]) if args.replay else None
        if demo is not None:
            restore_demo_layout(demo)
        hold = torch.tensor(jpos(), device=dev)[None].clone()
        if demo is None:
            hold[0, 5] = JAW_PRESHAPE_RAD  # the expert's first command: open toward the pre-shape
        for _ in range(2):  # 2 dropped frames, as in the recorder (the first render after reset is stale)
            obs, *_ = env.step(hold if demo is None else torch.tensor(demo["state"][0], device=dev)[None])
        if demo is not None:
            restore_demo_layout(demo)  # undo the 2 settle steps' drift: start exactly at the demo's frame 0
        d_xy = float(np.linalg.norm(obj_c()[:2] - bowl_pos()[:2]))
        if demo is not None or (d_xy >= args.min_obj_bowl_xy and bowl_tilt_deg() <= args.bowl_tilt_deg):
            break
        rejected.append({"seed": seed, "obj_bowl_xy": round(d_xy, 4), "bowl_tilt_deg": round(bowl_tilt_deg(), 1)})
        k += 1
        if k > 20:
            raise RuntimeError(f"20 rejected resets in a row for episode {ep}: {rejected[-3:]}")
    if conn is not None:
        conn.send({"cmd": "reset"}); conn.recv()
    start = {"obj": obj_c().round(4).tolist(), "bowl": bowl_pos().round(4).tolist(), "obj_bowl_xy": round(d_xy, 4),
             "jaw_units": round(float(to_units(jpos())[5]), 2)}
    z0, bowl0 = obj_c()[2], bowl_pos()
    bowl_last = (bowl0, bowl_tilt_deg())
    max_rise, lifted_step, blowup, task_success = 0.0, None, None, False
    frames, plan_ms, log_cmd, log_meas, log_dist = [], [], [], [], []
    t_wall = time.time()
    # replay: the demo's final action label is the post-reset target (recorder bug, see to_gr00t.py), so replay
    # actions[:-1] and then hold the last real command (RELEASE) for up to 1 s while the success term settles
    n_steps = len(demo["action"]) - 1 + 30 if demo is not None else int(args.max_seconds * 30)
    for step in range(n_steps):
        scene = obs["rgb_camera"]["scene"][0, ..., :3].to(torch.uint8).cpu().numpy()
        wrist = obs["rgb_camera"]["wrist"][0, ..., :3].to(torch.uint8).cpu().numpy()
        if demo is not None:
            q = demo["action"][min(step, len(demo["action"]) - 2)].astype(np.float32)
            cmd_u = to_units(q)
        elif args.hold:
            q = hold[0].cpu().numpy().astype(np.float32)
            cmd_u = to_units(q)
        else:
            conn.send({"cmd": "act", "scene": scene, "wrist": wrist, "state": to_units(jpos()), "task": task})
            r = conn.recv()
            if r.get("planned", r["ms"] > 50):
                plan_ms.append(r["ms"])
            cmd_u = np.asarray(r["action"], np.float32)
            q = from_units(cmd_u)
        ee = _t(robot.data.body_pose_w)[0, ee_idx]
        tcp = ee[:3] + quat_apply(ee[3:7][None], TCP_LOCAL[None])[0]
        log_dist.append(float(torch.linalg.norm(tcp - object_center_w(env)[0])))
        log_cmd.append(cmd_u)
        log_meas.append(to_units(jpos()))
        obs, rew, term, trunc, info = env.step(torch.tensor(q, device=dev)[None])
        if args.save_video:
            frames.append(np.concatenate([scene, wrist], 1))
        # A terminating step (success / object dropped / timeout) AUTO-RESETS the env inside env.step(), so every
        # state read after it belongs to the NEXT layout. Criteria therefore use the last pre-terminal reading
        # (1/30 s earlier) and are never updated on the terminal step.
        task_success = bool(env.termination_manager.get_term("success")[0])
        if task_success or bool(term[0]) or bool(trunc[0]):
            break
        oc = obj_c()
        rise = float(oc[2] - z0)
        if rise > max_rise:
            max_rise = rise
        if lifted_step is None and rise >= args.lift_rise:
            lifted_step = step
        bowl_last = (bowl_pos(), bowl_tilt_deg())
        vmax = float(_t(robot.data.joint_vel)[0].abs().max())
        ospeed = float(torch.linalg.norm(_t(obj_asset.data.root_lin_vel_w)[0]))
        if vmax > VEL_MAX or oc[2] < Z_FLOOR or bowl_last[0][2] < Z_FLOOR or ospeed > OBJ_SPEED_MAX:
            blowup = {"step": step, "joint_vel_max": round(vmax, 2), "obj_z": round(float(oc[2]), 4),
                      "bowl_z": round(float(bowl_last[0][2]), 4), "obj_speed": round(ospeed, 2)}
            break
    bowl_shift = float(np.linalg.norm(bowl_last[0][:2] - bowl0[:2]))
    tilt = bowl_last[1]
    checks = {"task_success": task_success, "lifted": lifted_step is not None,
              "bowl_ok": bowl_shift <= args.bowl_shift and tilt <= args.bowl_tilt_deg, "no_blowup": blowup is None}
    strict = all(checks.values())
    ms = np.array(plan_ms) if plan_ms else None
    rec = {"episode": ep, "object": args.object, "seed": seed, "rejected_resets": rejected, "success": task_success,
           "strict_success": strict, "checks": checks, "max_rise_m": round(max_rise, 4), "lifted_step": lifted_step,
           "bowl_shift_m": round(bowl_shift, 4), "bowl_tilt_deg": round(tilt, 1), "blowup": blowup, "start": start,
           "steps": step + 1, "seconds_sim": round((step + 1) / 30, 2), "wall_s": round(time.time() - t_wall, 1),
           "plan_ms_median": float(np.median(ms)) if ms is not None else None,
           "plan_ms_p90": float(np.percentile(ms, 90)) if ms is not None else None,
           "n_plans": int(len(plan_ms)), "replay": str(replay_files[ep]) if demo is not None else None}
    results.append(rec)
    print("EVAL", json.dumps(rec), flush=True)
    stem = f"{TAG}_ep{ep:02d}_{'strict' if strict else ('task' if task_success else 'fail')}"
    np.savez(OUT / f"diag_{TAG}_ep{ep:02d}.npz", cmd_units=np.stack(log_cmd), meas_units=np.stack(log_meas),
             tcp_obj_dist=np.array(log_dist), strict=strict, task_success=task_success)
    if args.save_video and frames:
        write_mp4(OUT / f"{stem}.mp4", frames)
        if task_success:
            write_strip(OUT / f"{stem}_strip.png", frames)

lat = [r["plan_ms_median"] for r in results if r["plan_ms_median"] is not None]
summary = {"object": args.object, "task": task, "mode": "replay" if args.replay else ("hold" if args.hold else "policy"),
           "episodes": len(results), "successes": sum(r["success"] for r in results),
           "strict_successes": sum(r["strict_success"] for r in results),
           "success_rate": sum(r["success"] for r in results) / max(1, len(results)),
           "strict_success_rate": sum(r["strict_success"] for r in results) / max(1, len(results)),
           "rejected_resets": sum(len(r["rejected_resets"]) for r in results),
           "start_jaw_units_median": float(np.median([r["start"]["jaw_units"] for r in results])),
           "plan_ms_median": float(np.median(lat)) if lat else None, "seed": args.seed, "max_seconds": args.max_seconds,
           "criteria": {"lift_rise_m": args.lift_rise, "bowl_shift_m": args.bowl_shift,
                        "bowl_tilt_deg": args.bowl_tilt_deg, "min_obj_bowl_xy_m": args.min_obj_bowl_xy,
                        "joint_vel_max": VEL_MAX, "obj_speed_max": OBJ_SPEED_MAX, "z_floor": Z_FLOOR}}
(OUT / f"results_{TAG}.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
print("SUMMARY", json.dumps(summary), flush=True)
if conn is not None:
    conn.close()
env.close()
app.close()
