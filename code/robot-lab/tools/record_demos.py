"""Scripted-expert demo recorder for RobotLab-SO101-MugBowl (Isaac Lab 3.0).

Expert: a vectorized per-env state machine on privileged object poses, commanding IK-Abs gripper poses
(top-down, yaw radial from the base) with speed-limited waypoints:
  PREGRASP -> DESCEND -> CLOSE -> LIFT -> CARRY -> LOWER -> RELEASE -> RETREAT -> SETTLE
Recorded per step (30 Hz): scene + wrist RGB (256x256 uint8), joint state (6), commanded joint targets (6),
timestamps, phase, privileged object poses. One .npz per finished episode (successes and failures, flagged).

Run from the IsaacLab checkout:
  OMNI_KIT_ACCEPT_EULA=YES uv run --extra teleop python /mnt/work/AI/robot-lab/tools/record_demos.py \
      --num_envs 8 --episodes 16 --out /mnt/weights/ai/robot-lab-data/so101_mug_bowl/raw
"""

import argparse
import json
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--episodes", type=int, default=16, help="stop after this many *finished* episodes")
parser.add_argument("--out", default="/mnt/weights/ai/robot-lab-data/so101_mug_bowl/raw")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_speed", type=float, default=0.20, help="TCP speed limit [m/s]")
parser.add_argument("--object", default="mug", help="object name from robot_lab.tasks.so101_pick_place")
parser.add_argument("--task", default="RobotLab-SO101-PickPlace-IK-Abs-v0")
args = parser.parse_args()
app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app

sys.path.insert(0, "/mnt/work/AI/robot-lab")
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import robot_lab.tasks  # noqa: E402,F401
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robot_lab.tasks.so101_mug_bowl import SCENE_CAM_EYE, SCENE_CAM_TARGET  # noqa: E402
from robot_lab.tasks.so101_pick_place import CAPTION, object_center_w, object_geometry, set_object  # noqa: E402

TASK = args.task
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

# ---- expert constants (world/env frame, table top at z = 0) ----
TCP_OFFSET = 0.09  # [m] depth of the approach axis used for pre-grasp back-off (kept for metadata)
# Grasp point in the gripper-link frame, from physics-accurate fingertip geometry (tools/probe_fingertips.py):
# static finger tip x~-0.011, jaw closes to x~+0.011, tips at z~-0.10. A 4.35 cm mug pressed against the static
# finger has its center at x~+0.014; center it 1.5 cm above the tips.
TCP_LOCAL = (0.014, 0.0, -0.085)
GRIP_PRESHAPE = 0.55  # closedness during approach: jaw ~45 deg open instead of 100 deg (smaller sweep)
Z_HOVER, Z_CARRY, Z_BOWL_RELEASE, Z_RETREAT = 0.07, 0.10, 0.075, 0.12  # lowered: 5-DOF top-down reach
GRASP_DZ = 0.004  # grasp slightly above the mug center
PITCH_MAX_DEG, R_VERTICAL, R_FULL_PITCH = 60.0, 0.17, 0.28  # tilt grows with reach; far grasps fell 2-3 cm short at 45 deg
APPROACH_BACKOFF = 0.07  # [m] pre-grasp point, backed off along the approach axis
LIFT_CHECK_Z = 0.045  # [m] mug center must be above this after LIFT, else the grasp missed
POS_TOL = 0.012  # [m] waypoint reached
GOOD_ENOUGH, SETTLE_STEPS = 0.025, 40  # non-contact phases only: advance if within 2.5 cm after 40 steps
HOLD_CLOSE, HOLD_RELEASE, HOLD_SETTLE = 20, 12, 15  # steps
PHASES = ["PREGRASP", "DESCEND", "CLOSE", "LIFT", "CARRY", "LOWER", "RELEASE", "RETREAT", "SETTLE", "DONE"]
P = {n: i for i, n in enumerate(PHASES)}
PHASE_BUDGET = 150  # steps per phase before the episode is abandoned (expert failure)

cfg = set_object(parse_env_cfg(TASK, device="cuda:0", num_envs=args.num_envs), args.object)
cfg.seed = args.seed
CAPTION_TEXT = CAPTION.format(args.object)
# The mug's bbox center includes the handle; grasp its cup body instead (measured, tools/probe_mug_geom.py).
GRASP_CENTER_OVERRIDE = {"mug": (0.0, 0.0223, 0.0019)}
TCP_MIN_Z = 0.017  # [m] keep fingertips (1.5 cm below the TCP) off the table for flat objects
env = gym.make(TASK, cfg=cfg).unwrapped
obs, _ = env.reset(seed=args.seed)
N, dev, dt = env.num_envs, env.device, env.step_dt
origins = env.scene.env_origins
env.scene["scene_cam"].set_world_poses_from_view(
    torch.tensor(SCENE_CAM_EYE, device=dev) + origins, torch.tensor(SCENE_CAM_TARGET, device=dev) + origins)

robot, mug, bowl = env.scene["robot"], env.scene["object"], env.scene["bowl"]
_center_local, _extents_local = object_geometry(env)
if args.object in GRASP_CENTER_OVERRIDE:
    _center_local = torch.tensor(GRASP_CENTER_OVERRIDE[args.object], device=dev).expand(N, 3).clone()
    env._geom_object = (_center_local, _extents_local)


def mug_body_center_w(env):  # grasp center of the current object (name kept from the mug-only version)
    return object_center_w(env)
ee_idx = robot.find_bodies("gripper")[0][0]
joint_ids = robot.find_joints(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"])[0]
joint_names = [robot.joint_names[i] for i in joint_ids]


def yaw_quat_xyzw(yaw):
    z = torch.zeros_like(yaw)
    return torch.stack([z, z, torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1)


def pitch_for(xy):
    """Approach tilt [rad] from the TCP's radial distance to the base: 0 at <= R_VERTICAL, PITCH_MAX at >= R_FULL."""
    r = torch.linalg.norm(xy, dim=-1)
    frac = ((r - R_VERTICAL) / (R_FULL_PITCH - R_VERTICAL)).clamp(0.0, 1.0)
    return frac * np.deg2rad(PITCH_MAX_DEG)


def grasp_quat_xyzw(yaw, pitch, roll=None):
    """World orientation: yaw about Z, tilt the approach axis (gripper -Z) outward by `pitch`, then roll about the
    approach axis (wrist_roll) so the jaw closes across the object's narrow side."""
    from isaaclab.utils.math import quat_mul

    h = -pitch / 2
    z = torch.zeros_like(h)
    q = quat_mul(yaw_quat_xyzw(yaw), torch.stack([z, torch.sin(h), z, torch.cos(h)], dim=-1))
    if roll is not None:
        q = quat_mul(q, yaw_quat_xyzw(roll))
    return q


def approach_dir(yaw, pitch):
    """Unit vector the fingertips point along (gripper local -Z) in the world."""
    from isaaclab.utils.math import quat_apply

    return quat_apply(grasp_quat_xyzw(yaw, pitch), torch.tensor([0.0, 0.0, -1.0], device=dev).expand(len(yaw), 3))


def tcp_pos_w():
    """Current TCP position (env frame): gripper origin + R @ TCP_LOCAL."""
    from isaaclab.utils.math import quat_apply

    ee = robot.data.body_pose_w[:, ee_idx]
    off = torch.tensor(TCP_LOCAL, device=dev).expand(N, 3)
    return ee[:, :3] + quat_apply(ee[:, 3:7], off) - origins


def phase_target(phase, grasp_yaw):
    """TCP goal (env frame) per env for its phase."""
    m = mug_body_center_w(env) - origins  # cup body center, not the asset root
    b = (bowl.data.root_pos_w.torch if hasattr(bowl.data.root_pos_w, "torch") else bowl.data.root_pos_w) - origins
    tgt = tcp_pos_w().clone()
    sel = lambda name: phase == P[name]  # noqa: E731
    grasp_pt = torch.cat([m[:, :2], (m[:, 2:3] + GRASP_DZ).clamp_min(TCP_MIN_Z)], -1)
    pre = grasp_pt - approach_dir(grasp_yaw, pitch_for(grasp_pt[:, :2])) * APPROACH_BACKOFF
    tgt[sel("PREGRASP")] = pre[sel("PREGRASP")]
    tgt[sel("DESCEND")] = grasp_pt[sel("DESCEND")]
    tgt[sel("LIFT")] = torch.cat([m[:, :2], torch.full((N, 1), Z_CARRY, device=dev)], -1)[sel("LIFT")]
    tgt[sel("CARRY")] = torch.cat([b[:, :2], torch.full((N, 1), Z_CARRY, device=dev)], -1)[sel("CARRY")]
    tgt[sel("LOWER")] = torch.cat([b[:, :2], torch.full((N, 1), Z_BOWL_RELEASE, device=dev)], -1)[sel("LOWER")]
    tgt[sel("RETREAT")] = torch.cat([b[:, :2], torch.full((N, 1), Z_RETREAT, device=dev)], -1)[sel("RETREAT")]
    return tgt


# ---- per-env state ----
phase = torch.zeros(N, dtype=torch.long, device=dev)
phase_steps = torch.zeros(N, dtype=torch.long, device=dev)
cmd_tcp = tcp_pos_w().clone()  # speed-limited commanded TCP (env frame)
grip = torch.zeros(N, device=dev)  # closedness c in [0, 1]
grasp_yaw = torch.zeros(N, device=dev)
grasp_roll = torch.zeros(N, device=dev)
buffers = [dict(scene=[], wrist=[], state=[], action=[], phase=[], mug=[], bowl=[]) for _ in range(N)]
finished, successes, t0, step_count = 0, 0, time.time(), 0
ep_index = 0


def obj_quat():
    q = mug.data.root_quat_w
    return q.torch if hasattr(q, "torch") else q


def reset_env_state(ids):
    phase[ids] = 0
    phase_steps[ids] = 0
    cmd_tcp[ids] = tcp_pos_w()[ids]
    grip[ids] = 0.0
    m = (mug_body_center_w(env) - origins)[ids]
    grasp_yaw[ids] = torch.atan2(m[:, 1], m[:, 0])
    # jaw (gripper local x) should close across the object's narrow horizontal axis; +-180 deg symmetric
    qo = obj_quat()[ids]
    obj_yaw = torch.atan2(2 * (qo[:, 3] * qo[:, 2] + qo[:, 0] * qo[:, 1]), 1 - 2 * (qo[:, 1] ** 2 + qo[:, 2] ** 2))
    narrow_is_y = (_extents_local[ids, 1] < _extents_local[ids, 0]).float()
    theta_n = obj_yaw + narrow_is_y * (np.pi / 2)
    d = theta_n - grasp_yaw[ids]
    grasp_roll[ids] = torch.atan2(torch.sin(2 * d), torch.cos(2 * d)) / 2  # wrap to [-90, 90] deg
    for i in ids.tolist():
        buffers[i] = dict(scene=[], wrist=[], state=[], action=[], phase=[], mug=[], bowl=[])


reset_env_state(torch.arange(N, device=dev))
max_step = args.max_speed * dt

while finished < args.episodes and app.is_running():
    # ---- expert: advance phases ----
    tgt = phase_target(phase, grasp_yaw)
    reached = torch.linalg.norm(tcp_pos_w() - tgt, dim=-1) < POS_TOL
    hold_done = ((phase == P["CLOSE"]) & (phase_steps >= HOLD_CLOSE)) | \
                ((phase == P["RELEASE"]) & (phase_steps >= HOLD_RELEASE)) | \
                ((phase == P["SETTLE"]) & (phase_steps >= HOLD_SETTLE))
    moving = ~torch.isin(phase, torch.tensor([P["CLOSE"], P["RELEASE"], P["SETTLE"], P["DONE"]], device=dev))
    err = torch.linalg.norm(tcp_pos_w() - tgt, dim=-1)
    precise = torch.isin(phase, torch.tensor([P["DESCEND"]], device=dev))
    good_enough = moving & ~precise & (phase_steps > SETTLE_STEPS) & (err < GOOD_ENOUGH)
    advance = (moving & reached) | hold_done | good_enough
    phase = torch.where(advance, phase + 1, phase)
    phase_steps = torch.where(advance, torch.zeros_like(phase_steps), phase_steps + 1)
    grip = torch.where(phase <= P["DESCEND"], torch.full_like(grip, GRIP_PRESHAPE), grip)
    grip = torch.where(phase == P["CLOSE"], torch.ones_like(grip), grip)
    # release to the pre-shape opening (~45 deg -> ~50 in dataset units), like real SO-101 operators (package q99 = 54),
    # instead of snapping fully open (100)
    grip = torch.where(phase == P["RELEASE"], torch.full_like(grip, GRIP_PRESHAPE), grip)

    # ---- speed-limited TCP command -> gripper-origin pose in the robot base frame ----
    tgt = phase_target(phase, grasp_yaw)
    delta = tgt - cmd_tcp
    dist = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-9)
    cmd_tcp = cmd_tcp + delta * torch.clamp(max_step / dist, max=1.0)
    yaw = torch.where(phase >= P["CARRY"], torch.atan2(cmd_tcp[:, 1], cmd_tcp[:, 0]), grasp_yaw)
    pitch = pitch_for(cmd_tcp[:, :2])  # tilt follows reach, continuously
    q_w = grasp_quat_xyzw(yaw, pitch, grasp_roll)
    from isaaclab.utils.math import quat_apply  # noqa: E402

    origin_w = cmd_tcp + origins - quat_apply(q_w, torch.tensor(TCP_LOCAL, device=dev).expand(N, 3))
    root = robot.data.root_pose_w
    pos_b, quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], origin_w, q_w)
    action = torch.cat([pos_b, quat_b, grip[:, None]], dim=-1)

    # ---- record observation *before* stepping (obs_t, action_t) ----
    o = obs
    state = robot.data.joint_pos[:, joint_ids].detach().cpu().numpy()
    scene_img = o["rgb_camera"]["scene"][..., :3].to(torch.uint8).cpu().numpy()
    wrist_img = o["rgb_camera"]["wrist"][..., :3].to(torch.uint8).cpu().numpy()
    mug_p = o["policy"]["object_pose"].cpu().numpy()
    bowl_p = o["policy"]["bowl_pose"].cpu().numpy()

    obs, rew, term, trunc, info = env.step(action)
    step_count += 1
    # commanded joint targets produced by the IK action term this step = the policy's action label
    joint_cmd = robot.data.joint_pos_target[:, joint_ids].detach().cpu().numpy()
    ph = phase.cpu().numpy()
    for i in range(N):
        b = buffers[i]
        b["scene"].append(scene_img[i]); b["wrist"].append(wrist_img[i]); b["state"].append(state[i])
        b["action"].append(joint_cmd[i]); b["phase"].append(ph[i]); b["mug"].append(mug_p[i])
        b["bowl"].append(bowl_p[i])

    # ---- episode ends: success/timeout/drop from the env, or expert stall ----
    grasp_missed = (phase == P["CARRY"]) & ((mug_body_center_w(env) - origins)[:, 2] < LIFT_CHECK_Z)
    stalled = (phase_steps > PHASE_BUDGET) | grasp_missed
    done = (term | trunc | stalled).nonzero().flatten()
    if len(done):
        success_flags = env.termination_manager.get_term("success")
        for i in done.tolist():
            b = buffers[i]
            ok = bool(success_flags[i]) and not bool(stalled[i]) and int(phase[i]) >= P["RELEASE"]
            if bool(stalled[i]):
                print(f"STALL env={i} phase={PHASES[int(phase[i])]} tcp={tcp_pos_w()[i].cpu().numpy().round(3)} "
                      f"target={phase_target(phase, grasp_yaw)[i].cpu().numpy().round(3)}", flush=True)
            for k in b:  # drop the first 2 frames: the first render after a reset is stale
                b[k] = b[k][2:]
            if len(b["state"]) > 10 and finished < args.episodes:
                np.savez_compressed(
                    OUT / f"ep_{ep_index:05d}.npz",
                    scene=np.stack(b["scene"]), wrist=np.stack(b["wrist"]),
                    state=np.stack(b["state"]).astype(np.float32), action=np.stack(b["action"]).astype(np.float32),
                    timestamp=(np.arange(len(b["state"])) * dt).astype(np.float32),
                    phase=np.array(b["phase"], np.int8), object_pose=np.stack(b["mug"]).astype(np.float32),
                    bowl_pose=np.stack(b["bowl"]).astype(np.float32), success=np.array(ok),
                    task=np.array(CAPTION_TEXT), object=np.array(args.object),
                )
                print(f"EPISODE {ep_index:05d} env={i} steps={len(b['state'])} success={ok} "
                      f"end_phase={PHASES[int(phase[i])]} stalled={bool(stalled[i])}", flush=True)
                ep_index += 1
                finished += 1
                successes += int(ok)
        if stalled.any():  # env did not reset itself: force it
            env._reset_idx(stalled.nonzero().flatten())
            obs = env.observation_manager.compute()
        reset_env_state(done)

meta = {
    "task": TASK, "fps": round(1 / dt, 3), "joint_names": joint_names, "state_units": "rad",
    "action": "commanded joint position targets [rad] from the IK-Abs action term (gripper last)",
    "cameras": {"scene": "fixed, 256x256 RGB", "wrist": "gripper-mounted, 256x256 RGB"},
    "caption": CAPTION_TEXT, "object": args.object, "episodes": finished, "successes": successes,
    "expert": {"tcp_local_m": TCP_LOCAL, "grip_preshape": GRIP_PRESHAPE, "pitch_max_deg": PITCH_MAX_DEG, "r_vertical": R_VERTICAL, "r_full_pitch": R_FULL_PITCH, "max_speed_mps": args.max_speed, "phases": PHASES},
    "seed": args.seed, "wall_s": round(time.time() - t0, 1), "sim_steps": step_count,
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print("SUMMARY", json.dumps(meta), flush=True)
env.close()
app.close()
