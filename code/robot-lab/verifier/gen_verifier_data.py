"""Verifier-VLM dataset generator: scripted SO-101 episodes with deliberate outcome diversity, labeled from sim state.

One object per process (Isaac Lab 3.0 EA clones one object type per run, see robot_lab.tasks.so101_pick_place).
All envs of a process run in synchronous ROUNDS: reset everything -> domain randomization (dome light, table cloth
colour, scene-camera jitter, object/bowl pose from the env's reset event, optional tipped bowl) -> settle -> every
env runs its own scenario script (success, hover-never-release, drop mid-carry, release next to the bowl, missed
grasp, knocked bowl, tipped bowl, regrasp, ...) -> every scenario ends with the skill-style clear finish (retreat
up + ~1 s settle) before its FINAL snapshot.

Snapshots (scene | wrist side by side, 512x256 PNG, the format of vlm-verifier-test/set*.json):
  before (after the start settle), after_lift (pick_up check), after_move (move_to check), release_instant (the
  ambiguous frame at the moment of release, like set.json v1), mid (one random step), final (after retreat+settle).
Each snapshot carries the RAW privileged state (object centre/quat/velocity, bowl pose, TCP, jaw angle, commanded
grip). Labels (in_bowl, held, over_bowl, bowl_upright, on_table, ...) are derived OFFLINE by build_dataset.py from
that state, so thresholds can be audited/changed without re-simulating. Nothing here is labeled by a VLM.

The expert motion (TCP waypoints, IK-Abs pose commands, grasp yaw/roll, approach tilt) is copied from
tools/record_demos.py (night-2 version) -- read-only reuse; that file runs at import so it cannot be imported.

Run (from the IsaacLab checkout, GPU chosen by CUDA_VISIBLE_DEVICES):
  uv run --no-sync python /mnt/work/AI/robot-lab/verifier/gen_verifier_data.py --object "red cylinder" \
      --num_envs 32 --rounds 2 --seed 11 --out /mnt/weights/ai/robot-lab-data/verifier-ft/data/raw/red_cylinder
"""

import argparse
import json
import math
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--object", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--rounds", type=int, default=2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", required=True)
parser.add_argument("--max_round_steps", type=int, default=900, help="30 Hz steps per round before truncation")
parser.add_argument("--max_speed", type=float, default=0.20)
parser.add_argument("--no_dr", action="store_true", help="disable light/cloth/camera randomization")
parser.add_argument("--scenarios", default="", help="comma list to restrict scenarios (pilot/debug)")
args = parser.parse_args()
app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app

sys.path.insert(0, "/mnt/work/AI/robot-lab")
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import robot_lab.tasks  # noqa: E402,F401
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg  # noqa: E402
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg  # noqa: E402
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd  # noqa: E402
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg  # noqa: E402
from isaaclab.sim.utils import clone  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robot_lab.tasks.so101_mug_bowl import BOWL_RADIUS, SCENE_CAM_EYE, SCENE_CAM_TARGET, _t  # noqa: E402
from robot_lab.tasks.so101_pick_place import ALL_OBJECTS, object_center_w, object_geometry, set_object  # noqa: E402

# ------------------------------------------------------------------------------------------------------------------
# Objects. "env": defined in robot_lab.tasks.so101_pick_place (same usd/scale as the GR00T demos and evals).
# "usd": asset already used as a rigid body elsewhere in Isaac Lab (stack task cubes). "usd_phys": YCB asset WITHOUT
# physics schemas; our spawner authors rigid body + mass + convex-hull colliders. "shape": Isaac Lab primitive.
# Held-out objects (mustard bottle, cracker box) must only ever be generated with --split test (pipeline enforces).
# ------------------------------------------------------------------------------------------------------------------
YCB_AA = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned"
OBJECTS = {
    "mug": dict(kind="env"), "blue block": dict(kind="env"), "soup can": dict(kind="env"),
    "sugar box": dict(kind="env"), "mustard bottle": dict(kind="env"), "cracker box": dict(kind="env"),
    "red block": dict(kind="usd", usd=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd", scale=0.8),
    "green block": dict(kind="usd", usd=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd", scale=0.8),
    "tuna can": dict(kind="usd_phys", usd=f"{YCB_AA}/007_tuna_fish_can.usd"),
    "pudding box": dict(kind="usd_phys", usd=f"{YCB_AA}/008_pudding_box.usd"),
    "gelatin box": dict(kind="usd_phys", usd=f"{YCB_AA}/009_gelatin_box.usd"),
    "potted meat can": dict(kind="usd_phys", usd=f"{YCB_AA}/010_potted_meat_can.usd"),
    "banana": dict(kind="usd_phys", usd=f"{YCB_AA}/011_banana.usd"),
    "foam brick": dict(kind="usd_phys", usd=f"{YCB_AA}/061_foam_brick.usd"),
    "marker": dict(kind="usd_phys", usd=f"{YCB_AA}/040_large_marker.usd"),
    "red cylinder": dict(kind="shape", shape="cylinder", radius=0.017, height=0.06, color=(0.80, 0.08, 0.08)),
    "orange ball": dict(kind="shape", shape="sphere", radius=0.021, color=(0.95, 0.45, 0.05)),
    "purple box": dict(kind="shape", shape="cuboid", size=(0.032, 0.045, 0.065), color=(0.45, 0.15, 0.65)),
}
HELDOUT = {"mustard bottle", "cracker box"}
FIT_NARROW, FIT_MAX_XY, FIT_MAX_Z = 0.034, 0.10, 0.09  # [m] auto-scale targets for usd_phys objects (jaw ~5 cm)

# ---- expert constants: copied from tools/record_demos.py (night-2) ----
TCP_LOCAL = (0.014, 0.0, -0.085)
GRIP_PRESHAPE = 0.55
Z_CARRY, Z_BOWL_RELEASE, Z_RETREAT = 0.10, 0.075, 0.12
GRASP_DZ = 0.004
PITCH_MAX_DEG, R_VERTICAL, R_FULL_PITCH = 60.0, 0.17, 0.28
APPROACH_BACKOFF = 0.07
POS_TOL = 0.012
GOOD_ENOUGH, SETTLE_STEPS = 0.025, 40
HOLD_CLOSE, HOLD_RELEASE = 20, 12
SEG_BUDGET = 150  # steps before a waypoint is abandoned (recorded as a stall; the script continues)
GRASP_CENTER_OVERRIDE = {"mug": (0.0, 0.0223, 0.0019)}
TCP_MIN_Z = 0.017
# ---- verifier-specific timing (operator 2026-10-06: skills end with a clear retreat + ~1 s settle) ----
START_SETTLE = 24   # steps after reset before the 'before' snapshot (objects drop ~0-3 mm, stale renders flushed)
FINAL_SETTLE = 30   # 1.0 s at 30 Hz after the retreat, then the FINAL snapshot
TASK = "RobotLab-SO101-PickPlace-IK-Abs-v0"

# Weighted toward the hard negatives (coordinator, 2026-10-06): lifted and held over/near the bowl but never released,
# and near-misses (released over the rim / next to the bowl). Labels still come from state, not from the scenario.
SCENARIO_WEIGHTS = {
    "success": 0.30, "hover_no_release": 0.14, "on_rim": 0.08, "next_to_bowl": 0.10, "hover_elsewhere": 0.04,
    "drop_mid_carry": 0.06, "early_release_table": 0.04, "missed_grasp": 0.08, "knock_bowl": 0.04,
    "tipped_bowl": 0.04, "regrasp": 0.05, "no_close": 0.03,
}
if args.scenarios:
    keep = [s.strip() for s in args.scenarios.split(",")]
    SCENARIO_WEIGHTS = {k: v for k, v in SCENARIO_WEIGHTS.items() if k in keep}
    assert SCENARIO_WEIGHTS, f"no known scenario in --scenarios {args.scenarios}"

spec = OBJECTS[args.object]
OUT = Path(args.out)
(OUT / "img").mkdir(parents=True, exist_ok=True)
_BODY = RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                               max_depenetration_velocity=5.0, disable_gravity=False)


# ------------------------------------------------------------------------------------------------------------------
# object spawning
# ------------------------------------------------------------------------------------------------------------------
def usd_local_bounds(usd_path: str):
    """(lo, hi) of the asset's default-prim subtree in its own frame [stage units * metersPerUnit]."""
    from pxr import Usd, UsdGeom

    st = Usd.Stage.Open(usd_path, Usd.Stage.LoadAll)
    if st is None:
        raise RuntimeError(f"cannot open {usd_path}")
    root = st.GetDefaultPrim() or st.GetPseudoRoot()
    mpu = UsdGeom.GetStageMetersPerUnit(st)
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = bc.ComputeWorldBound(root).ComputeAlignedRange()
    if rng.IsEmpty():
        raise RuntimeError(f"empty bounds for {usd_path}")
    return np.array(rng.GetMin()) * mpu, np.array(rng.GetMax()) * mpu


Y_UP_TO_Z_UP_XYZW = (0.70710678, 0.0, 0.0, 0.70710678)  # +90 deg about X: asset +Y -> world +Z


@clone
def spawn_usd_with_physics(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Rigid-body root Xform + the art asset on a child prim rotated Y-up -> Z-up, with mass and convex-hull colliders.

    YCB/Axis_Aligned assets ship without physics schemas and are modelled Y-up (measured 2026-10-06: the tuna can's
    3.35 cm height is its Y extent, metersPerUnit 1). The rotation lives on the child so the rigid-body root frame is
    upright: the env's yaw-only reset event and the expert's extents/narrow-axis logic stay valid.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab.sim import schemas
    from isaaclab.sim.utils import create_prim

    root = create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    spawn_from_usd.__wrapped__(f"{prim_path}/asset", cfg, (0.0, 0.0, 0.0), Y_UP_TO_Z_UP_XYZW, **kwargs)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.05)
    n = 0
    for p in Usd.PrimRange(root):
        if p.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(p)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
            n += 1
    if n == 0:
        raise RuntimeError(f"spawn_usd_with_physics: no editable Mesh prims under {prim_path} (instanced asset?)")
    schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props)
    print(f"SPAWN usd_phys {prim_path}: rigid+mass 0.05 kg, {n} convex-hull colliders, asset rotated Y-up->Z-up",
          flush=True)
    return root


def make_object_cfg(cfg, name: str):
    """Set cfg.scene.object for any OBJECTS entry; returns (scale, analytic (center, extents) or None)."""
    if spec["kind"] == "env":
        set_object(cfg, name)
        return ALL_OBJECTS[name][1], None
    cfg.object_name = name
    init = RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.08, 0.03], rot=[0, 0, 0, 1])
    if spec["kind"] == "usd":
        s = spec["scale"]
        spawn = UsdFileCfg(usd_path=spec["usd"], scale=(s,) * 3, rigid_props=_BODY,
                           mass_props=MassPropertiesCfg(mass=0.05))
        geom = None
    elif spec["kind"] == "usd_phys":
        lo, hi = usd_local_bounds(spec["usd"])
        ext = hi - lo
        ext = np.array([ext[0], ext[2], ext[1]])  # upright frame after the Y-up -> Z-up child rotation
        s = float(min(FIT_NARROW / max(1e-4, min(ext[0], ext[1])), FIT_MAX_XY / max(ext[0], ext[1]),
                      FIT_MAX_Z / max(1e-4, ext[2])))
        print(f"FIT {name}: raw extents_m={np.round(ext, 4)} -> scale {s:.3f} -> {np.round(ext * s, 4)}", flush=True)
        spawn = UsdFileCfg(func=spawn_usd_with_physics, usd_path=spec["usd"], scale=(s,) * 3, rigid_props=_BODY,
                           mass_props=MassPropertiesCfg(mass=0.05))
        geom = None
    else:
        s = 1.0
        mat = sim_utils.PreviewSurfaceCfg(diffuse_color=spec["color"], roughness=0.5)
        common = dict(rigid_props=_BODY, mass_props=MassPropertiesCfg(mass=0.05),
                      collision_props=CollisionPropertiesCfg(), visual_material=mat)
        if spec["shape"] == "cylinder":
            spawn = sim_utils.CylinderCfg(radius=spec["radius"], height=spec["height"], axis="Z", **common)
            ext = (2 * spec["radius"], 2 * spec["radius"], spec["height"])
        elif spec["shape"] == "sphere":
            spawn = sim_utils.SphereCfg(radius=spec["radius"], **common)
            ext = (2 * spec["radius"],) * 3
        else:
            spawn = sim_utils.CuboidCfg(size=spec["size"], **common)
            ext = spec["size"]
        geom = ((0.0, 0.0, 0.0), tuple(ext))
    cfg.scene.object = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Object", init_state=init, spawn=spawn)
    return s, geom


def generic_geometry(env):
    """Per-env (center_local, extents_local) of the object incl. root scale, for meshes AND implicit shapes."""
    from pxr import Usd, UsdGeom

    stage = env.sim.stage
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    centers, extents = [], []
    for i in range(env.num_envs):
        root = stage.GetPrimAtPath(f"/World/envs/env_{i}/Object")
        rng = bc.ComputeUntransformedBound(root).ComputeAlignedRange()
        m = np.array(UsdGeom.Xformable(root).GetLocalTransformation())
        scale = np.linalg.norm(m[:3, :3], axis=1)
        lo, hi = np.array(rng.GetMin()) * scale, np.array(rng.GetMax()) * scale
        centers.append((lo + hi) / 2)
        extents.append(hi - lo)
    return (torch.tensor(np.array(centers), device=env.device, dtype=torch.float32),
            torch.tensor(np.array(extents), device=env.device, dtype=torch.float32))


# ------------------------------------------------------------------------------------------------------------------
# env
# ------------------------------------------------------------------------------------------------------------------
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args.num_envs)
obj_scale, analytic_geom = make_object_cfg(cfg, args.object)
cfg.seed = args.seed
cfg.episode_length_s = 1000.0           # rounds are ended by this script, never by the env's timeout
cfg.terminations.success = None         # labels come from state; the env must not auto-reset on success
cfg.terminations.object_dropping = None  # an object knocked off the table is an outcome to photograph, not a reset
cfg.scene.cloth = AssetBaseCfg(          # thin visual-only table cloth for colour randomization (no collision)
    prim_path="{ENV_REGEX_NS}/Cloth", init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, 0.0, 0.0008)),
    spawn=sim_utils.CuboidCfg(size=(0.46, 0.60, 0.001),
                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), roughness=0.8)),
)
env = gym.make(TASK, cfg=cfg).unwrapped
obs, _ = env.reset(seed=args.seed)
N, dev = env.num_envs, env.device
origins = env.scene.env_origins
O_np = origins.cpu().numpy()
robot, obj, bowl = env.scene["robot"], env.scene["object"], env.scene["bowl"]
if spec["kind"] == "env":
    center_local, extents_local = object_geometry(env)
    BBOX_CEN = center_local.cpu().numpy().copy()  # true bbox centre (rest height), before any grasp override
    if args.object in GRASP_CENTER_OVERRIDE:
        center_local = torch.tensor(GRASP_CENTER_OVERRIDE[args.object], device=dev).expand(N, 3).clone()
        env._geom_object = (center_local, extents_local)
elif analytic_geom is not None:
    center_local = torch.tensor(analytic_geom[0], device=dev, dtype=torch.float32).expand(N, 3).clone()
    extents_local = torch.tensor(analytic_geom[1], device=dev, dtype=torch.float32).expand(N, 3).clone()
    env._geom_object = (center_local, extents_local)
else:
    center_local, extents_local = generic_geometry(env)
    env._geom_object = (center_local, extents_local)
EXT = extents_local.cpu().numpy()
CEN = center_local.cpu().numpy()
if spec["kind"] != "env":
    BBOX_CEN = CEN.copy()
print(f"GEOM {args.object}: center_local={CEN[0].round(4)} extents={EXT[0].round(4)} scale={obj_scale}", flush=True)
ee_idx = robot.find_bodies("gripper")[0][0]
jaw_id = robot.find_joints(["gripper"])[0][0]
stage = env.sim.stage


def state_np():
    """Privileged state of all envs (env frame), numpy."""
    c = (object_center_w(env) - origins).cpu().numpy()
    ee = _t(robot.data.body_pose_w)[:, ee_idx]
    tcp = (ee[:, :3] + quat_apply(ee[:, 3:7], torch.tensor(TCP_LOCAL, device=dev).expand(N, 3)) - origins)
    v = getattr(obj.data, "root_lin_vel_w", None)
    v = _t(v).cpu().numpy() if v is not None else np.zeros((N, 3))
    return dict(
        obj=c, obj_q=_t(obj.data.root_quat_w).cpu().numpy(), obj_v=v,
        obj_root=(_t(obj.data.root_pos_w) - origins).cpu().numpy(),
        bowl=(_t(bowl.data.root_pos_w) - origins).cpu().numpy(), bowl_q=_t(bowl.data.root_quat_w).cpu().numpy(),
        tcp=tcp.cpu().numpy(), jaw=_t(robot.data.joint_pos)[:, jaw_id].cpu().numpy(),
    )


def tilt_deg(q):  # xyzw -> angle between body z and world z
    x, y, z, w = q
    upz = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(upz, -1, 1))))


def pitch_for(xy):
    r = float(np.hypot(xy[0], xy[1]))
    return float(np.clip((r - R_VERTICAL) / (R_FULL_PITCH - R_VERTICAL), 0, 1) * np.deg2rad(PITCH_MAX_DEG))


def approach_dir(yaw, pitch):  # closed form of quat_apply(yaw*rot_y(-pitch), (0,0,-1)) -- see grasp_quat_xyzw
    return np.array([math.sin(pitch) * math.cos(yaw), math.sin(pitch) * math.sin(yaw), -math.cos(pitch)])


def yaw_quat_xyzw(yaw):
    z = torch.zeros_like(yaw)
    return torch.stack([z, z, torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1)


def grasp_quat_xyzw(yaw, pitch, roll):
    h = -pitch / 2
    z = torch.zeros_like(h)
    q = quat_mul(yaw_quat_xyzw(yaw), torch.stack([z, torch.sin(h), z, torch.cos(h)], dim=-1))
    return quat_mul(q, yaw_quat_xyzw(roll))


def grasp_yaw_roll(S, i):
    m = S["obj"][i]
    gy = math.atan2(m[1], m[0])
    x, y, z, w = S["obj_q"][i]
    oyaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    theta_n = oyaw + (math.pi / 2 if EXT[i, 1] < EXT[i, 0] else 0.0)
    d = theta_n - gy
    return gy, math.atan2(math.sin(2 * d), math.cos(2 * d)) / 2


def save_snapshot(i, kind, S, step, path_prefix):
    sc = obs["rgb_camera"]["scene"][i, ..., :3].to(torch.uint8).cpu().numpy()
    wr = obs["rgb_camera"]["wrist"][i, ..., :3].to(torch.uint8).cpu().numpy()
    img = np.concatenate([sc, wr], axis=1)
    p = OUT / "img" / f"{path_prefix}_{kind}.png"
    Image.fromarray(img).resize((512, 256)).save(p)
    st = {k: np.round(np.asarray(v[i], dtype=np.float64), 5).tolist() for k, v in S.items()}
    return {"kind": kind, "step": int(step), "image": str(p), "state": st, "grip_cmd": float(grip[i]),
            "seg": cur_seg_name(i), "mean_px": float(img.mean())}


# ------------------------------------------------------------------------------------------------------------------
# scenario scripts: lists of segments. move segments: dict(k=<target kind>, grip, snap, precise, z, ...).
# Targets: pre/grasp (live object), up (tcp xy fixed at segment start), bowl (live bowl xy + offset),
# point (absolute xy fixed at plan time or at segment start via 'frac'). hold: n steps. replan: recompute grasp.
# grip: 'open' (pre-shape), 'closed', 'keep'.
# ------------------------------------------------------------------------------------------------------------------
def S_pick(off=(0.0, 0.0)):
    return [dict(k="setgrasp", off=off), dict(k="pre", grip="open"), dict(k="grasp", grip="open", precise=True),
            dict(k="hold", n=HOLD_CLOSE, grip="closed", after_close=True)]


def S_release(snap="release_instant"):
    return [dict(k="hold", n=HOLD_RELEASE, grip="open", snap=snap)]


def S_finish():
    return [dict(k="up", z=Z_RETREAT, grip="keep"), dict(k="hold", n=FINAL_SETTLE, grip="keep", snap="final")]


def reach_ok(xy):
    r = math.hypot(xy[0], xy[1])
    return 0.12 <= r <= 0.27 and abs(xy[1]) <= 0.16 and xy[0] >= 0.08


def random_table_point(rng, bowl_xy, min_d, max_d=None, around=None):
    for _ in range(200):
        if around is None:
            p = np.array([rng.uniform(0.12, 0.27), rng.uniform(-0.15, 0.15)])
        else:
            a = rng.uniform(0, 2 * math.pi)
            p = np.asarray(around) + rng.uniform(min_d, max_d) * np.array([math.cos(a), math.sin(a)])
        if reach_ok(p) and (around is not None or np.linalg.norm(p - bowl_xy) >= min_d):
            return p
    return None


def build_plan(scn, rng, S, i):
    zc = float(rng.uniform(0.09, 0.12))         # carry height
    zr = float(rng.uniform(0.070, 0.085))       # release height over the bowl
    bxy = S["bowl"][i][:2]
    oxy = S["obj"][i][:2]
    zplace = float(S["obj"][i][2] + GRASP_DZ + 0.012)  # TCP height that sets the object back on the table
    params = dict(zc=round(zc, 4), zr=round(zr, 4))
    lift = dict(k="up", z=zc, grip="keep", snap="after_lift")
    if scn in ("success", "tipped_bowl"):
        plan = S_pick() + [lift, dict(k="bowl", z=zc, grip="keep", snap="after_move"),
                           dict(k="bowl", z=zr, grip="keep")] + S_release() + S_finish()
    elif scn == "on_rim":  # the full success motion, but released over the rim (object lands on/against the rim)
        a = rng.uniform(0, 2 * math.pi)
        d = float(rng.uniform(0.045, 0.075))
        off = d * np.array([math.cos(a), math.sin(a)])
        params.update(rim_offset=np.round(off, 4).tolist())
        plan = S_pick() + [lift, dict(k="bowl", z=zc, off=off, grip="keep", snap="after_move"),
                           dict(k="bowl", z=zr, off=off, grip="keep")] + S_release() + S_finish()
    elif scn == "hover_no_release":
        low = bool(rng.random() < 0.4)
        zh = 0.055 if low else float(rng.uniform(0.075, 0.10))
        params.update(low_hold=low, zh=round(zh, 4))
        plan = S_pick() + [lift, dict(k="bowl", z=zc, grip="keep", snap="after_move"),
                           dict(k="bowl", z=zh, grip="keep"), dict(k="hold", n=15, grip="keep")]
        plan += [dict(k="hold", n=FINAL_SETTLE, grip="keep", snap="final")] if low else S_finish()
    elif scn == "hover_elsewhere":
        p = random_table_point(rng, bxy, 0.13)
        if p is None:
            return build_plan("success", rng, S, i)
        drop = bool(rng.random() < 0.5)
        params.update(point=np.round(p, 4).tolist(), released=drop)
        plan = S_pick() + [lift, dict(k="point", xy=p, z=zc, grip="keep", snap="after_move"),
                           dict(k="hold", n=10, grip="keep")]
        if drop:
            plan += [dict(k="point", xy=p, z=zplace, grip="keep")] + S_release()
        plan += S_finish()
    elif scn == "drop_mid_carry":
        f = float(rng.uniform(0.25, 0.7))
        params.update(frac=round(f, 3))
        plan = S_pick() + [lift, dict(k="point", frac=f, z=zc, grip="keep", snap="after_move")] + S_release() \
            + S_finish()
    elif scn == "early_release_table":
        p = random_table_point(rng, bxy, 0.03, 0.07, around=oxy)
        if p is None or np.linalg.norm(p - bxy) < 0.10:
            p = oxy
        params.update(point=np.round(p, 4).tolist())
        plan = S_pick() + [dict(k="up", z=0.06, grip="keep", snap="after_lift"),
                           dict(k="point", xy=p, z=zplace, grip="keep")] + S_release() + S_finish()
    elif scn == "next_to_bowl":
        p = None
        for _ in range(50):
            a = rng.uniform(0, 2 * math.pi)
            q = bxy + rng.uniform(0.09, 0.14) * np.array([math.cos(a), math.sin(a)])
            if reach_ok(q) and np.linalg.norm(q - oxy) > 0.02:
                p = q
                break
        if p is None:
            return build_plan("drop_mid_carry", rng, S, i)
        params.update(point=np.round(p, 4).tolist())
        plan = S_pick() + [lift, dict(k="point", xy=p, z=zc, grip="keep", snap="after_move"),
                           dict(k="point", xy=p, z=zplace, grip="keep")] + S_release() + S_finish()
    elif scn == "missed_grasp":
        a = rng.uniform(0, 2 * math.pi)
        d = float(rng.uniform(0.028, 0.045))
        off = (d * math.cos(a), d * math.sin(a))
        params.update(grasp_offset=np.round(off, 4).tolist())
        plan = S_pick(off) + [lift, dict(k="bowl", z=zc, grip="keep", snap="after_move"),
                              dict(k="bowl", z=zr, grip="keep")] + S_release() + S_finish()
    elif scn == "knock_bowl":
        a = rng.uniform(0, 2 * math.pi)
        u = np.array([math.cos(a), math.sin(a)])
        params.update(dir=np.round(u, 3).tolist())
        plan = S_pick() + [lift, dict(k="bowl", z=zc, off=0.075 * u, grip="keep", snap="after_move"),
                           dict(k="bowl", z=0.035, off=0.075 * u, grip="keep"),
                           dict(k="bowl", z=0.035, off=-0.03 * u, grip="keep")] + S_release() + S_finish()
    elif scn == "regrasp":
        plan = S_pick() + [dict(k="up", z=0.06, grip="keep", snap="after_lift"), dict(k="hold", n=HOLD_RELEASE,
                                                                                       grip="open"),
                           dict(k="up", z=0.10, grip="open"), dict(k="hold", n=15, grip="open")] \
            + S_pick() + [dict(k="up", z=zc, grip="keep"), dict(k="bowl", z=zc, grip="keep", snap="after_move"),
                          dict(k="bowl", z=zr, grip="keep")] + S_release() + S_finish()
    elif scn == "no_close":
        plan = [dict(k="setgrasp", off=(0.0, 0.0)), dict(k="pre", grip="open"),
                dict(k="grasp", grip="open", precise=True), dict(k="hold", n=15, grip="open")] + S_finish()
    else:
        raise ValueError(scn)
    return plan, params


# ------------------------------------------------------------------------------------------------------------------
# domain randomization
# ------------------------------------------------------------------------------------------------------------------
def find_shader(root_path):
    from pxr import Usd, UsdShade

    root = stage.GetPrimAtPath(root_path)
    for p in Usd.PrimRange(root):
        if p.IsA(UsdShade.Shader):
            return UsdShade.Shader(p)
    return None


CLOTH_SHADERS = [find_shader(f"/World/envs/env_{i}/Cloth") for i in range(N)]
print(f"DR cloth shaders found: {sum(s is not None for s in CLOTH_SHADERS)}/{N}", flush=True)
CLOTH_PALETTE = [(0.92, 0.92, 0.90), (0.15, 0.15, 0.17), (0.55, 0.38, 0.22), (0.30, 0.45, 0.30), (0.25, 0.32, 0.55),
                 (0.60, 0.60, 0.62), (0.75, 0.68, 0.55), (0.45, 0.20, 0.20)]


def randomize(rng, round_idx):
    from pxr import Gf, UsdGeom, UsdLux

    rec = {"env": [None] * N}
    if args.no_dr:
        return rec
    light = stage.GetPrimAtPath("/World/light")
    if light.IsValid():
        inten = float(rng.uniform(1200, 4800))
        tint = rng.uniform(0.85, 1.15, 3)
        col = np.clip(0.75 * tint / tint.mean(), 0, 1)
        dl = UsdLux.DomeLight(light)
        dl.GetIntensityAttr().Set(inten)
        dl.GetColorAttr().Set(Gf.Vec3f(*[float(c) for c in col]))
        rec["light"] = dict(intensity=round(inten, 1), color=np.round(col, 3).tolist())
    else:
        rec["light"] = "MISSING /World/light"
        print("DR WARNING: /World/light not found, lighting not randomized", flush=True)
    eyes = np.array(SCENE_CAM_EYE)[None] + rng.normal(0, [0.02, 0.02, 0.015], (N, 3))
    tgts = np.array(SCENE_CAM_TARGET)[None] + rng.normal(0, 0.01, (N, 3))
    env.scene["scene_cam"].set_world_poses_from_view(
        torch.tensor(eyes, device=dev, dtype=torch.float32) + origins,
        torch.tensor(tgts, device=dev, dtype=torch.float32) + origins)
    for i in range(N):
        visible = bool(rng.random() < 0.75)
        prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Cloth")
        col = None
        if prim.IsValid():
            img = UsdGeom.Imageable(prim)
            img.MakeVisible() if visible else img.MakeInvisible()
            if visible and CLOTH_SHADERS[i] is not None:
                base = np.array(CLOTH_PALETTE[rng.integers(len(CLOTH_PALETTE))])
                col = np.clip(base + rng.normal(0, 0.05, 3), 0, 1)
                inp = CLOTH_SHADERS[i].GetInput("diffuseColor")
                if inp:
                    inp.Set(Gf.Vec3f(*[float(c) for c in col]))
        rec["env"][i] = dict(cloth_visible=visible, cloth_color=None if col is None else np.round(col, 3).tolist(),
                             cam_eye=np.round(eyes[i], 4).tolist(), cam_target=np.round(tgts[i], 4).tolist())
    return rec


def place_objects(rng, tipped_ids):
    """Rest the object on the table (root z from its measured geometry) and optionally tip the bowl over."""
    pos = _t(obj.data.root_pos_w).clone()
    quat = _t(obj.data.root_quat_w).clone()
    lo_z = float(BBOX_CEN[0, 2] - EXT[0, 2] / 2)
    pos[:, 2] = origins[:, 2] + (-lo_z + 0.003)
    ids = torch.arange(N, device=dev)
    obj.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], -1), env_ids=ids)
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(N, 6, device=dev), env_ids=ids)
    if tipped_ids:
        t = torch.tensor(tipped_ids, device=dev)
        bp = _t(bowl.data.root_pos_w)[t].clone()
        bp[:, 2] = origins[t, 2] + 0.075
        qs = []
        for _ in tipped_ids:
            ang = float(rng.choice([rng.uniform(95, 125), rng.uniform(165, 180)])) * math.pi / 180
            az = rng.uniform(0, 2 * math.pi)
            axis = np.array([math.cos(az), math.sin(az), 0.0])
            qs.append([*(axis * math.sin(ang / 2)), math.cos(ang / 2)])
        bq = torch.tensor(qs, device=dev, dtype=torch.float32)
        bowl.write_root_pose_to_sim_index(root_pose=torch.cat([bp, bq], -1), env_ids=t)
        bowl.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(tipped_ids), 6, device=dev), env_ids=t)


# ------------------------------------------------------------------------------------------------------------------
# main loop
# ------------------------------------------------------------------------------------------------------------------
plan = [[] for _ in range(N)]
seg_i = np.zeros(N, int)
seg_t = np.zeros(N, int)
seg_fix = [None] * N            # fixed target for the current segment (when not live)
cmd_tcp = np.zeros((N, 3))
grip = np.full(N, GRIP_PRESHAPE)
gyaw = np.zeros(N)
groll = np.zeros(N)
goff = np.zeros((N, 2))
after_close = np.zeros(N, bool)
done = np.zeros(N, bool)
stalls = [[] for _ in range(N)]


def cur_seg_name(i):
    if done[i]:
        return "done"
    if seg_i[i] >= len(plan[i]):
        return "end"
    s = plan[i][seg_i[i]]
    return s["k"] + (f":{s['snap']}" if s.get("snap") else "")


def seg_target(i, s, S):
    k = s["k"]
    if k in ("pre", "grasp"):
        m = S["obj"][i]
        g = np.array([m[0] + goff[i, 0], m[1] + goff[i, 1], max(m[2] + GRASP_DZ, TCP_MIN_Z)])
        if k == "grasp":
            return g
        return g - approach_dir(gyaw[i], pitch_for(g[:2])) * APPROACH_BACKOFF
    if k == "bowl":
        off = s.get("off", (0.0, 0.0))
        b = S["bowl"][i]
        return np.array([b[0] + off[0], b[1] + off[1], s["z"]])
    return seg_fix[i]  # up / point / hold


def start_segment(i, S):
    """Enter plan[i][seg_i[i]]: run instant segments, fix targets. Returns False when the plan is finished."""
    while seg_i[i] < len(plan[i]):
        s = plan[i][seg_i[i]]
        seg_t[i] = 0
        if s["k"] == "setgrasp":
            goff[i] = s["off"]
            gyaw[i], groll[i] = grasp_yaw_roll(S, i)
            after_close[i] = False
            seg_i[i] += 1
            continue
        if s["k"] == "up":
            seg_fix[i] = np.array([S["tcp"][i][0], S["tcp"][i][1], s["z"]])
        elif s["k"] == "point":
            if "frac" in s:
                p0, b = S["tcp"][i][:2], S["bowl"][i][:2]
                xy = p0 + s["frac"] * (b - p0)
            else:
                xy = s["xy"]
            seg_fix[i] = np.array([xy[0], xy[1], s["z"]])
        elif s["k"] == "hold":
            seg_fix[i] = cmd_tcp[i].copy()
        if s.get("grip") == "open":
            grip[i] = GRIP_PRESHAPE
        elif s.get("grip") == "closed":
            grip[i] = 1.0
        if s.get("after_close"):
            after_close[i] = True
        return True
    return False


meta_runs = []
t_start = time.time()
episodes_written = 0
names = list(SCENARIO_WEIGHTS)
probs = np.array([SCENARIO_WEIGHTS[k] for k in names])
probs = probs / probs.sum()
max_step = args.max_speed * env.step_dt
jsonl = open(OUT / "episodes.jsonl", "a")
for rnd in range(args.rounds):
    rng = np.random.default_rng(args.seed * 1000 + rnd)
    if rnd > 0:
        env._reset_idx(torch.arange(N, device=dev))
        obs = env.observation_manager.compute()
    scen = [str(rng.choice(names, p=probs)) for _ in range(N)]
    tipped = [i for i in range(N) if scen[i] == "tipped_bowl"]
    place_objects(rng, tipped)
    dr = randomize(rng, rnd)
    S = state_np()
    cmd_tcp[:] = S["tcp"]
    grip[:] = GRIP_PRESHAPE
    done[:] = False
    seg_i[:] = 0
    goff[:] = 0
    after_close[:] = False
    for i in range(N):
        gyaw[i], groll[i] = grasp_yaw_roll(S, i)
        stalls[i] = []
    episodes = [dict(round=rnd, env=i, object=args.object, scenario=scen[i], snaps=[], dr=dr["env"][i],
                     light=dr.get("light")) for i in range(N)]
    mid_step = rng.integers(START_SETTLE + 30, START_SETTLE + 400, N)
    rest = None
    max_rise = np.zeros(N)
    t_round = time.time()
    step = 0
    while step < args.max_round_steps + START_SETTLE and app.is_running():
        S = state_np()
        if step == START_SETTLE:
            rest = dict(obj=S["obj"].copy(), bowl=S["bowl"].copy(), bowl_q=S["bowl_q"].copy())
            for i in range(N):
                episodes[i]["rest"] = dict(obj=np.round(S["obj"][i], 5).tolist(), bowl=np.round(S["bowl"][i], 5).tolist(),
                                           bowl_tilt_deg=round(tilt_deg(S["bowl_q"][i]), 2))
                episodes[i]["snaps"].append(save_snapshot(i, "before", S, step, f"r{rnd}_e{i:02d}"))
                gyaw[i], groll[i] = grasp_yaw_roll(S, i)
                p, prm = build_plan(scen[i], rng, S, i)
                plan[i], episodes[i]["params"] = p, prm
                seg_i[i] = 0
                start_segment(i, S)
        if step >= START_SETTLE:
            max_rise = np.maximum(max_rise, S["obj"][:, 2] - rest["obj"][:, 2])
            for i in range(N):
                if done[i]:
                    continue
                if step == mid_step[i]:
                    episodes[i]["snaps"].append(save_snapshot(i, "mid", S, step, f"r{rnd}_e{i:02d}"))
                s = plan[i][seg_i[i]]
                if s["k"] == "hold":
                    adv = seg_t[i] >= s["n"]
                else:
                    err = float(np.linalg.norm(S["tcp"][i] - seg_target(i, s, S)))
                    adv = err < POS_TOL or (not s.get("precise") and seg_t[i] > SETTLE_STEPS and err < GOOD_ENOUGH)
                    if seg_t[i] > SEG_BUDGET:
                        adv = True
                        stalls[i].append(dict(seg=seg_i[i], k=s["k"], err=round(err, 4), step=step))
                if adv:
                    if s.get("snap"):
                        episodes[i]["snaps"].append(save_snapshot(i, s["snap"], S, step, f"r{rnd}_e{i:02d}"))
                    seg_i[i] += 1
                    if not start_segment(i, S):
                        done[i] = True
                        episodes[i]["end_step"] = step
                        continue
                seg_t[i] += 1
            if done.all():
                break
        # ---- command: speed-limited TCP toward each env's target ----
        tgt = cmd_tcp.copy()
        if step >= START_SETTLE:
            for i in range(N):
                if not done[i]:
                    tgt[i] = seg_target(i, plan[i][seg_i[i]], S)
        delta = tgt - cmd_tcp
        dist = np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
        cmd_tcp = cmd_tcp + delta * np.minimum(max_step / dist, 1.0)
        yaw = np.where(after_close, np.arctan2(cmd_tcp[:, 1], cmd_tcp[:, 0]), gyaw)
        pitch = np.array([pitch_for(c[:2]) for c in cmd_tcp])
        yaw_t = torch.tensor(yaw, device=dev, dtype=torch.float32)
        q_w = grasp_quat_xyzw(yaw_t, torch.tensor(pitch, device=dev, dtype=torch.float32),
                              torch.tensor(groll, device=dev, dtype=torch.float32))
        cmd_t = torch.tensor(cmd_tcp, device=dev, dtype=torch.float32)
        origin_w = cmd_t + origins - quat_apply(q_w, torch.tensor(TCP_LOCAL, device=dev).expand(N, 3))
        root = _t(robot.data.root_pose_w)
        pos_b, quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], origin_w, q_w)
        action = torch.cat([pos_b, quat_b, torch.tensor(grip, device=dev, dtype=torch.float32)[:, None]], dim=-1)
        obs, _, term, trunc, _ = env.step(action)
        if bool((_t(term) | _t(trunc)).any()):
            print(f"WARNING env-triggered reset at step {step} envs {(_t(term) | _t(trunc)).nonzero().flatten().tolist()}",
                  flush=True)
        step += 1
    # ---- round end: truncated envs get their final snapshot as-is ----
    S = state_np()
    n_in = 0
    for i in range(N):
        e = episodes[i]
        e["truncated"] = not bool(done[i])
        if not done[i]:
            e["snaps"].append(save_snapshot(i, "final", S, step, f"r{rnd}_e{i:02d}"))
            e["end_step"] = step
        e["stalls"] = stalls[i]
        e["max_rise_m"] = round(float(max_rise[i]), 4)
        e["object_scale"] = obj_scale
        e["extents_local"] = np.round(EXT[i], 4).tolist()
        e["center_local"] = np.round(CEN[i], 4).tolist()
        f = e["snaps"][-1]["state"]
        hd = math.hypot(f["obj"][0] - f["bowl"][0], f["obj"][1] - f["bowl"][1])
        n_in += int(hd < BOWL_RADIUS and f["obj"][2] - f["bowl"][2] < 0.07 and f["jaw"] > 0.5)
        jsonl.write(json.dumps(e) + "\n")
        episodes_written += 1
    jsonl.flush()
    print(f"ROUND {rnd} steps={step} wall={time.time() - t_round:.0f}s done={int(done.sum())}/{N} "
          f"prelim_in_bowl_final={n_in}/{N} scenarios={ {k: scen.count(k) for k in set(scen)} } "
          f"light={dr.get('light')}", flush=True)

meta = dict(object=args.object, spec={k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()},
            heldout=args.object in HELDOUT, scale=obj_scale, num_envs=N, rounds=args.rounds, seed=args.seed,
            episodes=episodes_written, wall_s=round(time.time() - t_start, 1), scenario_weights=SCENARIO_WEIGHTS,
            dr=not args.no_dr, task=TASK, fps=round(1 / env.step_dt, 3), final_settle_steps=FINAL_SETTLE,
            bowl_radius=BOWL_RADIUS, max_round_steps=args.max_round_steps)
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print("SUMMARY", json.dumps(meta), flush=True)
jsonl.close()
env.close()
app.close()
