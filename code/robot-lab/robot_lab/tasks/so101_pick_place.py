"""SO-101: put <object> in the yellow bowl — multi-object, with a held-out object split.

Generalizes so101_mug_bowl.py:
  * one `object` slot filled per env from an object set (Isaac Lab clones prototypes sequentially: env i gets
    object i % len(set)), so a batch of envs covers the whole set evenly
  * object geometry (body center, extents) is MEASURED from the live stage at runtime, per env — no hand-tuned
    offsets like MUG_BODY_CENTER_LOCAL (the lesson from the mug: asset origins are arbitrary)
  * TRAIN and HELDOUT object sets are separate gym ids; held-out objects are never recorded for training
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.contrib.stack import mdp
from isaaclab.envs import mdp as core_mdp

from .so101_mug_bowl import BOWL_RADIUS, SO101MugBowlEnvCfg, _t

YCB = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics"
# (name used in the caption, usd, uniform scale). Scales bring each object within the SO-101 jaw (~5 cm).
TRAIN_OBJECTS = [
    ("mug", f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd", 0.5),
    ("blue block", f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd", 0.8),
    ("soup can", f"{YCB}/005_tomato_soup_can.usd", 0.5),
    ("sugar box", f"{YCB}/004_sugar_box.usd", 0.4),
]
HELDOUT_OBJECTS = [
    ("mustard bottle", f"{YCB}/006_mustard_bottle.usd", 0.45),
    ("cracker box", f"{YCB}/003_cracker_box.usd", 0.3),
]
CAPTION = "put the {} in the yellow bowl"
ALL_OBJECTS = {name: (usd, scale) for name, usd, scale in TRAIN_OBJECTS + HELDOUT_OBJECTS}
_BODY = RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                               max_depenetration_velocity=5.0, disable_gravity=False)


def set_object(cfg, name: str):
    """Put one object type in every env (homogeneous cloning). Call after parse_env_cfg, before gym.make.

    One object type per run: Isaac Lab 3.0 EA's heterogeneous MultiAssetSpawner cloning created fewer object
    instances than envs (3 for 4) on 2026-09-23, so runs loop over objects instead.
    """
    usd, scale = ALL_OBJECTS[name]
    cfg.object_name = name
    cfg.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.08, 0.03], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(usd_path=usd, scale=(scale,) * 3, rigid_props=_BODY, mass_props=MassPropertiesCfg(mass=0.05)),
    )
    return cfg


def object_names(env) -> list[str]:
    """Caption name of the object in each env (one object type per run)."""
    return [env.cfg.object_name] * env.num_envs


def object_geometry(env, name: str = "object"):
    """Per-env (center_local (N,3), extents_local (N,3)) of the object's meshes in its root frame, root scale applied.

    Static geometry read from USD relative to the root (never world transforms: physics lives in Fabric).
    Cached on the env.
    """
    cache_key = f"_geom_{name}"
    if hasattr(env, cache_key):
        return getattr(env, cache_key)
    from pxr import Usd, UsdGeom

    stage = env.sim.stage
    centers, extents = [], []
    xf = UsdGeom.XformCache()
    for i in range(env.num_envs):
        root = stage.GetPrimAtPath(f"/World/envs/env_{i}/{name.capitalize()}")
        pts = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                p = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                if p:
                    rel, _ = xf.ComputeRelativeTransform(prim, root)
                    pts.append((np.c_[np.array(p), np.ones(len(p))] @ np.array(rel))[:, :3])
        P = np.concatenate(pts)
        m = np.array(UsdGeom.Xformable(root).GetLocalTransformation())
        scale = np.linalg.norm(m[:3, :3], axis=1)  # root scale (the relative transform excludes it)
        lo, hi = P.min(0) * scale, P.max(0) * scale
        centers.append((lo + hi) / 2)
        extents.append(hi - lo)
    out = (torch.tensor(np.array(centers), device=env.device, dtype=torch.float32),
           torch.tensor(np.array(extents), device=env.device, dtype=torch.float32))
    setattr(env, cache_key, out)
    return out


def object_center_w(env, name: str = "object") -> torch.Tensor:
    """World position of the object's geometric center (not the asset root)."""
    from isaaclab.utils.math import quat_apply

    data = env.scene[name].data
    center_local, _ = object_geometry(env, name)
    return _t(data.root_pos_w) + quat_apply(_t(data.root_quat_w), center_local)


def object_in_bowl(env, radius: float = BOWL_RADIUS, max_height: float = 0.07) -> torch.Tensor:
    obj = object_center_w(env)
    bowl = _t(env.scene["bowl"].data.root_pos_w)
    inside = torch.linalg.norm(obj[:, :2] - bowl[:, :2], dim=-1) < radius
    low = (obj[:, 2] - bowl[:, 2]) < max_height
    robot = env.scene["robot"]
    grip_idx = robot.find_joints(env.cfg.gripper_joint_names)[0]
    opened = _t(robot.data.joint_pos)[:, grip_idx].mean(-1) > 0.5  # [rad] jaw released (~29 deg); expert opens to ~45 deg
    return inside & low & opened


def object_pose_env(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return torch.cat([_t(data.root_pos_w) - env.scene.env_origins, _t(data.root_quat_w)], dim=-1)


@configclass
class PickPlaceObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        object_pose = ObsTerm(func=object_pose_env, params={"asset_cfg": SceneEntityCfg("object")})
        bowl_pose = ObsTerm(func=object_pose_env, params={"asset_cfg": SceneEntityCfg("bowl")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CameraObsCfg(ObsGroup):
        scene = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("scene_cam"), "data_type": "rgb",
                                                "normalize": False})
        wrist = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb",
                                                "normalize": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    rgb_camera: CameraObsCfg = CameraObsCfg()


@configclass
class PickPlaceTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(func=mdp.root_height_below_minimum,
                               params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")})
    success = DoneTerm(func=object_in_bowl)


@configclass
class SO101PickPlaceEnvCfg(SO101MugBowlEnvCfg):
    object_set = TRAIN_OBJECTS
    object_name: str = "mug"

    def __post_init__(self):
        super().__post_init__()
        self.scene.mug = None
        set_object(self, self.object_set[0][0])  # default; tools call set_object(cfg, name) to choose
        self.observations = PickPlaceObservationsCfg()
        self.terminations = PickPlaceTerminationsCfg()
        self.events.randomize_objects.params["asset_cfgs"] = [SceneEntityCfg("object"), SceneEntityCfg("bowl")]
        self.events.mug_friction.params["asset_cfg"] = SceneEntityCfg("object")


@configclass
class SO101PickPlaceHeldoutEnvCfg(SO101PickPlaceEnvCfg):
    object_set = HELDOUT_OBJECTS


@configclass
class SO101PickPlaceJointEnvCfg(SO101PickPlaceEnvCfg):
    """Evaluation variant: the policy commands ABSOLUTE joint targets [rad] (5 arm + jaw), as FLUX SO-101 outputs.

    Replaces the IK-Abs pose action (used by the scripted expert) with raw joint-position targets, scale 1, no offset.
    """

    def __post_init__(self):
        super().__post_init__()
        from isaaclab_tasks.contrib.stack.config.so101.stack_ik_abs_env_cfg import SO101IkActionsCfg

        self.actions = SO101IkActionsCfg(
            arm_action=mdp.JointPositionActionCfg(
                asset_name="robot", joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                scale=1.0, use_default_offset=False),
            gripper_action=mdp.JointPositionActionCfg(asset_name="robot", joint_names=["gripper"], scale=1.0,
                                                      use_default_offset=False),
        )
        self.isaac_teleop = None
