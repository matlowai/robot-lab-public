"""SO-101: put the mug in the yellow bowl.

Built on Isaac Lab 3.0's SO-101 IK-Abs stack task (seated arm, soft gripper, full-pose IK) with:
  * the three cubes removed; a mug (the "new object") and the yellow sorting bowl added, both randomized on reset
  * 30 Hz control (sim.dt = 1/120 s, decimation 4) to match the FLUX 3 Action SO-101 policy
  * a fixed scene camera and a wrist camera, 256x256 RGB, like the policy's `scene` / `wrist` inputs
  * success = mug inside the bowl footprint, resting low in it, gripper open

Action (inherited, IK-Abs): [pos_xyz (robot base frame, m), quat_xyzw, gripper closedness c in [0, 1]].
The recorder stores the resulting *joint* targets, which is what the FLUX SO-101 policy predicts.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import mdp as core_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.config.so101 import stack_ik_abs_env_cfg
from isaaclab_tasks.contrib.stack.mdp import stack_events

# ---- task constants (tune in-sim with tools/check_scene.py) -----------------------------------
MUG_SCALE = 0.5  # the stock mug is ~10 cm; halve it so the SO-101 jaw can close around it [I, verify]
MUG_USD = f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd"
BOWL_USD = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bowl_yellow.usd"  # 12 x 12 x 5 cm
BOWL_RADIUS = 0.055  # [m] inner footprint radius used by the success check
# The mug's root origin is NOT the cup center: the cup body is centered at local (0, +0.0446, 0) in asset units
# (handle along +x). Measured with tools/probe_mug_geom.py (collision mesh, 2026-09-23). Scaled with MUG_SCALE.
MUG_BODY_CENTER_LOCAL = (0.0, 0.0446 * MUG_SCALE, 0.0)


def mug_body_center_w(env, mug_name: str = "mug") -> torch.Tensor:
    """World position of the cup body center (not the asset root)."""
    from isaaclab.utils.math import quat_apply

    data = env.scene[mug_name].data
    # Isaac Lab 3.0 data fields are warp-backed ProxyArrays; TorchScript math needs real tensors (.torch).
    pos, quat = _t(data.root_pos_w), _t(data.root_quat_w)
    off = torch.tensor(MUG_BODY_CENTER_LOCAL, device=env.device).expand(pos.shape[0], 3)
    return pos + quat_apply(quat, off)


def _t(x):
    return x.torch if hasattr(x, "torch") else x
CONTROL_HZ = 30.0
CAMERA_HW = (256, 256)
# Workspace in world frame [m]; the SO-101 is seated at the origin facing +x with ~0.3 m reach.
OBJECT_POSE_RANGE = {"x": (0.15, 0.25), "y": (-0.10, 0.10), "z": (0.03, 0.03), "yaw": (-3.14, 3.14)}  # reach-limited (pilot6: all stalls at r >= 0.24)
# z: the mug origin is at its center (half-height ~2.6 cm); both objects drop ~3 cm onto the table at reset
SCENE_CAM_EYE = (0.46, -0.20, 0.30)  # set at runtime via set_world_poses_from_view (per env origin)
SCENE_CAM_TARGET = (0.20, 0.0, 0.03)


def mug_in_bowl(
    env,
    mug_cfg: SceneEntityCfg = SceneEntityCfg("mug"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    radius: float = BOWL_RADIUS,
    max_height: float = 0.06,
) -> torch.Tensor:
    """Mug center within the bowl footprint and resting low (not held above it), gripper open."""
    mug = mug_body_center_w(env, mug_cfg.name)
    bowl = env.scene[bowl_cfg.name].data.root_pos_w
    inside = torch.linalg.norm(mug[:, :2] - bowl[:, :2], dim=-1) < radius
    low = (mug[:, 2] - bowl[:, 2]) < max_height
    robot = env.scene["robot"]
    grip_idx = robot.find_joints(env.cfg.gripper_joint_names)[0]
    opened = robot.data.joint_pos[:, grip_idx].mean(-1) > (env.cfg.gripper_open_val - 0.5)
    return inside & low & opened


def object_pose_w(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Object root pose [pos_xyz, quat] in world frame, relative to the env origin."""
    data = env.scene[asset_cfg.name].data
    return torch.cat([data.root_pos_w - env.scene.env_origins, data.root_quat_w], dim=-1)


@configclass
class MugBowlObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioception + privileged object state (the recorder/scripted expert uses these)."""

        joint_pos = ObsTerm(func=mdp.joint_pos)  # absolute joint positions [rad]
        joint_vel = ObsTerm(func=mdp.joint_vel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        mug_pose = ObsTerm(func=object_pose_w, params={"asset_cfg": SceneEntityCfg("mug")})
        bowl_pose = ObsTerm(func=object_pose_w, params={"asset_cfg": SceneEntityCfg("bowl")})

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
class MugBowlTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    mug_dropping = DoneTerm(func=mdp.root_height_below_minimum,
                            params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("mug")})
    success = DoneTerm(func=mug_in_bowl)


@configclass
class MugBowlEventsCfg:
    # Grip friction: the stock 0.5/0.5 lets the 16.5 g mug slip out of the jaw (verified 2026-09-23).
    mug_friction = EventTerm(
        func=core_mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("mug"), "static_friction_range": (1.0, 1.0),
                "dynamic_friction_range": (0.9, 0.9), "restitution_range": (0.0, 0.0), "num_buckets": 1},
    )
    robot_friction = EventTerm(
        func=core_mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "static_friction_range": (1.0, 1.0),
                "dynamic_friction_range": (0.9, 0.9), "restitution_range": (0.0, 0.0), "num_buckets": 1},
    )
    randomize_joint_state = EventTerm(
        func=stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )
    randomize_objects = EventTerm(
        func=stack_events.randomize_object_pose,
        mode="reset",
        params={"pose_range": OBJECT_POSE_RANGE, "min_separation": 0.13,
                "asset_cfgs": [SceneEntityCfg("mug"), SceneEntityCfg("bowl")]},
    )


@configclass
class SO101MugBowlEnvCfg(stack_ik_abs_env_cfg.SO101CubeStackEnvCfg):
    """SO-101 IK-Abs control, mug -> yellow bowl, 30 Hz, scene + wrist cameras."""

    def __post_init__(self):
        super().__post_init__()

        # ---- timing: 30 Hz control ----
        self.sim.dt = 1.0 / 120.0
        self.decimation = int(round(1.0 / (CONTROL_HZ * self.sim.dt)))  # 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0

        # ---- scene: drop cubes, add mug + bowl ----
        self.scene.cube_1 = None
        self.scene.cube_2 = None
        self.scene.cube_3 = None
        body = RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                                      max_depenetration_velocity=5.0, disable_gravity=False)
        self.scene.mug = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Mug",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.18, 0.08, 0.0], rot=[0, 0, 0, 1]),
            spawn=UsdFileCfg(usd_path=MUG_USD, scale=(MUG_SCALE,) * 3, rigid_props=body,
                             mass_props=MassPropertiesCfg(mass=0.05), semantic_tags=[("class", "mug")]),
        )
        self.scene.bowl = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bowl",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.24, -0.08, 0.0], rot=[0, 0, 0, 1]),
            spawn=UsdFileCfg(usd_path=BOWL_USD, rigid_props=body, mass_props=MassPropertiesCfg(mass=1.0),
                             semantic_tags=[("class", "bowl")]),
        )

        # ---- cameras (256x256 RGB) ----
        pinhole = sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955,
                                             clipping_range=(0.01, 3.0))
        self.scene.scene_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/scene_cam", update_period=0.0, height=CAMERA_HW[0], width=CAMERA_HW[1],
            data_types=["rgb"], spawn=pinhole,
            offset=CameraCfg.OffsetCfg(pos=SCENE_CAM_EYE, rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        )
        # Wrist camera at the real SO-101 camera_mount (tools/probe_camera_mount.py, 2026-09-23): lens at gripper-frame
        # (-0.002, 0.056, -0.053) m (camera placed 1.8 cm forward along the axis to clear the lens cover), optical axis = PCB normal (0, -0.574, -0.819) -> looks down/in at the jaws, like the
        # physical SO-101 wrist cam. ~92 deg HFOV (focal 10 mm on a 20.955 mm aperture), typical of those modules.
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_cam", update_period=0.0, height=CAMERA_HW[0],
            width=CAMERA_HW[1], data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, horizontal_aperture=20.955, clipping_range=(0.005, 2.0)),
            offset=CameraCfg.OffsetCfg(pos=(-0.002, 0.0477, -0.0667), rot=(-0.326369, -0.326369, 0.627282, -0.627282),
                                       convention="world"),
        )

        # ---- gripper: the IK-Abs base caps grip torque at 1 N·m (tuned for cubes); the mug slips at that ----
        self.scene.robot.actuators["gripper"] = ImplicitActuatorCfg(
            joint_names_expr=["gripper"], joint_effort_limit=3.0, joint_velocity_limit=2.0, stiffness=17.8, damping=0.60)

        # ---- managers ----
        self.observations = MugBowlObservationsCfg()
        self.terminations = MugBowlTerminationsCfg()
        self.events = MugBowlEventsCfg()
        self.rerender_on_reset = True
