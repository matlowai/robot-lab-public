"""Gym registrations for robot-lab tasks (import after the simulation app has launched)."""

import gymnasium as gym

gym.register(
    id="RobotLab-SO101-MugBowl-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.so101_mug_bowl:SO101MugBowlEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="RobotLab-SO101-PickPlace-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.so101_pick_place:SO101PickPlaceEnvCfg"},
    disable_env_checker=True,
)
gym.register(
    id="RobotLab-SO101-PickPlace-Heldout-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.so101_pick_place:SO101PickPlaceHeldoutEnvCfg"},
    disable_env_checker=True,
)
gym.register(
    id="RobotLab-SO101-PickPlace-Joint-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.so101_pick_place:SO101PickPlaceJointEnvCfg"},
    disable_env_checker=True,
)
