"""GR00T N1.7 modality config for the robot-lab SO-101 sim dataset (tools/to_gr00t.py output).

Same structure as Isaac-GR00T's official examples/SO100/so100_config.py (the SO-100/SO-101 post-training recipe),
with our camera names: `scene` (fixed third-person, 256x256) and `wrist` (gripper camera_mount, 256x256).
  * state: single_arm (5 joints) + gripper (1), current step only
  * action: 16-step chunk at 30 Hz; arm RELATIVE to the current state (official recipe), gripper ABSOLUTE
  * language: the dataset task string ("put the <object> in the yellow bowl")
Registered under EmbodimentTag.NEW_EMBODIMENT (the tag NVIDIA's SO-101 walkthrough uses for custom arms).

Passed to launch_finetune.py as --modality-config-path; the fine-tuned checkpoint stores the config, so the policy
server does not need it.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ACTION_HORIZON = 16

so101_sim_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["scene", "wrist"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["single_arm", "gripper"]),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(so101_sim_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
