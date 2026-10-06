"""Dataset units <-> sim radians, and the constants shared by both venvs (numpy only, no torch / Isaac / GR00T).

Same conversion as tools/gr00t_eval.py (to_units / from_units), read from the dataset's units.json:
  arm  : degrees(sim rad) + per-joint offset
  jaw  : (degrees(sim rad) - GMIN) / (GMAX - GMIN) * 100      (sim -10..100 deg -> 0..100 units)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_UNITS = "/mnt/weights/ai/robot-lab-data/overnight/full-20260923-2207/lerobot_ds/units.json"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CONTROL_HZ = 30
CHUNK = 16            # GR00T action horizon for this embodiment (gr00t_so101_config.ACTION_HORIZON)
EXEC_H = 8            # executed steps per plan (night-1 eval setting) == one RL step
JAW_PRESHAPE_RAD = 0.785   # dataset command 50 = 45 deg (tools/gr00t_eval.py:78)
JAW_SUCCESS_RAD = 0.5      # robot_lab/tasks/so101_pick_place.py object_in_bowl "opened" threshold


class Units:
    def __init__(self, path: str = DEFAULT_UNITS):
        u = json.loads(Path(path).read_text())
        assert u["joints"] == JOINTS, u["joints"]
        self.off = np.asarray(u["arm_offsets_deg"], np.float32)
        self.gmin, self.gmax = (float(x) for x in u["gripper"]["sim_deg_range"])

    def to_units(self, q: np.ndarray) -> np.ndarray:
        """(..., 6) rad -> (..., 6) dataset units."""
        q = np.asarray(q, np.float32)
        u = np.empty_like(q)
        u[..., :5] = np.degrees(q[..., :5]) + self.off
        u[..., 5] = (np.degrees(q[..., 5]) - self.gmin) / (self.gmax - self.gmin) * 100.0
        return u

    def from_units(self, u: np.ndarray) -> np.ndarray:
        """(..., 6) dataset units -> (..., 6) rad."""
        u = np.asarray(u, np.float32)
        q = np.empty_like(u)
        q[..., :5] = np.radians(u[..., :5] - self.off)
        q[..., 5] = np.radians(u[..., 5] / 100.0 * (self.gmax - self.gmin) + self.gmin)
        return q

    def jaw_rad_to_units(self, rad: float) -> float:
        return (np.degrees(rad) - self.gmin) / (self.gmax - self.gmin) * 100.0
