"""GR00T N1.7 fine-tune with RELEASE/GRASP WEIGHTING (GR00T night 2). Wraps Isaac-GR00T, source untouched.

Runs gr00t/experiment/launch_finetune.py unchanged (same CLI, run via runpy) after patching ONE method in-process:
Gr00tN1d7Processor.__call__. When training, if the sample's raw 16-step gripper action chunk crosses the open/closed
midpoint (OPEN_UNITS = 30 dataset units; closed/holding ~9, open ~50) -- i.e. a gripper close (grasp) or open
(release) transition lies within the predicted chunk -- the sample's action_mask is multiplied by RELEASE_WEIGHT.

Why this is an exact per-sample loss weight: the action head's loss is
  sum(mse * action_mask) / sum(action_mask)          (gr00t/model/gr00t_n1d7/gr00t_n1d7.py:277-278)
so scaling a sample's mask by w gives the weighted mean  sum_b w_b * L_b / sum_b w_b * n_b.
What is weighted, exactly: training samples (frame t) whose action targets a[t..t+15] contain a gripper command
transition = the 15 frames before each grasp close and each release open, plus the transition frame. Both
transitions are weighted (the night-1 failure was the release; the grasp transition is the same kind of rare,
binary event). Not weighted: everything else, including the post-release tail (now ~40 frames per demo).

Settings via env (tyro owns the CLI): GR00T_RELEASE_WEIGHT (default 5.0), GR00T_OPEN_UNITS (30.0),
GR00T_WEIGHT_LOG_EVERY (5000 samples per dataloader worker). Each worker prints
  RELEASE_WEIGHT pid=<pid> seen=<n> weighted=<k> frac=<k/n> weight=<w>
so the log proves the patch is live inside the (forked) dataloader workers; the orchestrator fails the stage if no
such line with weighted > 0 appears.

  cd <Isaac-GR00T> && GR00T_RELEASE_WEIGHT=5 .venv/bin/python /mnt/work/AI/robot-lab/tools/gr00t_finetune_weighted.py \
      --base_model_path ... --dataset_path ... (exactly launch_finetune.py's arguments)
"""

import os
import runpy
import sys

import numpy as np

GR = "/mnt/weights/ai/nvidia-action/Isaac-GR00T"
W = float(os.environ.get("GR00T_RELEASE_WEIGHT", "5.0"))
OPEN = float(os.environ.get("GR00T_OPEN_UNITS", "30.0"))
EVERY = int(os.environ.get("GR00T_WEIGHT_LOG_EVERY", "5000"))

sys.path.insert(0, GR)
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor  # noqa: E402

_orig_call = Gr00tN1d7Processor.__call__
_count = {"seen": 0, "weighted": 0}


def _weighted_call(self, messages):
    out = _orig_call(self, messages)
    if self.training and out.get("action_mask") is not None:
        g = np.asarray(messages[0]["content"].actions["gripper"], np.float32).reshape(-1)
        crosses = bool((g > OPEN).any() and (g <= OPEN).any())
        _count["seen"] += 1
        if crosses:
            out["action_mask"] = out["action_mask"] * W
            _count["weighted"] += 1
        if _count["seen"] == 200 or _count["seen"] % EVERY == 0:
            print(f"RELEASE_WEIGHT pid={os.getpid()} seen={_count['seen']} weighted={_count['weighted']} "
                  f"frac={_count['weighted'] / _count['seen']:.3f} weight={W} open_units={OPEN}", flush=True)
    return out


Gr00tN1d7Processor.__call__ = _weighted_call
print(f"RELEASE_WEIGHT patch installed: weight={W} open_units={OPEN} (Gr00tN1d7Processor.__call__)", flush=True)

os.chdir(GR)
sys.argv = [f"{GR}/gr00t/experiment/launch_finetune.py", *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
