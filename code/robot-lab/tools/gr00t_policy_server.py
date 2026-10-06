"""GR00T N1.7 SO-101 policy server (Isaac-GR00T venv). Isaac Lab (its own venv) connects over a local socket.

  cd /mnt/weights/ai/nvidia-action/Isaac-GR00T && CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      /mnt/work/AI/robot-lab/tools/gr00t_policy_server.py --policy <checkpoint dir> [--port 6100] [--exec_horizon 8]

Speaks the SAME protocol as tools/policy_server.py (FLUX), so tools/gr00t_eval.py drives either:
  {"cmd": "reset"}                                             -> {"ok": True}
  {"cmd": "act", "scene": HxWx3 uint8, "wrist": HxWx3 uint8,
   "state": (6,) float32 dataset units, "task": str}           -> {"action": (6,) float32 dataset units, "ms": float,
                                                                   "planned": bool}
  {"cmd": "close"}
Receding horizon: GR00T predicts a 16-step chunk (0.53 s at 30 Hz); the server executes the first --exec_horizon
steps, one per "act" call, then replans from the newest observation. "planned" marks ticks that ran the model, so
latency is measured on real inferences (FLUX's ms>50 heuristic would misfire on a fast model).
"""

import argparse
import time
from multiprocessing.connection import Listener

import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

p = argparse.ArgumentParser()
p.add_argument("--policy", required=True, help="fine-tuned checkpoint dir (checkpoint-N)")
p.add_argument("--port", type=int, default=6100)
p.add_argument("--exec_horizon", type=int, default=8)
p.add_argument("--embodiment_tag", default="NEW_EMBODIMENT")
args = p.parse_args()

t0 = time.time()
policy = Gr00tPolicy(EmbodimentTag.resolve(args.embodiment_tag), args.policy, device="cuda:0", strict=True)
mc = policy.get_modality_config()
horizon = len(mc["action"].delta_indices)
assert 1 <= args.exec_horizon <= horizon, (args.exec_horizon, horizon)
assert mc["video"].modality_keys == ["scene", "wrist"], mc["video"].modality_keys

queue: list[np.ndarray] = []


def plan(msg) -> list[np.ndarray]:
    s = np.asarray(msg["state"], np.float32)
    obs = {
        "video": {"scene": msg["scene"][None, None], "wrist": msg["wrist"][None, None]},  # (B=1, T=1, H, W, 3) uint8
        "state": {"single_arm": s[None, None, :5], "gripper": s[None, None, 5:6]},       # (B, T, D)
        "language": {"annotation.human.task_description": [[msg["task"]]]},
    }
    chunk, _ = policy.get_action(obs)
    a = np.concatenate([chunk["single_arm"][0], chunk["gripper"][0]], axis=-1)  # (horizon, 6) dataset units
    return list(a[: args.exec_horizon])


with Listener(("127.0.0.1", args.port), authkey=b"robot-lab") as listener:
    # announce readiness only after the port is bound (2026-10-05: a "ready" printed before bind hid an
    # "Address already in use" and the eval client then talked to another process on that port)
    print(f"SERVER ready policy={args.policy} port={args.port} chunk={horizon} exec_horizon={args.exec_horizon} "
          f"video={mc['video'].modality_keys} state={mc['state'].modality_keys} load_s={time.time() - t0:.1f}", flush=True)
    while True:
        conn = listener.accept()
        try:
            while True:
                msg = conn.recv()
                if msg["cmd"] == "reset":
                    queue.clear()
                    policy.reset()
                    conn.send({"ok": True})
                elif msg["cmd"] == "act":
                    t = time.perf_counter()
                    planned = not queue
                    if planned:
                        queue.extend(plan(msg))
                        torch.cuda.synchronize()
                    conn.send({"action": queue.pop(0).astype(np.float32), "ms": (time.perf_counter() - t) * 1e3,
                               "planned": planned})
                elif msg["cmd"] == "close":
                    conn.close()
                    raise SystemExit(0)
        except EOFError:
            conn.close()
