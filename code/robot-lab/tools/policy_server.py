"""FLUX 3 Action SO-101 policy server (LeRobot venv). Isaac (another venv) connects over a local socket.

  /mnt/weights/ai/lerobot/lerobot/.venv/bin/python tools/policy_server.py --policy <dir> [--port 6100]

<dir> is either the complete SO-101 package (base policy) or a LeRobot LoRA checkpoint's pretrained_model(_ema) dir,
which references its base. Encoders are always the local copies (no 11 GB re-download).

Protocol (multiprocessing.connection, pickled dicts):
  {"cmd": "reset"}                                             -> {"ok": True}
  {"cmd": "act", "scene": HxWx3 uint8, "wrist": HxWx3 uint8,
   "state": (6,) float32 in dataset units, "task": str}         -> {"action": (6,) float32 dataset units, "ms": float}
  {"cmd": "close"}
Call "act" on EVERY control tick: the processors buffer observation/command history and the policy queues its chunk.
"""

import argparse
import time
from multiprocessing.connection import Listener

import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.flux3 import Flux3Policy

BASE = "/mnt/weights/ai/flux3-action/flux-3-action-base"
p = argparse.ArgumentParser()
p.add_argument("--policy", required=True)
p.add_argument("--port", type=int, default=6100)
p.add_argument("--compile", action="store_true")
args = p.parse_args()

cfg = PreTrainedConfig.from_pretrained(args.policy)
cfg.video_vae_id = f"{BASE}/video_vae.safetensors"
cfg.text_encoder_id = f"{BASE}/text_encoder"
cfg.device = "cuda"
if args.compile:
    cfg.compile_model = True
if getattr(cfg, "use_peft", False):
    # LoRA checkpoint: load the base it references, then apply the adapter (same path as lerobot.policies.factory)
    from peft import PeftConfig, PeftModel

    pc = PeftConfig.from_pretrained(args.policy)
    base = Flux3Policy.from_pretrained(pc.base_model_name_or_path, config=cfg)
    policy = PeftModel.from_pretrained(base, args.policy, config=pc, is_trainable=False).to("cuda").eval()
    print(f"SERVER adapter on base {pc.base_model_name_or_path}", flush=True)
else:
    policy = Flux3Policy.from_pretrained(args.policy, config=cfg).to("cuda").eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=args.policy)
device = torch.device("cuda")
pcfg = cfg
print(f"SERVER ready policy={args.policy} port={args.port} n_obs_steps={pcfg.n_obs_steps} "
      f"chunk={pcfg.chunk_size} n_action_steps={pcfg.n_action_steps}", flush=True)

with Listener(("127.0.0.1", args.port), authkey=b"robot-lab") as listener:
    while True:
        conn = listener.accept()
        try:
            while True:
                msg = conn.recv()
                if msg["cmd"] == "reset":
                    policy.reset()
                    pre.reset()
                    post.reset()
                    conn.send({"ok": True})
                elif msg["cmd"] == "act":
                    obs = {"observation.images.scene": msg["scene"], "observation.images.wrist": msg["wrist"],
                           "observation.state": np.asarray(msg["state"], np.float32)}
                    t0 = time.perf_counter()
                    a = predict_action(obs, policy, device, pre, post, use_amp=False, task=msg["task"],
                                       robot_type="so101_follower")
                    torch.cuda.synchronize()
                    conn.send({"action": a.float().cpu().numpy().reshape(-1)[:6], "ms": (time.perf_counter() - t0) * 1e3})
                elif msg["cmd"] == "close":
                    conn.close()
                    raise SystemExit(0)
        except EOFError:
            conn.close()
