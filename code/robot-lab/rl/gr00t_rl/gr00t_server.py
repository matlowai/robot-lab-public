"""Batched GR00T N1.7 planning server for RL (runs in the Isaac-GR00T venv; Isaac Lab connects over a local socket).

  cd /mnt/weights/ai/nvidia-action/Isaac-GR00T && CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=<ffmpeg7 lib> \
      .venv/bin/python /mnt/work/AI/robot-lab/rl/gr00t_rl/gr00t_server.py --policy <checkpoint-N> --port 6190
  ... --selftest      # parity of our noise-injected denoiser vs GR00T's own get_action (same noise), then exit
  ... --bench 1,8,32  # latency / VRAM per batch size on random images, then exit

Why a separate process: Isaac Lab's venv has torch 2.11, GR00T's has torch 2.9 + flash-attn built for it, so GR00T
cannot be imported into the simulator process. Unlike tools/gr00t_policy_server.py (one env, one obs per call), one
"plan" call here carries ALL envs' observations and runs ONE batched forward pass.

What this adds over Gr00tPolicy.get_action (whose code we call piecewise, never edit):
  * noise injection: the flow-matching head starts from torch.randn inside
    Gr00tN1d7ActionHead.get_action_with_features (gr00t/model/gr00t_n1d7/gr00t_n1d7.py:349). We re-implement that
    loop (no RTC branch) with the initial noise passed in, so (a) eval episodes can use common random numbers per
    (seed, env, chunk) for base vs RL comparisons, and (b) a noise-steering (DSRL) actor can overwrite the noise
    block of the used action dims. --selftest checks parity with the original on identical noise.
  * a pooled vision embedding per env (mean of the action head's processed backbone features over IMAGE tokens),
    returned so the residual actor can see what GR00T sees without any privileged simulator state.

Protocol (multiprocessing.connection, authkey b"robot-lab-rl"):
  {"cmd": "info"} -> {"chunk", "max_action_dim", "max_horizon", "emb_dim", "policy"}
  {"cmd": "plan", "scene": (B,H,W,3) u8, "wrist": (B,H,W,3) u8, "state": (B,6) f32 dataset units, "tasks": [str]*B,
   "noise_seeds": (B,) int64 or None, "noise": (B,h,d) f32 or None}
      -> {"actions": (B,16,6) f32 dataset units, "emb": (B,E) f16, "ms": {"pre", "model", "post", "total"}}
  {"cmd": "close"} closes this connection (server keeps listening);  {"cmd": "shutdown"} exits.
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener

import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

AUTHKEY = b"robot-lab-rl"


class BatchedGr00t:
    def __init__(self, ckpt: str, device: str = "cuda:0", embodiment: str = "NEW_EMBODIMENT"):
        self.policy = Gr00tPolicy(EmbodimentTag.resolve(embodiment), ckpt, device=device, strict=False)
        self.model = self.policy.model
        self.head = self.model.action_head
        self.device = torch.device(device)
        mc = self.policy.get_modality_config()
        assert mc["video"].modality_keys == ["scene", "wrist"], mc["video"].modality_keys
        assert mc["state"].modality_keys == ["single_arm", "gripper"], mc["state"].modality_keys
        self.chunk = len(mc["action"].delta_indices)
        self.H, self.D = self.head.config.action_horizon, self.head.action_dim  # padded noise shape (40, 132)
        self.emb_dim = self.head.config.backbone_embedding_dim
        self.ckpt = ckpt

    # ---- inputs -------------------------------------------------------------------------------------------------
    def _collate(self, scene, wrist, state, tasks):
        obs = {
            "video": {"scene": scene[:, None], "wrist": wrist[:, None]},
            "state": {"single_arm": state[:, None, :5].astype(np.float32),
                      "gripper": state[:, None, 5:6].astype(np.float32)},
            "language": {self.policy.language_key: [[t] for t in tasks]},
        }
        processed, states = [], []
        for o in self.policy._unbatch_observation(obs):
            v = self.policy._to_vla_step_data(o)
            states.append(v.states)
            processed.append(self.policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": v}]))
        coll = _rec_to_dtype(self.policy.collate_fn(processed), dtype=torch.bfloat16)
        bstates = {k: np.stack([s[k] for s in states], 0) for k in ("single_arm", "gripper")}
        return coll, bstates

    def make_noise(self, B: int, seeds=None, override=None) -> torch.Tensor:
        """(B, H, D) bf16 initial noise. seeds: per-env generator seeds (so env i's noise never depends on the batch);
        None -> global RNG, drawn exactly like GR00T (torch.randn, bf16, on device). override (B, h, d): written into
        the leading [:h, :d] block (the used horizon x action dims) -- the noise-steering hook."""
        if seeds is None:
            noise = torch.randn((B, self.H, self.D), dtype=torch.bfloat16, device=self.device)
        else:
            rows = []
            for s in np.asarray(seeds, np.int64).reshape(-1):
                g = torch.Generator(device=self.device).manual_seed(int(s) & 0x7FFFFFFFFFFFFFFF)
                rows.append(torch.randn((self.H, self.D), generator=g, device=self.device, dtype=torch.float32))
            noise = torch.stack(rows).to(torch.bfloat16)
        if override is not None:
            o = torch.as_tensor(np.asarray(override, np.float32), device=self.device).to(torch.bfloat16)
            noise[:, : o.shape[1], : o.shape[2]] = o
        return noise

    # ---- the denoiser: Gr00tN1d7ActionHead.get_action_with_features without RTC, initial noise passed in ------
    @torch.no_grad()
    def _denoise(self, feats, backbone_output, embodiment_id, actions):
        head, cfg = self.head, self.head.config
        vl, sf = feats.backbone_features, feats.state_features
        B = vl.shape[0]
        dt = 1.0 / head.num_inference_timesteps
        for t in range(head.num_inference_timesteps):
            t_disc = int(t / float(head.num_inference_timesteps) * head.num_timestep_buckets)
            ts = torch.full(size=(B,), fill_value=t_disc, device=vl.device)
            af = head.action_encoder(actions, ts, embodiment_id)
            if cfg.add_pos_embed:
                pos = torch.arange(af.shape[1], dtype=torch.long, device=vl.device)
                af = af + head.position_embedding(pos).unsqueeze(0)
            sa = torch.cat((sf, af), dim=1)
            if cfg.use_alternate_vl_dit:
                out = head.model(hidden_states=sa, encoder_hidden_states=vl, timestep=ts,
                                 image_mask=backbone_output.image_mask,
                                 backbone_attention_mask=backbone_output.backbone_attention_mask)
            else:
                out = head.model(hidden_states=sa, encoder_hidden_states=vl, timestep=ts)
            pred = head.action_decoder(out, embodiment_id)
            actions = actions + dt * pred[:, -head.action_horizon:]
        return actions

    @torch.no_grad()
    def plan(self, scene, wrist, state, tasks, noise_seeds=None, noise=None):
        t0 = time.perf_counter()
        B = len(tasks)
        coll, bstates = self._collate(scene, wrist, state, tasks)
        t1 = time.perf_counter()
        with torch.inference_mode():
            bb_in, act_in = self.model.prepare_input(coll["inputs"])
            bb_out = self.model.backbone(bb_in)
            feats = self.head._encode_features(bb_out, act_in)  # applies vlln + vl_self_attention to bb_out in place
            a0 = self.make_noise(B, noise_seeds, noise)
            act = self._denoise(feats, bb_out, act_in.embodiment_id, a0)
            img = bb_out.image_mask.unsqueeze(-1).to(feats.backbone_features.dtype)
            emb = (feats.backbone_features * img).sum(1) / img.sum(1).clamp_min(1.0)
            act = act.float().cpu().numpy()
            emb = emb.float().cpu().numpy().astype(np.float16)
        t2 = time.perf_counter()
        dec = self.policy.processor.decode_action(act, self.policy.embodiment_tag, bstates)
        a = np.concatenate([dec["single_arm"], dec["gripper"]], axis=-1).astype(np.float32)  # (B, chunk, 6) units
        t3 = time.perf_counter()
        ms = {"pre": (t1 - t0) * 1e3, "model": (t2 - t1) * 1e3, "post": (t3 - t2) * 1e3, "total": (t3 - t0) * 1e3}
        return a, emb, ms


def _random_obs(B, rng):
    scene = rng.integers(0, 255, (B, 256, 256, 3), dtype=np.uint8)
    wrist = rng.integers(0, 255, (B, 256, 256, 3), dtype=np.uint8)
    state = np.tile(np.array([0.0, 100.0, 90.0, 60.0, 0.0, 16.0], np.float32), (B, 1))
    state += rng.normal(0, 2, state.shape).astype(np.float32)
    return scene, wrist, state, ["put the mug in the yellow bowl"] * B


def selftest(g: BatchedGr00t) -> int:
    """Our denoiser + injected noise must reproduce GR00T's own get_action given the same noise draw."""
    rng = np.random.default_rng(0)
    scene, wrist, state, tasks = _random_obs(4, rng)
    coll, bstates = g._collate(scene, wrist, state, tasks)
    torch.manual_seed(1234)
    with torch.inference_mode():
        ref = g.model.get_action(**coll)["action_pred"].float().cpu().numpy()
    torch.manual_seed(1234)
    ref_noise = torch.randn((4, g.H, g.D), dtype=torch.bfloat16, device=g.device)  # the draw GR00T made
    coll, bstates = g._collate(scene, wrist, state, tasks)
    with torch.inference_mode():
        bb_in, act_in = g.model.prepare_input(coll["inputs"])
        bb_out = g.model.backbone(bb_in)
        feats = g.head._encode_features(bb_out, act_in)
        ours = g._denoise(feats, bb_out, act_in.embodiment_id, ref_noise.clone()).float().cpu().numpy()
    d = np.abs(ref - ours)[:, : g.chunk, :6]
    print(f"SELFTEST parity: max|ref-ours| over used block = {d.max():.3e}, mean = {d.mean():.3e} "
          f"(normalized action space; ref |a| mean {np.abs(ref[:, :g.chunk, :6]).mean():.3f})", flush=True)
    a1, _, _ = g.plan(scene, wrist, state, tasks, noise_seeds=[7, 8, 9, 10])
    a2, _, _ = g.plan(scene, wrist, state, tasks, noise_seeds=[7, 8, 9, 10])
    a3, _, _ = g.plan(scene[[1, 0, 2, 3]], wrist[[1, 0, 2, 3]], state[[1, 0, 2, 3]], tasks, noise_seeds=[8, 7, 9, 10])
    a4, _, _ = g.plan(scene, wrist, state, tasks, noise_seeds=[107, 108, 109, 110])
    rep = np.abs(a1 - a2).max()
    perm = np.abs(a1[[1, 0, 2, 3]] - a3).max()
    diff = np.abs(a1 - a4).max()
    print(f"SELFTEST seeded noise: same seeds max diff {rep:.3e} units; permuted batch max diff {perm:.3e} units; "
          f"other seeds max diff {diff:.3f} units (must be > 0)", flush=True)
    ok = d.max() < 5e-2 and rep < 0.5 and perm < 0.5 and diff > 1e-3
    print("SELFTEST", "OK" if ok else "FAIL", flush=True)
    return 0 if ok else 1


def bench(g: BatchedGr00t, sizes) -> None:
    rng = np.random.default_rng(0)
    for B in sizes:
        obs = _random_obs(B, rng)
        g.plan(*obs)  # warm-up
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(5):
            _, _, ms = g.plan(*obs, noise_seeds=np.arange(B))
            ts.append(ms)
        med = {k: float(np.median([t[k] for t in ts])) for k in ts[0]}
        print(f"BENCH B={B} total_ms={med['total']:.1f} pre={med['pre']:.1f} model={med['model']:.1f} "
              f"post={med['post']:.1f} per_env_ms={med['total'] / B:.2f} "
              f"peak_alloc_GB={torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--port", type=int, default=6190)  # NOT 6120-6139: the imitation pipelines use those
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--bench", default="")
    args = p.parse_args()
    t0 = time.time()
    g = BatchedGr00t(args.policy)
    print(f"LOADED policy={args.policy} chunk={g.chunk} noise=({g.H},{g.D}) emb_dim={g.emb_dim} "
          f"load_s={time.time() - t0:.1f}", flush=True)
    if args.selftest:
        raise SystemExit(selftest(g))
    if args.bench:
        bench(g, [int(x) for x in args.bench.split(",")])
        return
    with Listener(("127.0.0.1", args.port), authkey=AUTHKEY) as listener:
        print(f"SERVER ready port={args.port}", flush=True)
        while True:
            try:
                conn = listener.accept()
            except (AuthenticationError, OSError, EOFError) as e:  # a foreign client (wrong authkey) must not kill us
                print(f"SERVER rejected a connection: {type(e).__name__}: {e}", flush=True)
                continue
            try:
                while True:
                    msg = conn.recv()
                    cmd = msg["cmd"]
                    if cmd == "plan":
                        a, emb, ms = g.plan(msg["scene"], msg["wrist"], msg["state"], msg["tasks"],
                                            msg.get("noise_seeds"), msg.get("noise"))
                        conn.send({"actions": a, "emb": emb, "ms": ms})
                    elif cmd == "info":
                        conn.send({"chunk": g.chunk, "max_horizon": g.H, "max_action_dim": g.D,
                                   "emb_dim": g.emb_dim, "policy": g.ckpt})
                    elif cmd == "close":
                        conn.close()
                        break
                    elif cmd == "shutdown":
                        conn.close()
                        print("SERVER shutdown", flush=True)
                        return
            except EOFError:
                conn.close()


if __name__ == "__main__":
    main()
