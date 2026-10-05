"""Contact forensics inside the RL training env (rl/env.py), for comparison with the sim2d benchmark's
(tools/contact_forensics.py). No training: the checkpoint only acts. CPU is fine.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tools/rl_env_contacts.py --ckpt <ckpt> --preset hard-dense \
        --mode sample,det --envs 256 --steps 1500 --out <dir>

Modes: sample = the training policy (tanh of a Gaussian sample, the training behaviour); det = tanh of the mean
(what rl/eval.py deploys); meanvx = det for vy / wz, but vx = E[tanh(mu + sigma * eps)] (the forward speed the
training noise actually produced on average), a deploy-time probe of the vx-noise mismatch.

Every contact terminates an env episode. The contact person is found from a snapshot taken just before the reset:
event (the scripted straight-line people; "standing" when their velocity is 0) | wanderer | hunter_lock (locked /
committed) | hunter_unlocked. Bearing, speeds and approach components as in contact_forensics (0.5 s windows).
Exposure = person-seconds within 4 m (centre distance) by kind, so rates compare across envs of different lengths.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from benchmarks.avoidance.pedestrians import COMMITTED, HUNTER, LOCKED, WANDERER
from benchmarks.avoidance.rl import env as E
from benchmarks.avoidance.rl.controller import load_policy

HIST = 11  # robot / people positions for the last 1.0 s (10 steps back + now)


class ProbeEnv(E.AvoidEnv):
    """AvoidEnv that snapshots the envs it is about to reset (the terminal state of a finished episode)."""
    capture = False

    def _reset_idx(self, ids):
        if self.capture and len(ids):
            pp, act = self.people()
            snap = {"ids": ids.clone(), "pos": self.pos[ids].clone(), "yaw": self.yaw[ids].clone(),
                    "vel": self.vel[ids].clone(), "pp": pp[ids].clone(), "act": act[ids].clone(),
                    "pvel": self.pvel[ids].clone(), "steps": self.steps[ids].clone()}
            if self.crowd is not None:
                snap["ckind"], snap["cstate"] = self.crowd.kind[ids].clone(), self.crowd.state[ids].clone()
            self.snaps.append(snap)
        super()._reset_idx(ids)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def run(ckpt, preset, mode, n_envs, steps, seed):
    model, ck = load_policy(ckpt)
    cfg = E.preset(preset, obs_version=ck["obs_version"])
    env = ProbeEnv(n_envs, "cpu", seed=seed, cfg=cfg)
    env.snaps = []
    obs = env.reset()
    env.capture = True
    std = model.log_std.detach().clamp(-5.0, 1.0).exp()
    P = env.P
    gh = torch.Generator().manual_seed(seed + 1)
    eps_q = torch.randn(256, generator=gh)  # quadrature points for meanvx
    hpos, hpp, hyaw = [], [], []  # ring buffers (python lists of tensors)
    recs, outcomes = [], Counter()
    expo = Counter()
    speeds, mus = [], []
    for k in range(steps):
        with torch.no_grad():
            mu = model.actor(model.obs_rms(obs))
            if mode == "sample":
                a = torch.tanh(mu + std * torch.randn(mu.shape))
            elif mode == "det":
                a = torch.tanh(mu)
            elif mode == "meanvx":
                a = torch.tanh(mu)
                a[:, 0] = torch.tanh(mu[:, 0:1] + std[0] * eps_q[None, :]).mean(-1)
            else:
                raise ValueError(mode)
        mus.append(mu[::4].clone())
        pp, act = env.people()
        hpos.append(env.pos.clone())
        hpp.append(pp.clone())
        hyaw.append(env.yaw.clone())
        if len(hpos) > HIST:
            hpos.pop(0)
            hpp.pop(0)
            hyaw.pop(0)
        # exposure (before the step, the state the policy acted on)
        d = torch.hypot(pp[..., 0] - env.pos[:, 0:1], pp[..., 1] - env.pos[:, 1:2])
        near = (d < 4.0) & act
        expo["event_s"] += float(near[:, :P].sum()) * E.DT
        if env.crowd is not None:
            ck_ = env.crowd.kind
            expo["wanderer_s"] += float((near[:, P:] & (ck_ == WANDERER)).sum()) * E.DT
            expo["hunter_s"] += float((near[:, P:] & (ck_ == HUNTER)).sum()) * E.DT
        expo["env_s"] += n_envs * E.DT
        speeds.append(torch.hypot(env.vel[:, 0], env.vel[:, 1]).mean().item())
        env.snaps = []
        obs, r, done, info = env.step(a)
        ep = info["episodes"]
        for key in ("success", "contact", "wall", "timeout"):
            outcomes[key] += int(ep[key].sum())
        if not env.snaps:
            continue
        snap = env.snaps[0]
        ids = snap["ids"]
        contact = ep["contact"]
        for j in torch.nonzero(contact).squeeze(-1).tolist():
            i = int(ids[j])
            rp, pp_ = snap["pos"][j], snap["pp"][j]
            dd = torch.hypot(pp_[:, 0] - rp[0], pp_[:, 1] - rp[1]) - E.PERSON_R - E.SPOT_R
            dd = torch.where(snap["act"][j], dd, torch.full_like(dd, float("inf")))
            p = int(dd.argmin())
            if p < P:
                who = "event_standing" if float(snap["pvel"][j, p].abs().sum()) == 0 else "event_walker"
            else:
                h = p - P
                kind, st = int(snap["ckind"][j, h]), int(snap["cstate"][j, h])
                who = "wanderer" if kind == WANDERER else ("hunter_lock" if st in (LOCKED, COMMITTED) else "hunter_unlocked")
            n_steps = int(snap["steps"][j])
            yaw = float(snap["yaw"][j])

            def bearing(pos_r, pos_p, yaw=yaw):
                v = pos_p - pos_r
                return math.degrees(_wrap(math.atan2(float(v[1]), float(v[0])) - yaw))

            rec = {"mode": mode, "who": who, "steps": n_steps, "bearing_0s": round(bearing(rp, pp_[p]), 1)}
            # hpos[-1] is the state before this step (0.1 s before contact); hpos[-6] ~0.5 s before contact
            if n_steps >= 6 and len(hpos) >= 6:
                back = 5  # hpos[-5] = 0.5 s before the contact state
                vr = (rp - hpos[-back][i]) / (back * E.DT)
                vp = (pp_[p] - hpp[-back][i, p]) / (back * E.DT)
                nrm = (pp_[p] - rp) / max(float(torch.linalg.norm(pp_[p] - rp)), 1e-9)
                rec |= {"robot_approach": round(float(vr @ nrm), 2), "person_approach": round(float(-vp @ nrm), 2),
                        "robot_speed": round(float(torch.hypot(snap["vel"][j, 0], snap["vel"][j, 1])), 2)}
            if n_steps >= HIST and len(hpos) >= HIST:
                rec["bearing_1s"] = round(bearing(hpos[0][i], hpp[0][i, p], float(hyaw[0][i])), 1)
            recs.append(rec)
    mu = torch.cat(mus).numpy()
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    mu_q = {name: [round(float(v), 2) for v in np.quantile(mu[:, i], q)] for i, name in enumerate(("vx", "vy", "wz"))}
    sat = {name: round(float((np.abs(np.tanh(mu[:, i])) > 0.95).mean()), 3) for i, name in enumerate(("vx", "vy", "wz"))}
    return recs, outcomes, expo, float(np.mean(speeds)), std.tolist(), {"mu_quantiles": dict(zip(["q"], [q])) | mu_q,
                                                                        "tanh_mu_saturated_frac": sat}


def summarize(recs, outcomes, expo, speed, mode):
    who = Counter(r["who"] for r in recs)
    amb = [r for r in recs if r["who"] != "hunter_lock"]
    moving = [r for r in amb if "robot_approach" in r]
    pin = sum(1 for r in moving if r["person_approach"] >= 0.2 and r["robot_approach"] < 0.2)
    rin = sum(1 for r in moving if r["robot_approach"] >= 0.2 and r["person_approach"] < 0.2)
    both = sum(1 for r in moving if r["robot_approach"] >= 0.2 and r["person_approach"] >= 0.2)
    behind = sum(1 for r in amb if abs(r["bearing_0s"]) > 135)
    eps = sum(outcomes.values())
    return {
        "mode": mode, "episodes": eps, "success": outcomes["success"] / max(eps, 1),
        "contact": outcomes["contact"] / max(eps, 1), "timeout": outcomes["timeout"] / max(eps, 1),
        "mean_speed": round(speed, 3), "contacts_by_who": dict(who),
        "wanderer_contacts_per_100_wanderer_s_4m": round(100 * who["wanderer"] / max(expo["wanderer_s"], 1e-9), 2),
        "hunter_unlocked_per_100_hunter_s_4m": round(100 * who["hunter_unlocked"] / max(expo["hunter_s"], 1e-9), 2),
        "event_walker_per_100_event_s_4m": round(100 * (who["event_walker"] + who["event_standing"]) / max(expo["event_s"], 1e-9), 2),
        "ambient_contacts_per_1000_s": round(1000 * len(amb) / max(expo["env_s"], 1e-9), 2),
        "exposure_s": {k: round(v) for k, v in expo.items()},
        "ambient_mover": {"person_into_robot": pin, "robot_into_person": rin, "both": both, "n": len(moving)},
        "ambient_behind_135": behind, "ambient_n": len(amb),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", default="hard-dense")
    ap.add_argument("--mode", default="sample,det")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for mode in a.mode.split(","):
        t0 = time.time()
        recs, outcomes, expo, speed, std, mu = run(a.ckpt, a.preset, mode, a.envs, a.steps, a.seed)
        s = summarize(recs, outcomes, expo, speed, mode) | {"wall_s": round(time.time() - t0, 1), "std_pre_tanh": std,
                                                            "preset": a.preset, "ckpt": a.ckpt} | mu
        summaries.append(s)
        (out / f"contacts-{mode}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        print(json.dumps(s), flush=True)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
