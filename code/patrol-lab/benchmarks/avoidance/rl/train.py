"""Train the PPO avoidance policy in the vectorised env.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -m benchmarks.avoidance.rl.train --envs 4096 --hours 2.5 \
        --out /mnt/weights/ai/patrol-lab-data/rl/avoid-ppo-<stamp> --seed 0 --preset hard

    ... --preset hard-dense --obs avoid-v2 --ramp w_space=0:0.25:0.5   # RL v3 (D44)

--preset picks the env (rl/env.py::PRESETS): legacy = straight-line readers only; base / hard = closed-loop hunters
of that crowd.LIVE_TIERS tier on top of ambient traffic; hard-dense = hard + the benchmark's wanderer density near
the robot + the personal-space reward (D44).

Writes <out>/train.csv (one row per PPO update), <out>/config.json, ckpt_<update>.pt every --ckpt-min minutes and
at the end, and latest.pt (always the newest checkpoint). Stats in the CSV are over the episodes that finished
during that update's rollout, with the stochastic (training) policy. Closed-loop columns: lock_rate = share of
episodes in which at least one hunter locked on, locks_per_ep / commits_per_ep, hit_per_commit = hunter contacts
per committed hunter (NaN when nothing committed). Reward bookkeeping: r_<term> = that term's mean per-episode sum
(rl/env.py::REWARD_TERMS: progress, time, smooth, near, space, contact, wall, success; they add up to mean_return),
space_frac = mean share of an episode's steps with someone inside the personal-space radius (0 when it is off).

--set FIELD=JSON overrides a preset field for the whole run (ablations). --ramp FIELD=START:F0:F1 is a curriculum on
a float field: START until F0 of the time budget, then linear to the preset's value at F1, which it keeps. Episodes
sample the current value when they start; the reward weights apply from the next step. Ramped fields get a
"cfg_<field>" column; config.json and every checkpoint record the preset's (final) values plus the ramp in args.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn

from benchmarks.avoidance.rl.env import (ACT_DIM, OBS_SPECS, OBS_VERSION, PRESETS, REWARD_TERMS, AvoidEnv, EnvConfig,
                                         obs_spec, preset)
from benchmarks.avoidance.rl.ppo import ActorCritic, PPOConfig, gae, mlp, ppo_update

CSV_FIELDS = ["update", "env_steps", "wall_s", "mean_return", "success_rate", "contact_rate", "wall_rate",
              "timeout_rate", "near_miss_rate", "mean_ep_len", "episodes", "fps", "pi_loss", "v_loss", "entropy",
              "kl", "clipfrac", "std_vx", "std_vy", "std_wz", "lock_rate", "locks_per_ep", "commits_per_ep",
              "hit_per_commit", "space_frac", *(f"r_{k}" for k in REWARD_TERMS),
              # D45 / review diagnostics: is the forward-speed channel alive? (rollout samples, env-side action)
              "ttc_frac", "mu_vx_mean", "mu_vx_sat", "vx_mid_frac", "actor_frozen",
              # D49 PPO-Lagrangian: mean ambient cost per step (rollout), the multiplier, cost-critic loss
              "ambient_contact_rate", "cost_step", "lambda", "c_loss"]
EP_KEYS = ("ret", "len", "success", "contact", "wall", "timeout", "near", "locks", "commits", "hits", "space_frac",
           "ttc_frac", "cost", "ambient_contact", *(f"r_{k}" for k in REWARD_TERMS))


def save_ckpt(path: Path, model: ActorCritic, opt, update: int, env_steps: int, env_cfg: EnvConfig, ppo_cfg: PPOConfig,
              hidden, extra: dict | None = None) -> None:
    spec = obs_spec(env_cfg.obs_version)
    torch.save((extra or {}) | {
        "obs_version": env_cfg.obs_version, "obs_layout": spec["layout"], "obs_dim": spec["dim"], "act_dim": ACT_DIM,
        "hidden": list(hidden), "model": model.state_dict(), "optimizer": opt.state_dict(),
        "obs_rms": {"mean": model.obs_rms.mean.cpu(), "var": model.obs_rms.var.cpu(), "count": model.obs_rms.count.cpu()},
        "update": update, "env_steps": env_steps, "env_cfg": asdict(env_cfg), "ppo_cfg": asdict(ppo_cfg),
        "squash": model.squash, "log_std_max": model.log_std_max,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--hours", type=float, default=2.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--ckpt-min", type=float, default=15.0)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--preset", default="legacy", choices=sorted(PRESETS), help="env preset (rl/env.py::PRESETS)")
    ap.add_argument("--obs", default=OBS_VERSION, choices=sorted(OBS_SPECS),
                    help="observation version (rl/env.py::OBS_SPECS); recorded in every checkpoint")
    ap.add_argument("--squash", default="tanh", choices=("tanh", "clip"),
                    help="policy output: tanh-squashed Gaussian (v1-v3) or clipped Gaussian (rsl_rl style, D46)")
    ap.add_argument("--log-std-max", type=float, default=1.0, help="upper clamp on log std (v1-v3: 1.0 = std e)")
    ap.add_argument("--bound-coef", type=float, default=0.0, help="bounds loss on the mean beyond +-1 (rl_games)")
    ap.add_argument("--ent-coef", type=float, default=None, help="override PPOConfig.ent_coef")
    ap.add_argument("--lr", type=float, default=None, help="override PPOConfig.lr")
    ap.add_argument("--init-ckpt", default=None,
                    help="warm start: actor, log std and obs normalisation from this checkpoint (e.g. a BC policy)")
    ap.add_argument("--critic-warmup", type=int, default=0,
                    help="with --init-ckpt: train only the critic for this many updates before the actor moves")
    ap.add_argument("--init-critic", action="store_true",
                    help="with --init-ckpt: also load the critic (and a cost critic if the checkpoint has one)")
    ap.add_argument("--cost-limit", type=float, default=None,
                    help="D49 PPO-Lagrangian: keep the mean ambient cost per step (needs --set ambient_cost=true) at or "
                         "below this; a separate cost critic and multiplier lambda (advantage (A_r - lam A_c)/(1 + lam))")
    ap.add_argument("--lambda-lr", type=float, default=0.02, help="lambda += lr * (cost - limit) / limit per update")
    ap.add_argument("--lambda-init", type=float, default=0.0)
    ap.add_argument("--lambda-max", type=float, default=20.0)
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=JSON",
                    help="override an EnvConfig field of the preset, e.g. --set w_space=0 (repeatable; ablations)")
    ap.add_argument("--ramp", action="append", default=[], metavar="FIELD=START:F0:F1",
                    help="curriculum: float FIELD is START until fraction F0 of --hours, then linear to the preset's "
                         "value at F1 (repeatable)")
    a = ap.parse_args(argv)
    overrides = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        if k not in EnvConfig.__dataclass_fields__:
            ap.error(f"--set: unknown EnvConfig field {k!r}")
        val = json.loads(v)
        overrides[k] = tuple(val) if isinstance(val, list) else val
    ramps = {}
    for kv in a.ramp:
        k, _, v = kv.partition("=")
        try:
            start, f0, f1 = (float(x) for x in v.split(":"))
        except ValueError:
            ap.error(f"--ramp: expected FIELD=START:F0:F1, got {kv!r}")
        if k not in EnvConfig.__dataclass_fields__ or not isinstance(getattr(EnvConfig, k, None), float):
            ap.error(f"--ramp: {k!r} is not a float EnvConfig field")
        if not 0.0 <= f0 < f1 <= 1.0:
            ap.error("--ramp: need 0 <= F0 < F1 <= 1")
        ramps[k] = (start, f0, f1)

    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    ppo_over = {k: v for k, v in (("bound_coef", a.bound_coef), ("ent_coef", a.ent_coef), ("lr", a.lr)) if v is not None}
    env_cfg, ppo_cfg, hidden = (preset(a.preset, obs_version=a.obs, **overrides), PPOConfig(rollout=a.rollout, **ppo_over),
                                (256, 256))
    if a.cost_limit is not None and not env_cfg.ambient_cost:
        ap.error("--cost-limit needs --set ambient_cost=true")
    env = AvoidEnv(a.envs, dev, seed=a.seed, cfg=replace(env_cfg))  # its own copy: ramps edit env.cfg in place
    obs_dim = env.obs_dim
    model = ActorCritic(obs_dim, ACT_DIM, hidden, squash=a.squash, log_std_max=a.log_std_max).to(dev)
    if a.init_ckpt:
        ck = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)
        if ck["obs_dim"] != obs_dim or ck.get("squash", "tanh") != a.squash:
            raise SystemExit(f"--init-ckpt {a.init_ckpt}: obs_dim/squash {ck['obs_dim']}/{ck.get('squash', 'tanh')} "
                             f"!= {obs_dim}/{a.squash}")
        keep = ("actor.", "log_std", "obs_rms.") + (("critic.",) if a.init_critic else ())
        sd = {k: v for k, v in ck["model"].items() if k.startswith(keep)}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith("critic.") for k in missing), (missing, unexpected)
        print(f"warm start from {a.init_ckpt}: actor + log_std + obs_rms; critic "
              f"{'loaded' if a.init_critic else 'fresh'}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=ppo_cfg.lr, eps=1e-5)
    lagrange = a.cost_limit is not None
    lam = a.lambda_init
    cost_critic = mlp(obs_dim, 1, hidden).to(dev) if lagrange else None
    if lagrange and a.init_ckpt and a.init_critic and "cost_critic" in ck:
        cost_critic.load_state_dict(ck["cost_critic"])
    opt_c = torch.optim.Adam(cost_critic.parameters(), lr=ppo_cfg.lr, eps=1e-5) if lagrange else None
    cval = (lambda x: cost_critic(x).squeeze(-1)) if lagrange else None  # noqa: E731
    extra = lambda: {"cost_critic": cost_critic.state_dict(), "lagrange": {"lambda": lam, "limit": a.cost_limit}} \
        if lagrange else None  # noqa: E731
    (out / "config.json").write_text(json.dumps({
        "args": vars(a), "env_cfg": asdict(env_cfg), "ppo_cfg": asdict(ppo_cfg), "hidden": hidden,
        "obs_version": env_cfg.obs_version, "obs_layout": obs_spec(env_cfg.obs_version)["layout"],
        "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
    }, indent=2))

    def apply_ramps(frac: float) -> None:
        for k, (start, f0, f1) in ramps.items():
            w = min(max((frac - f0) / (f1 - f0), 0.0), 1.0)
            setattr(env.cfg, k, start + (getattr(env_cfg, k) - start) * w)

    N, T = a.envs, ppo_cfg.rollout
    apply_ramps(0.0)  # the first episodes sample the ramps' start values
    obs = env.reset()
    if not a.init_ckpt:  # a warm start keeps the normaliser its actor was trained with
        model.obs_rms.update(obs)
    buf_obs = torch.zeros(T, N, obs_dim, device=dev)
    buf_raw = torch.zeros(T, N, obs_dim, device=dev)
    buf_u = torch.zeros(T, N, ACT_DIM, device=dev)
    buf_logp, buf_val, buf_rew, buf_done = (torch.zeros(T, N, device=dev) for _ in range(4))
    buf_cost, buf_cval, buf_craw = (torch.zeros(T, N, device=dev) for _ in range(3))

    csv_f = open(out / "train.csv", "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS + [f"cfg_{k}" for k in ramps])
    writer.writeheader()
    t_start = time.time()
    last_ckpt, update, env_steps = t_start, 0, 0
    budget = a.hours * 3600.0
    while time.time() - t_start < budget:
        t_up = time.time()
        apply_ramps((t_up - t_start) / budget)
        ep = {k: [] for k in EP_KEYS}
        for t in range(T):
            with torch.no_grad():
                nobs = model.obs_rms(obs)
                d = model.dist(nobs)
                u = d.sample()
                buf_logp[t] = d.log_prob(u).sum(-1)
                buf_val[t] = model.value(nobs)
                if lagrange:
                    buf_cval[t] = cval(nobs)
                if t == 0:
                    mu_vx = d.mean[:, 0]
            buf_obs[t], buf_raw[t], buf_u[t] = nobs, obs, u
            obs, rew, done, info = env.step(model.env_action(u))
            trunc = info["truncated"]
            cost = info["cost"]
            buf_craw[t] = cost
            if trunc.any():  # time limit is not a terminal state: bootstrap from the pre-reset observation
                with torch.no_grad():
                    fobs = model.obs_rms(info["final_obs"])
                    rew = rew + ppo_cfg.gamma * trunc.float() * model.value(fobs)
                    if lagrange:
                        cost = cost + ppo_cfg.gamma * trunc.float() * cval(fobs)
            buf_rew[t], buf_done[t], buf_cost[t] = rew, done.float(), cost
            for k, v in info["episodes"].items():
                ep[k].append(v.float())
        with torch.no_grad():
            last_val = model.value(model.obs_rms(obs))
        adv, ret = gae(buf_rew, buf_val, buf_done, last_val, ppo_cfg.gamma, ppo_cfg.lam)
        frozen = bool(a.init_ckpt) and update < a.critic_warmup
        c_loss = math.nan
        if lagrange:
            with torch.no_grad():
                cadv, cret = gae(buf_cost, buf_cval, buf_done, cval(model.obs_rms(obs)), ppo_cfg.gamma, ppo_cfg.lam)
            adv = (adv - lam * cadv) / (1.0 + lam)  # ppo_update normalises it again
        stats = ppo_update(model, opt, ppo_cfg, buf_obs.view(T * N, -1), buf_u.view(T * N, -1), buf_logp.view(-1),
                           buf_val.view(-1), adv.view(-1), ret.view(-1), train_actor=not frozen)
        if lagrange:  # the cost critic: plain regression on the cost returns, same epochs / minibatches as PPO
            fo, fr, B = buf_obs.view(T * N, -1), cret.view(-1), T * N
            tot = 0.0
            for _ in range(ppo_cfg.epochs):
                perm = torch.randperm(B, device=dev)
                for i in range(ppo_cfg.minibatches):
                    idx = perm[i * (B // ppo_cfg.minibatches):(i + 1) * (B // ppo_cfg.minibatches)]
                    loss_c = ((cval(fo[idx]) - fr[idx]) ** 2).mean()
                    opt_c.zero_grad(set_to_none=True)
                    loss_c.backward()
                    nn.utils.clip_grad_norm_(cost_critic.parameters(), ppo_cfg.max_grad_norm)
                    opt_c.step()
                    tot += loss_c.item()
            c_loss = tot / (ppo_cfg.epochs * ppo_cfg.minibatches)
            if not frozen:  # the multiplier moves with the policy, not while a warm-started actor is held
                jc = float(buf_craw.mean())
                lam = min(max(lam + a.lambda_lr * (jc - a.cost_limit) / a.cost_limit, 0.0), a.lambda_max)
        if not a.init_ckpt:  # warm-started runs keep the BC normaliser fixed (the actor's input scale must not drift)
            model.obs_rms.update(buf_raw)
        with torch.no_grad():
            avx = model.env_action(buf_u[..., 0])  # env-side vx action in [-1, 1]: speed = (avx + 1) / 2 m/s
            vx_mid = float(((avx > -0.8) & (avx < 0.8)).float().mean())  # 0.1 .. 0.9 m/s
            mu_sat = float((mu_vx.abs() > (2.0 if model.squash == "tanh" else 1.0)).float().mean())
        update += 1
        env_steps += T * N
        ep = {k: torch.cat(v) if v else torch.zeros(0, device=dev) for k, v in ep.items()}
        n_ep = int(ep["ret"].numel())
        mean = lambda k: float(ep[k].mean()) if n_ep else math.nan  # noqa: E731
        std = model.log_std.detach().clamp(-5, model.log_std_max).exp().tolist()
        commits = float(ep["commits"].sum()) if n_ep else 0.0
        row = {
            "update": update, "env_steps": env_steps, "wall_s": round(time.time() - t_start, 1),
            "mean_return": round(mean("ret"), 4), "success_rate": round(mean("success"), 4),
            "contact_rate": round(mean("contact"), 4), "wall_rate": round(mean("wall"), 4),
            "timeout_rate": round(mean("timeout"), 4), "near_miss_rate": round(mean("near"), 4),
            "mean_ep_len": round(mean("len"), 1), "episodes": n_ep, "fps": round(T * N / (time.time() - t_up)),
            **{k: round(v, 5) for k, v in stats.items()},
            "std_vx": round(std[0], 4), "std_vy": round(std[1], 4), "std_wz": round(std[2], 4),
            "lock_rate": round(float((ep["locks"] > 0).float().mean()), 4) if n_ep else math.nan,
            "locks_per_ep": round(mean("locks"), 4), "commits_per_ep": round(mean("commits"), 4),
            "hit_per_commit": round(float(ep["hits"].sum()) / commits, 4) if commits else math.nan,
            "space_frac": round(mean("space_frac"), 4), **{f"r_{k}": round(mean(f"r_{k}"), 4) for k in REWARD_TERMS},
            "ttc_frac": round(mean("ttc_frac"), 4), "mu_vx_mean": round(float(mu_vx.mean()), 3),
            "mu_vx_sat": round(mu_sat, 4), "vx_mid_frac": round(vx_mid, 4), "actor_frozen": int(frozen),
            "ambient_contact_rate": round(mean("ambient_contact"), 4), "cost_step": round(float(buf_craw.mean()), 5),
            "lambda": round(lam, 4), "c_loss": round(c_loss, 5),
            **{f"cfg_{k}": round(getattr(env.cfg, k), 5) for k in ramps},
        }
        writer.writerow(row)
        csv_f.flush()
        if update % a.print_every == 0 or update == 1:
            print(" ".join(f"{k}={row[k]}" for k in ("update", "env_steps", "mean_return", "success_rate",
                                                     "contact_rate", "near_miss_rate", "lock_rate", "hit_per_commit",
                                                     "mean_ep_len", "r_space", "space_frac", "std_vx", "vx_mid_frac",
                                                     "mu_vx_sat", "ambient_contact_rate", "cost_step", "lambda",
                                                     "fps")), flush=True)
        if time.time() - last_ckpt >= a.ckpt_min * 60:
            p = out / f"ckpt_{update:05d}.pt"
            save_ckpt(p, model, opt, update, env_steps, env_cfg, ppo_cfg, hidden, extra())
            save_ckpt(out / "latest.pt", model, opt, update, env_steps, env_cfg, ppo_cfg, hidden, extra())
            last_ckpt = time.time()
            print(f"saved {p}", flush=True)
    p = out / f"ckpt_{update:05d}.pt"
    save_ckpt(p, model, opt, update, env_steps, env_cfg, ppo_cfg, hidden, extra())
    save_ckpt(out / "latest.pt", model, opt, update, env_steps, env_cfg, ppo_cfg, hidden, extra())
    csv_f.close()
    print(f"done: {update} updates, {env_steps} env steps, {time.time() - t_start:.0f} s; saved {p}", flush=True)
    (out / "DONE").write_text(json.dumps({"updates": update, "env_steps": env_steps, "final": str(p)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
