"""Minimal PPO for the avoidance env: separate actor / critic MLPs, tanh-squashed Gaussian policy, running obs
normalisation (stored in checkpoints), GAE, clipped surrogate, plain MSE value loss, KL early stop.

The squash: the stored action is the pre-tanh Gaussian sample u; the env receives tanh(u). PPO's probability ratio
is computed on u, where the tanh Jacobian cancels exactly, so no correction term is needed.

squash="clip" (D46, RL v4 arm A1) is the rsl_rl / rl_games parameterisation instead: the env receives clamp(u, -1, 1)
and deployment uses clamp(mean). The ratio and the Gaussian entropy are unchanged (the clamp is outside the policy,
like the tanh). PPOConfig.bound_coef adds rl_games' bounds loss, coef * relu(|mean| - 1)^2, so the mean cannot run
deep past the action limits the way the tanh mean did (median pre-tanh vx mean +11 in every v1-v3 run, D45).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-4, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(eps, dtype=torch.float64))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).double()
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = bm - self.mean, self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self.mean.to(x.dtype), torch.sqrt(self.var.to(x.dtype) + 1e-8)
        return ((x - m) / s).clamp(-self.clip, self.clip)


def mlp(inp: int, out: int, hidden=(256, 256), act=nn.ELU) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), init_log_std: float = -0.5,
                 squash: str = "tanh", log_std_max: float = 1.0):
        super().__init__()
        if squash not in ("tanh", "clip"):
            raise ValueError(f"squash must be 'tanh' or 'clip', got {squash!r}")
        self.squash, self.log_std_max = squash, float(log_std_max)
        self.obs_rms = RunningMeanStd(obs_dim)
        self.actor = mlp(obs_dim, act_dim, hidden)
        self.critic = mlp(obs_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def dist(self, nobs: torch.Tensor) -> torch.distributions.Normal:
        mu = self.actor(nobs)
        std = self.log_std.clamp(-5.0, self.log_std_max).exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def value(self, nobs: torch.Tensor) -> torch.Tensor:
        return self.critic(nobs).squeeze(-1)

    def env_action(self, u: torch.Tensor) -> torch.Tensor:
        """Policy sample (or mean) -> the action the env receives, in [-1, 1]."""
        return torch.tanh(u) if self.squash == "tanh" else u.clamp(-1.0, 1.0)

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Raw obs -> deterministic action in [-1, 1] (tanh or clamp of the mean)."""
        return self.env_action(self.actor(self.obs_rms(obs)))


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.003
    max_grad_norm: float = 1.0
    target_kl: float = 0.03  # stop the epoch loop early above this
    rollout: int = 32
    bound_coef: float = 0.0  # rl_games bounds loss on the Gaussian mean beyond +-1 (0 = off, the v1-v3 setting)


def gae(rew, val, done, last_val, gamma, lam):
    """rew/val/done [T, N] (done = episode ended AFTER this step), last_val [N] -> (advantages, returns)."""
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    last = torch.zeros_like(last_val)
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def ppo_update(model: ActorCritic, opt: torch.optim.Optimizer, cfg: PPOConfig, nobs, u, old_logp, old_val, adv, ret,
               train_actor: bool = True):
    """All inputs flattened to [B, ...]. Returns a dict of mean losses / stats. train_actor=False trains the critic
    only (a warm-started actor is left untouched while its fresh critic catches up)."""
    B = nobs.shape[0]
    mb = B // cfg.minibatches
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0, "n": 0}
    for _ in range(cfg.epochs):
        perm = torch.randperm(B, device=nobs.device)
        kls = []
        for i in range(cfg.minibatches):
            idx = perm[i * mb:(i + 1) * mb]
            d = model.dist(nobs[idx])
            logp = d.log_prob(u[idx]).sum(-1)
            ratio = torch.exp(logp - old_logp[idx])
            a = adv[idx]
            pi_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
            v = model.value(nobs[idx])
            v_loss = ((v - ret[idx]) ** 2).mean()  # unclipped: returns are O(10), a 0.2 value clip would stall it
            ent = d.entropy().sum(-1).mean()
            loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
            if cfg.bound_coef:
                loss = loss + cfg.bound_coef * ((d.mean.abs() - 1.0).clamp_min(0.0) ** 2).sum(-1).mean()
            if not train_actor:
                loss = cfg.vf_coef * v_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # clip actor and critic separately so large value gradients don't shrink the policy step
            nn.utils.clip_grad_norm_(list(model.actor.parameters()) + [model.log_std], cfg.max_grad_norm)
            nn.utils.clip_grad_norm_(model.critic.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                kls.append(kl)
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["kl"] += kl
                stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                stats["n"] += 1
        if sum(kls) / len(kls) > cfg.target_kl:
            break
    n = max(stats.pop("n"), 1)
    return {k: v / n for k, v in stats.items()}
