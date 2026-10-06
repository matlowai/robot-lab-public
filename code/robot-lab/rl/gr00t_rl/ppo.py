"""PPO for the residual actor (pure torch; style after patrol-lab benchmarks/avoidance/rl/ppo.py).

Action per RL step (one executed GR00T chunk = 8 ticks):
  * arm residual: 5-d Gaussian sample u (pre-tanh); the env receives ARM_SCALE_DEG * tanh(u) degrees added to every
    executed arm target of the chunk. PPO's ratio is computed on u (the tanh is outside the policy, Jacobian cancels).
  * release gate: Bernoulli(sigmoid(logit)); 1 = command the jaw open for this chunk (and hold it open for the next
    chunk too, see isaac_env.RELEASE_HOLD_CHUNKS).
Deterministic deployment: arm = ARM_SCALE_DEG * tanh(mean), release = sigmoid(logit) > 0.5. With the zero-initialised
output layer and the negative release bias, the deterministic policy at step 0 IS the base GR00T policy (tested).

Asymmetric actor-critic: the actor sees only deployable inputs (proprioception, GR00T's planned chunk, GR00T's pooled
image embedding, the task's object id, time); the critic additionally sees privileged simulator state.
"""

from __future__ import annotations

import math
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


class Encoder(nn.Module):
    """Normalised low-dim obs ++ a learned 2048->emb_out projection of GR00T's pooled image embedding."""

    def __init__(self, low_dim: int, emb_dim: int, emb_out: int = 64):
        super().__init__()
        self.low_rms = RunningMeanStd(low_dim)
        self.emb_rms = RunningMeanStd(emb_dim) if emb_dim else None
        self.emb_proj = nn.Sequential(nn.Linear(emb_dim, emb_out), nn.ELU()) if emb_dim else None
        self.out_dim = low_dim + (emb_out if emb_dim else 0)

    def forward(self, low, emb=None):
        x = self.low_rms(low)
        if self.emb_proj is not None:
            x = torch.cat([x, self.emb_proj(self.emb_rms(emb))], -1)
        return x

    @torch.no_grad()
    def update(self, low, emb=None):
        self.low_rms.update(low)
        if self.emb_rms is not None:
            self.emb_rms.update(emb)


class ResidualActorCritic(nn.Module):
    def __init__(self, actor_dim: int, critic_dim: int, emb_dim: int, arm_dim: int = 5, hidden=(256, 256),
                 init_log_std: float = math.log(0.3), release_bias: float = -3.0, critic_uses_emb: bool = True):
        super().__init__()
        self.arm_dim = arm_dim
        self.actor_enc = Encoder(actor_dim, emb_dim)
        self.critic_enc = Encoder(critic_dim, emb_dim if critic_uses_emb else 0)
        self.actor = mlp(self.actor_enc.out_dim, arm_dim + 1, hidden)
        self.critic = mlp(self.critic_enc.out_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.full((arm_dim,), float(init_log_std)))
        with torch.no_grad():  # start exactly at the base policy (deterministic) with small exploration
            self.actor[-1].weight.zero_()
            self.actor[-1].bias.zero_()
            self.actor[-1].bias[arm_dim] = release_bias

    def dist(self, a_low, emb):
        out = self.actor(self.actor_enc(a_low, emb))
        mean, logit = out[..., : self.arm_dim], out[..., self.arm_dim]
        std = self.log_std.clamp(-5.0, 0.5).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), torch.distributions.Bernoulli(logits=logit)

    def value(self, c_low, emb):
        return self.critic(self.critic_enc(c_low, emb)).squeeze(-1)

    @torch.no_grad()
    def act(self, a_low, emb, c_low, deterministic: bool = False):
        g, b = self.dist(a_low, emb)
        if deterministic:
            u, rel = g.mean, (b.probs > 0.5).float()
        else:
            u, rel = g.sample(), b.sample()
        logp = g.log_prob(u).sum(-1) + b.log_prob(rel)
        return u, rel, logp, self.value(c_low, emb), b.probs


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 5
    minibatches: int = 4
    ent_coef: float = 0.003
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    bound_coef: float = 0.0


def gae(rew, val, done, last_val, gamma, lam):
    """rew/val/done: (T, N); done[t] = 1 if the episode ended AT step t (no bootstrap past it)."""
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


def ppo_update(ac: ResidualActorCritic, opt, batch: dict, cfg: PPOConfig) -> dict:
    """batch: flat tensors a_low, c_low, emb, u, rel, logp, adv, ret (only valid transitions)."""
    n = batch["adv"].shape[0]
    adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0, "updates": 0}
    stop = False
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, device=adv.device)
        for idx in perm.chunk(cfg.minibatches):
            emb = None if batch["emb"] is None else batch["emb"][idx]
            g, b = ac.dist(batch["a_low"][idx], emb)
            logp = g.log_prob(batch["u"][idx]).sum(-1) + b.log_prob(batch["rel"][idx])
            ratio = (logp - batch["logp"][idx]).exp()
            a = adv[idx]
            pi_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
            v = ac.value(batch["c_low"][idx], emb)
            v_loss = (v - batch["ret"][idx]).pow(2).mean()
            ent = g.entropy().sum(-1).mean() + b.entropy().mean()
            loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
            if cfg.bound_coef:
                loss = loss + cfg.bound_coef * torch.relu(g.mean.abs() - 2.0).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                kl = (batch["logp"][idx] - logp).mean().item()
                stats["pi_loss"] += pi_loss.item(); stats["v_loss"] += v_loss.item(); stats["entropy"] += ent.item()
                stats["kl"] += kl; stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                stats["updates"] += 1
            if kl > 1.5 * cfg.target_kl:
                stop = True
                break
        if stop:
            break
    k = max(1, stats["updates"])
    return {key: (v / k if key != "updates" else v) for key, v in stats.items()} | {"early_stop": stop}
