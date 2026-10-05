"""Torch port of the closed-loop people in benchmarks/avoidance/pedestrians.py (hunters and wanderers), batched over
N envs x H person slots, for the RL env (benchmarks/avoidance/rl/env.py).

The state machine is the reference's, step for step (WALK / PAUSE milling about a post, LOCKED with re-aim at the
constant-velocity intercept point, COMMITTED straight line for dist + overshoot_m, COOLDOWN walking back to the
post; vision cone only while walking, line of sight blocked by walls, lock timeout / lose-sight, max_locks).
tests/test_rl_hunters.py drives this class and a pedestrians.LiveCrowd from the same state with the same robot
trajectory and requires the same positions and states every step.

What differs from the reference is what surrounds the state machine:
  * geometry: the env's active wall segments stand in for the compound's buildings: they block line of sight
    and refuse a hunter's step that would cross them (as pedestrians._advance does). There is no site boundary
    (the reference also refuses steps that leave the site) and no restricted zone.
  * random choices (new destinations, pause lengths, initial placement) come from the torch generator: K
    candidates at once, the first usable one wins, else the last (the reference retries up to 50 times).
    "Usable" = farther than building_margin_m from every wall, more than 2 m away, and the straight path crosses
    no wall (the reference also samples points along the path against the building margin). Wanderers roam a
    disc around the leg instead of the whole yard. So RNG-driven choices never match the reference sample for
    sample; the parity test replaces them with the same deterministic choices on both sides.
  * robot history: the last HIST = 6 positions (the 0.5 s window at 10 Hz, all the reference's velocity estimate
    reads; the reference keeps 20 and searches them).
Times are float64 (env step / 10, the nearest double to k/10, which is what the reference's round(t + dt, 6)
produces), so every timer comparison is bit-identical to the reference's. Positions follow the env dtype.
"""

from __future__ import annotations

import math

import torch

from benchmarks.avoidance.pedestrians import (COMMITTED, COOLDOWN, DT, HUNTER, LOCKED, PAUSE, PERSON_R, WALK, WANDERER,
                                              CrowdConfig)

HIST = 6  # robot positions t-0.5 .. t
NEVER = -1e9
FAR = 1.0e3  # where inactive slots park (they are masked out anyway)
K_TRIES = 8
RELEASE_REASONS = ("passed", "timeout", "lost_sight")


def seg_blocked(p, q, segs, sact):
    """Batched pedestrians.segment_blocked: p, q [..., 2]; segs [..., W, 4], sact [..., W] (broadcastable against
    p[..., None]) -> [...] bool, True when p->q crosses or touches an active segment."""
    shape = torch.broadcast_shapes(p.shape[:-1], q.shape[:-1])
    if segs.shape[-2] == 0:
        return torch.zeros(shape, dtype=torch.bool, device=p.device)
    px, py = p[..., 0:1], p[..., 1:2]
    rx, ry = q[..., 0:1] - px, q[..., 1:2] - py
    ax, ay, bx, by = segs.unbind(-1)
    sx, sy = bx - ax, by - ay
    den = rx * sy - ry * sx
    good = den.abs() > 1e-12
    sden = torch.where(good, den, torch.ones_like(den))
    t = ((ax - px) * sy - (ay - py) * sx) / sden
    u = ((ax - px) * ry - (ay - py) * rx) / sden
    hit = good & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1) & sact
    return hit.any(-1)


def wall_dist(p, segs, sact):
    """p [..., 2], segs [..., W, 4], sact [..., W] -> [...] distance to the nearest active segment (inf if none)."""
    if segs.shape[-2] == 0:
        return torch.full(p.shape[:-1], float("inf"), dtype=p.dtype, device=p.device)
    ax, ay, bx, by = segs.unbind(-1)
    ex, ey = bx - ax, by - ay
    px, py = p[..., 0:1], p[..., 1:2]
    t = (((px - ax) * ex + (py - ay) * ey) / (ex * ex + ey * ey).clamp_min(1e-12)).clamp(0, 1)
    d = torch.hypot(px - (ax + t * ex), py - (ay + t * ey))
    return torch.where(sact, d, torch.full_like(d, float("inf"))).amin(-1)


def intercept_point(hunter, speed, robot, robot_vel, cap_s: float):
    """Batched pedestrians.intercept_point: hunter/robot/robot_vel [..., 2], speed [...] -> aim point [..., 2]."""
    px, py = robot[..., 0] - hunter[..., 0], robot[..., 1] - hunter[..., 1]
    vx, vy = robot_vel[..., 0], robot_vel[..., 1]
    a = vx * vx + vy * vy - speed * speed
    b = 2 * (px * vx + py * vy)
    c = px * px + py * py
    inf = torch.full_like(a, float("inf"))
    lin = a.abs() < 1e-9
    tau_lin = torch.where(b < 0, -c / torch.where(b < 0, b, torch.ones_like(b)), inf)
    disc = b * b - 4 * a * c
    r = torch.sqrt(disc.clamp_min(0.0))
    a2 = 2 * torch.where(lin, torch.ones_like(a), a)
    r1, r2 = (-b - r) / a2, (-b + r) / a2
    tau_q = torch.minimum(torch.where(r1 > 0, r1, inf), torch.where(r2 > 0, r2, inf))
    tau_q = torch.where(disc >= 0, tau_q, inf)
    tau = torch.where(lin, tau_lin, tau_q)
    tau = torch.where(torch.isinf(tau), torch.zeros_like(tau), tau).clamp(max=cap_s)  # no solution: the robot itself
    return torch.stack([robot[..., 0] + vx * tau, robot[..., 1] + vy * tau], -1)


def _wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


class TorchCrowd:
    """H stateful people per env. Slot kinds are fixed per crowd (``kind``); ``active`` says who is in the scene.

    Per-person tensors mirror LiveCrowd's arrays: pos, heading, speed, state, target, until, post, commit_m,
    locks_used, lock_t, last_aim_t, last_seen_t, travel_left. Added: roam_r (radius of the milling / roaming
    disc around post; the reference uses post_radius_m for hunters and the whole yard for wanderers), lock_hit
    (the current lock has made contact), and per-env episode counters n_locks, n_commits, n_hits, n_released[3]."""

    def __init__(self, n: int, h: int, cfg: CrowdConfig, device, dtype=torch.float32, gen: torch.Generator | None = None,
                 robot_r: float = 0.55):
        self.n, self.h, self.cfg, self.device, self.dtype, self.robot_r = n, h, cfg, torch.device(device), dtype, robot_r
        self.gen = gen
        f, d64 = dict(device=self.device, dtype=dtype), dict(device=self.device, dtype=torch.float64)
        lng = dict(device=self.device, dtype=torch.long)
        self.kind = torch.full((n, h), HUNTER, **lng)
        self.active = torch.zeros(n, h, dtype=torch.bool, device=self.device)
        self.pos, self.target, self.post = torch.full((n, h, 2), FAR, **f), torch.zeros(n, h, 2, **f), torch.zeros(n, h, 2, **f)
        self.heading, self.speed, self.roam_r = torch.zeros(n, h, **f), torch.zeros(n, h, **f), torch.zeros(n, h, **f)
        self.commit_m, self.travel_left = torch.zeros(n, h, **f), torch.zeros(n, h, **f)
        self.state, self.locks_used = torch.full((n, h), WALK, **lng), torch.zeros(n, h, **lng)
        self.until = torch.zeros(n, h, **d64)
        self.lock_t, self.last_aim_t, self.last_seen_t = (torch.full((n, h), NEVER, **d64) for _ in range(3))
        self.lock_hit = torch.zeros(n, h, dtype=torch.bool, device=self.device)
        self.hist, self.hist_t, self.hist_n = torch.zeros(n, HIST, 2, **f), torch.zeros(n, HIST, **d64), torch.zeros(n, **lng)
        self.n_locks, self.n_commits, self.n_hits = (torch.zeros(n, **lng) for _ in range(3))
        self.n_released = torch.zeros(n, len(RELEASE_REASONS), **lng)
        self.half_fov = math.radians(cfg.vision_fov_deg) / 2

    # -- randomness (overridden by the parity test with deterministic choices) --------------------------------------
    def _u(self, *shape, lo=0.0, hi=1.0, dtype=None):
        return lo + (hi - lo) * torch.rand(*shape, generator=self.gen, device=self.device, dtype=dtype or self.dtype)

    def _sample_destinations(self, mask, segs, sact):
        """[n, h, 2] new destinations (only rows under mask are used): K candidates in the roam disc around post,
        the first usable one (> 2 m away, clear of walls, straight path not crossing one), else the last."""
        n, h, K = self.n, self.h, K_TRIES
        r = self.roam_r[..., None] * torch.sqrt(self._u(n, h, K))
        a = self._u(n, h, K, lo=-math.pi, hi=math.pi)
        cand = self.post[:, :, None, :] + torch.stack([r * torch.cos(a), r * torch.sin(a)], -1)  # [n,h,K,2]
        p = self.pos[:, :, None, :]
        s4, sa = segs[:, None, None], sact[:, None, None]
        ok = (torch.hypot(*(cand - p).unbind(-1)) > 2.0) & ~seg_blocked(p, cand, s4, sa)
        ok &= wall_dist(cand, s4, sa) > self.cfg.building_margin_m
        q = _pick_first(cand, ok)
        return torch.where(ok.any(-1)[..., None], q, self.pos)  # nowhere reachable: stay put, then pause

    def _sample_pause(self, mask):
        """[n, h] float64 pause lengths (only rows under mask are used)."""
        lo, hi = self.cfg.pause_s
        return self._u(self.n, self.h, lo=lo, hi=hi, dtype=torch.float64)

    def _new_destination(self, mask, segs, sact) -> None:
        if not bool(mask.any()):
            return
        q = self._sample_destinations(mask, segs, sact)
        self.target = torch.where(mask[..., None], q, self.target)
        d = q - self.pos
        self.heading = torch.where(mask, torch.atan2(d[..., 1], d[..., 0]), self.heading)
        self.state = torch.where(mask, WALK, self.state)

    # -- perception ------------------------------------------------------------------------------------------------
    def robot_velocity(self):
        """LiveCrowd._robot_velocity, batched: displacement over the last <= 0.5 s of recorded robot positions."""
        k = self.hist_n.clamp(max=HIST)
        i0 = (HIST - k).clamp(max=HIST - 1)
        ar = torch.arange(self.n, device=self.device)
        x0, t0 = self.hist[ar, i0], self.hist_t[ar, i0]
        dt = self.hist_t[:, -1] - t0
        ok = (k >= 2) & (dt > 1e-9)
        v = (self.hist[:, -1] - x0) / torch.where(ok, dt, torch.ones_like(dt)).to(self.dtype)[:, None]
        return torch.where(ok[:, None], v, torch.zeros_like(v))

    # -- stepping --------------------------------------------------------------------------------------------------
    def step(self, robot, t, segs, sact) -> None:
        """Advance every active person one DT to time t [n] (float64) given the robot position [n, 2] at t and the
        walls segs [n, W, 4] / sact [n, W]. One LiveCrowd._step_once per env."""
        cfg = self.cfg
        robot = robot.to(self.dtype)
        self.hist = torch.cat([self.hist[:, 1:], robot[:, None]], 1)
        self.hist_t = torch.cat([self.hist_t[:, 1:], t[:, None]], 1)
        self.hist_n = (self.hist_n + 1).clamp(max=HIST)
        rvel = self.robot_velocity()
        tt = t[:, None]  # [n,1] float64
        sb, sab = segs[:, None], sact[:, None]
        act = self.active
        S = self.state.clone()
        hunter = act & (self.kind == HUNTER)

        # hunters not engaged: a finished cooldown sends them back to milling about, then they may lock on
        cool_over = hunter & (S == COOLDOWN) & (tt >= self.until)
        self._new_destination(cool_over, segs, sact)
        S = torch.where(cool_over, WALK, S)
        rp = robot[:, None, :].expand_as(self.pos)
        to_r = rp - self.pos
        d = torch.hypot(to_r[..., 0], to_r[..., 1])
        in_range = d <= cfg.trigger_m
        clear_los = ~seg_blocked(self.pos, rp, sb, sab)
        off = _wrap(torch.atan2(to_r[..., 1], to_r[..., 0]) - self.heading).abs()
        sees = in_range & ((S != WALK) | (off <= self.half_fov)) & clear_los  # walking: cone; paused: all around
        lock = hunter & ((S == WALK) | (S == PAUSE)) & (self.locks_used < cfg.max_locks) & sees
        S = torch.where(lock, LOCKED, S)
        self.locks_used = self.locks_used + lock.long()
        self.lock_t = torch.where(lock, tt, self.lock_t)
        self.last_seen_t = torch.where(lock, tt, self.last_seen_t)
        self.last_aim_t = torch.where(lock, torch.full_like(self.last_aim_t, NEVER), self.last_aim_t)
        self.lock_hit &= ~lock
        self.n_locks += lock.sum(-1)

        # LOCKED: eyes on the robot (no cone); timeout / lost sight -> cooldown; commit inside commit_m; else re-aim
        L = act & (S == LOCKED)
        self.last_seen_t = torch.where(L & in_range & clear_los, tt, self.last_seen_t)
        timeout = (tt - self.lock_t) > cfg.lock_timeout_s
        lost = (tt - self.last_seen_t) > cfg.lose_sight_s
        rel_lock = L & (timeout | lost)
        Lc = L & ~rel_lock
        commit = Lc & (d <= self.commit_m)
        self.travel_left = torch.where(commit, d + cfg.overshoot_m, self.travel_left)
        self.n_commits += commit.sum(-1)
        reaim = Lc & (commit | ((tt - self.last_aim_t) >= cfg.reaim_s - 1e-9))  # the commit line is a fresh aim
        aim = intercept_point(self.pos, self.speed, rp, rvel[:, None, :].expand_as(self.pos), cfg.predict_cap_s)
        self.heading = torch.where(reaim, torch.atan2(aim[..., 1] - self.pos[..., 1], aim[..., 0] - self.pos[..., 0]),
                                   self.heading)
        self.last_aim_t = torch.where(reaim, tt, self.last_aim_t)

        # COMMITTED: straight on; COOLDOWN (not over): back toward the post at 0.7 speed
        C = act & (S == COMMITTED)
        Cd = act & (S == COOLDOWN)
        to_post = self.post - self.pos
        dpost = torch.hypot(to_post[..., 0], to_post[..., 1])
        home = Cd & (dpost > 0.5)
        self.heading = torch.where(home, torch.atan2(to_post[..., 1], to_post[..., 0]), self.heading)
        stride = self.speed * DT
        step_len = torch.where(home, torch.minimum(dpost, self.speed * 0.7 * DT), stride)
        self._advance(Lc | C | home, step_len, sb, sab)
        self.travel_left = torch.where(C | commit, self.travel_left - stride, self.travel_left)
        passed = C & (self.travel_left <= 0)

        # PAUSE over -> new destination (no step this tick); WALK straight at the target, pause on arrival
        P = act & (S == PAUSE)
        self._new_destination(P & (tt >= self.until), segs, sact)
        Wk = act & (S == WALK)
        to_t = self.target - self.pos
        dt_ = torch.hypot(to_t[..., 0], to_t[..., 1])
        arrive = Wk & (dt_ <= stride)
        moved = self.pos + to_t / dt_.clamp_min(1e-12)[..., None] * stride[..., None]
        self.pos = torch.where(arrive[..., None], self.target, torch.where((Wk & ~arrive)[..., None], moved, self.pos))
        pause = self._sample_pause(arrive)
        self.until = torch.where(arrive, tt + pause, self.until)

        # state writes (branches dispatched on S, so nothing cascades within a tick, as in the reference)
        state = torch.where(lock, LOCKED, self.state)
        state = torch.where(commit, COMMITTED, state)
        state = torch.where(arrive, PAUSE, state)
        released = rel_lock | passed
        state = torch.where(released, COOLDOWN, state)
        self.state = state
        self.until = torch.where(released, tt + cfg.cooldown_s, self.until)
        self.n_released[:, 0] += passed.sum(-1)
        self.n_released[:, 1] += (rel_lock & timeout).sum(-1)
        self.n_released[:, 2] += (rel_lock & ~timeout).sum(-1)

        # a hit = a lock (locked or committed) during which the robot and the hunter touched
        opened = act & ((state == LOCKED) | (state == COMMITTED) | released)  # the release tick still counts
        dr = self.pos - rp
        clear = torch.hypot(dr[..., 0], dr[..., 1]) - PERSON_R - self.robot_r
        new_hit = opened & (clear < 0) & ~self.lock_hit
        self.lock_hit |= new_hit
        self.n_hits += new_hit.sum(-1)

    def _advance(self, mask, step_len, sb, sab) -> None:
        """pedestrians._advance: step along the heading unless that crosses a wall (then stay)."""
        q = self.pos + step_len[..., None] * torch.stack([torch.cos(self.heading), torch.sin(self.heading)], -1)
        ok = mask & ~seg_blocked(self.pos, q, sb, sab)
        self.pos = torch.where(ok[..., None], q, self.pos)

    def open_lock(self):
        """[n, h] bool: people currently locked on or committed."""
        return self.active & ((self.state == LOCKED) | (self.state == COMMITTED))

    # -- episode placement -----------------------------------------------------------------------------------------
    def reset_leg(self, ids, D, u, nrm, segs, sact, hunter_on, wander_on, n_hunter_slots: int,
                  wanderer_start_min_m: float = 6.0) -> None:
        """Place a fresh cast for envs ids on a leg from the origin to D * u (nrm = left normal). Slots
        [0, n_hunter_slots) are hunters posted beside the leg (post_offset_m to either side, 25-75 % along; the
        reference's _post_beside), the rest wanderers roaming a disc around the leg. Hunters start > 6 m from the
        robot, wanderers > wanderer_start_min_m (default 6 m), all clear of walls, either walking to a fresh
        destination or part-way through a pause."""
        cfg, m, h, K = self.cfg, len(ids), self.h, K_TRIES
        if m == 0 or h == 0:
            return
        f = dict(device=self.device, dtype=self.dtype)
        D, u, nrm = D.to(self.dtype), u.to(self.dtype), nrm.to(self.dtype)
        is_h = (torch.arange(h, device=self.device) < n_hunter_slots)[None, :].expand(m, h)
        sg, sa = segs[ids][:, None, None], sact[ids][:, None, None]
        # posts
        s = self._u(m, h, K, lo=0.25, hi=0.75) * D[:, None, None]
        side = torch.where(self._u(m, h, K) < 0.5, -1.0, 1.0).to(self.dtype)
        off = side * self._u(m, h, K, lo=cfg.post_offset_m[0], hi=cfg.post_offset_m[1])
        cand = u[:, None, None, :] * s[..., None] + nrm[:, None, None, :] * off[..., None]
        ok = (torch.hypot(*cand.unbind(-1)) > 6.0) & (wall_dist(cand, sg, sa) > cfg.building_margin_m)
        post_h = _pick_first(cand, ok)
        mid = (u * (0.5 * D)[:, None])[:, None, :].expand(m, h, 2)
        post = torch.where(is_h[..., None], post_h, mid)
        roam = torch.where(is_h, torch.full((m, h), cfg.post_radius_m, **f), (0.5 * D + 4.0)[:, None].expand(m, h))
        # start positions
        r = roam[..., None] * torch.sqrt(self._u(m, h, K))
        a = self._u(m, h, K, lo=-math.pi, hi=math.pi)
        cand = post[:, :, None, :] + torch.stack([r * torch.cos(a), r * torch.sin(a)], -1)
        min_start = torch.where(is_h, 6.0, wanderer_start_min_m)[..., None]
        ok = (torch.hypot(*cand.unbind(-1)) > min_start) & (wall_dist(cand, sg, sa) > cfg.building_margin_m)
        pos = _pick_first(cand, ok)
        hs, ws = self._u(m, h, lo=cfg.hunter_speed[0], hi=cfg.hunter_speed[1]), self._u(m, h, lo=cfg.walk_speed[0], hi=cfg.walk_speed[1])
        active = torch.where(is_h, hunter_on, wander_on)
        self.kind[ids] = torch.where(is_h, HUNTER, WANDERER)
        self.active[ids] = active
        self.post[ids], self.roam_r[ids] = post, roam
        self.pos[ids] = torch.where(active[..., None], pos, torch.full_like(pos, FAR))
        self.speed[ids] = torch.where(is_h, hs, ws)
        self.commit_m[ids] = self._u(m, h, lo=cfg.commit_m[0], hi=cfg.commit_m[1])
        self.state[ids] = WALK
        self.locks_used[ids] = 0
        self.lock_t[ids], self.last_aim_t[ids], self.last_seen_t[ids] = NEVER, NEVER, NEVER
        self.travel_left[ids] = 0.0
        self.lock_hit[ids] = False
        self.hist_n[ids] = 0
        self.n_locks[ids], self.n_commits[ids], self.n_hits[ids], self.n_released[ids] = 0, 0, 0, 0
        pausing = self._u(m, h) < 0.5
        until = self._u(m, h, lo=0.0, hi=cfg.pause_s[1], dtype=torch.float64)
        mask = torch.zeros(self.n, h, dtype=torch.bool, device=self.device)
        mask[ids] = active
        self._new_destination(mask, segs, sact)  # sets target, heading, WALK
        self.state[ids] = torch.where(active & pausing, PAUSE, self.state[ids])
        self.until[ids] = torch.where(pausing, until, torch.zeros_like(until))

    def people(self):
        """(positions [n, h, 2], active [n, h]) for the lidar and the clearances."""
        return self.pos, self.active


def _pick_first(cand, ok):
    """cand [..., K, 2], ok [..., K] -> [..., 2]: the first ok candidate, else the last one."""
    K = cand.shape[-2]
    idx = torch.where(ok.any(-1), ok.float().argmax(-1), torch.full(ok.shape[:-1], K - 1, device=ok.device))
    return torch.gather(cand, -2, idx[..., None, None].expand(*idx.shape, 1, 2)).squeeze(-2)
