"""Vectorised 2-D avoidance env for PPO: N local avoidance segments stepped in parallel on one torch device.

One episode = one patrol leg's local problem: the robot starts at the origin with a random heading error, the goal
is 12-25 m away in a random direction, and 0-4 "book readers" (people who walk straight lines and never react)
are placed with the same event types as benchmarks/avoidance/crowd.py (head-on, side-by-side pair head-on,
standing on the line, perpendicular crossing timed to meet the robot) plus the odd random slow walker. 0-2 wall
segments (corridor wall, box edge, fence behind the goal) teach the policy that static geometry is also lidar.

Closed-loop people (presets "base" / "hard", see ``preset``): on top of that ambient traffic, hunter_slots hunters
posted beside the leg and wanderer_slots wanderers, stepped by rl/hunters.py::TorchCrowd with the semantics of
benchmarks/avoidance/pedestrians.py (the model sim2d's and Isaac's live crowds use): a hunter that sees the robot
locks on, walks to intercept, commits inside commit_m and walks a straight line past. Hunter parameters come from
pedestrians.CrowdConfig + crowd.LIVE_TIERS[crowd_tier]. Finished episodes report locks / commits / hits.
The default EnvConfig ("legacy") has no closed-loop people and samples exactly as before. Preset "hard-dense"
(D44) adds wanderers until the density of people near the robot matches the sim2d hard tier (see PRESETS) and a
personal-space penalty (``personal_space_penalty``); finished episodes also report each reward term's sum
(REWARD_TERMS, as "r_<term>") and space_frac, the share of steps with someone inside the personal-space radius.

Reward per step: progress (w_progress per metre closed) - time_penalty - w_smooth * ||a_t - a_{t-1}||^2
- w_near * near-miss depth (linear, clearance near_margin_m -> 0) - personal space (quadratic, clearance
space_radius_m -> 0; off unless set) - contact_penalty / wall_penalty (terminal) + success_bonus (terminal).

Everything that the deployed controller must reproduce lives in module-level functions that accept either a
torch tensor or a numpy array (``sectorize``, ``goal_features``, ``build_obs``, ``scale_action``), so training and
deployment compute the same features from the same code.

Robot model = sim2d's exactly: body-frame (vx, vy, wz) commands, per-axis acceleration limits ACC_VX/ACC_VY/ACC_WZ,
dt = 0.1 s, yaw integrated first and position with the new yaw. Lidar = sim2d's exactly: 181 rays over 270 deg
(-135..+135 relative to heading), 12 m max range, hits closer than 0.65 m ignored, first root only for circles.

Observation layout (OBS_VERSION = "avoid-v1", OBS_DIM = 134, float32):
    [0:64)    scan_now   min range per angular sector / 12 m, sector 0 = rightmost (-135 deg), 63 = leftmost
    [64:128)  scan_prev  the same features one control step (0.1 s) earlier (= scan_now on the first step)
    [128]     goal_dist  min(distance to goal, 10 m) / 10
    [129]     goal_cos   cos(bearing of goal in the robot frame)
    [130]     goal_sin   sin(bearing of goal in the robot frame)   (bearing > 0 = goal to the left)
    [131:134) vel        current body velocity (vx m/s, vy m/s, wz rad/s), i.e. sim2d's internal state
Sector k contains rays j with (j * 64) // 181 == k (2 or 3 rays each), so the ray->sector map is integer exact.

Observation layout "avoid-v2" (OBS_DIM_V2 = 198; select with EnvConfig.obs_version / train.py --obs avoid-v2):
    [0:64)    scan_now   as v1
    [64:128)  scan_lag1  the scan 0.3 s (3 control steps) earlier
    [128:192) scan_lag2  the scan 0.6 s (6 control steps) earlier
    [192:195) goal       as v1 (dist, cos, sin)
    [195:198) vel        as v1
Before the episode has that much history the oldest scan of the episode stands in (on the first step all three are
scan_now). Why 0.3 / 0.6 s: in 0.1 s a hard-tier hunter (1.3-1.8 m/s) moves 0.13-0.18 m, ~1.5 % of the
normalised range, which is lost in sector quantisation; over 0.6 s it moves ~1 m, and 0.6 s still ends before a
committed hunter (commit_m 1.8-3.0 m, closing speed ~2-3 m/s) arrives. Two lags give position, velocity and a
curvature cue (re-aims every 0.5 s bend the track). The deployed controller (rl/controller.py) keeps scans by
timestamp and picks the one closest to t - 0.3 / t - 0.6, so irregular control ticks (Isaac) still line up.

Action: the policy outputs a in [-1, 1]^3 (tanh of a Gaussian sample); ``scale_action`` maps it to
    vx = (a0 + 1) / 2 * 1.0  in [0, 1.0] m/s,   vy = 0.5 * a1  in [-0.5, 0.5] m/s,   wz = 1.0 * a2  in [-1, 1] rad/s
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from benchmarks.avoidance.crowd import LIVE_TIERS
from benchmarks.avoidance.pedestrians import CrowdConfig
from benchmarks.avoidance.rl.hunters import TorchCrowd

# --- geometry shared with sim2d (keep in sync with benchmarks/avoidance/sim2d.py) --------------------------------
PERSON_R, SPOT_R, NEAR_MISS_M = 0.30, 0.55, 0.30
DT, ARRIVAL_M = 0.1, 0.4
ACC_VX, ACC_VY, ACC_WZ = 1.5, 1.5, 3.0
RAYS, FOV, MAX_RANGE, RAY_START = 181, math.radians(270), 12.0, 0.65
RAY_REL = np.linspace(-FOV / 2, FOV / 2, RAYS)  # float64, identical to sim2d.lidar
RAY_STEP = FOV / (RAYS - 1)

# --- observation / action contract --------------------------------------------------------------------------------
SECTORS = 64
GOAL_CLIP_M = 10.0
TRACK_VEL_SCALE, TRACK_MIN_MPS, TRACK_MAX_MPS = 2.0, 0.25, 3.0  # avoid-v2t features; = ObstacleTracker defaults
VX_MAX, VY_MAX, WZ_MAX = 1.0, 0.5, 1.0
OBS_VERSION = "avoid-v1"
OBS_LAYOUT = {
    "scan_now": [0, 64], "scan_prev": [64, 128], "goal_dist": [128, 129], "goal_cos": [129, 130],
    "goal_sin": [130, 131], "vel": [131, 134],
    "notes": "scan = min range per sector / 12 m over 181 rays (-135..+135 deg), sector k = rays j with j*64//181==k; "
             "goal_dist = min(d, 10)/10; vel = body (vx, vy, wz); action a in [-1,1]^3 -> "
             "vx=(a0+1)/2*1.0, vy=0.5*a1, wz=1.0*a2",
}
OBS_DIM = 134
ACT_DIM = 3
OBS_VERSION_V2 = "avoid-v2"
OBS_LAYOUT_V2 = {
    "scan_now": [0, 64], "scan_lag1": [64, 128], "scan_lag2": [128, 192], "goal_dist": [192, 193],
    "goal_cos": [193, 194], "goal_sin": [194, 195], "vel": [195, 198], "scan_lags_s": [0.3, 0.6],
    "notes": OBS_LAYOUT["notes"] + "; scan_lagK = the scan closest to t - scan_lags_s[K-1] (the episode's first scan "
             "while the history is shorter)",
}
OBS_DIM_V2 = 198
# every observation version: scan_now, then one scan per lag (in control steps of DT), then goal (3), vel (3)
OBS_SPECS = {
    OBS_VERSION: {"dim": OBS_DIM, "layout": OBS_LAYOUT, "lag_steps": (1,)},
    OBS_VERSION_V2: {"dim": OBS_DIM_V2, "layout": OBS_LAYOUT_V2, "lag_steps": (3, 6)},
    # avoid-v2 + planner-derived features appended (rl/track_features.py). avoid-v2t (D47 probe, D48 training): in
    # sim2d / Isaac the planner's ObstacleTracker supplies them; AvoidEnv emulates that tracker from the true people
    # velocities (AvoidEnv._sense). avoid-v2ts adds the planner's detour commitment and is sim2d only (D47).
    "avoid-v2t": {"dim": OBS_DIM_V2 + 128, "lag_steps": (3, 6), "tracks": "vel",
                  "layout": OBS_LAYOUT_V2 | {"track_vx": [198, 262], "track_vy": [262, 326]}},
    "avoid-v2ts": {"dim": OBS_DIM_V2 + 132, "lag_steps": (3, 6), "tracks": "vel+commitment", "sim2d_only": True,
                   "layout": OBS_LAYOUT_V2 | {"track_vx": [198, 262], "track_vy": [262, 326], "side": [326, 327],
                                              "detour": [327, 328], "target_cos": [328, 329], "target_sin": [329, 330]}},
}


def obs_spec(version: str) -> dict:
    if version not in OBS_SPECS:
        raise ValueError(f"unknown obs_version {version!r} (known: {sorted(OBS_SPECS)})")
    return OBS_SPECS[version]


def _sector_table() -> np.ndarray:
    of = (np.arange(RAYS) * SECTORS) // RAYS
    width = int(np.bincount(of, minlength=SECTORS).max())
    tab = np.zeros((SECTORS, width), dtype=np.int64)
    for k in range(SECTORS):
        members = np.nonzero(of == k)[0]
        tab[k, : len(members)] = members
        tab[k, len(members):] = members[0]  # padding repeats a member: harmless under min
    return tab


SECTOR_TABLE = _sector_table()


def sectorize(ranges):
    """[..., 181] ray ranges (m) -> [..., 64] normalised sector features (min range per sector / 12, in [0, 1]).
    Works on a torch tensor (any device, any batch shape) or a numpy array; the one reduction for train + deploy."""
    if isinstance(ranges, torch.Tensor):
        idx = torch.as_tensor(SECTOR_TABLE, device=ranges.device)
        return ranges[..., idx].amin(dim=-1).clamp(0.0, MAX_RANGE) / MAX_RANGE
    r = np.asarray(ranges, dtype=np.float64)
    return np.clip(r[..., SECTOR_TABLE].min(axis=-1), 0.0, MAX_RANGE) / MAX_RANGE


def ranges_from_points(scan) -> np.ndarray:
    """sim2d/Isaac scan (list of robot-frame (x, y) hit points) -> [181] per-ray ranges, 12 m where nothing hit.
    Each point is assigned to the nearest of the 181 ray angles (they are 1.5 deg apart, so this is exact for sim2d)."""
    out = np.full(RAYS, MAX_RANGE, dtype=np.float64)
    if len(scan) == 0:
        return out
    p = np.asarray(scan, dtype=np.float64).reshape(-1, 2)
    r = np.hypot(p[:, 0], p[:, 1])
    j = np.rint((np.arctan2(p[:, 1], p[:, 0]) + FOV / 2) / RAY_STEP).astype(np.int64)
    ok = (j >= 0) & (j < RAYS)
    np.minimum.at(out, j[ok], r[ok])
    return out


def goal_features(gx, gy):
    """Goal in the robot frame (x forward, y left) -> (dist clipped/normalised, cos bearing, sin bearing)."""
    if isinstance(gx, torch.Tensor):
        d = torch.hypot(gx, gy)
        safe = d.clamp_min(1e-6)
        return torch.stack([d.clamp(max=GOAL_CLIP_M) / GOAL_CLIP_M, gx / safe, gy / safe], dim=-1)
    d = math.hypot(gx, gy)
    safe = max(d, 1e-6)
    return np.array([min(d, GOAL_CLIP_M) / GOAL_CLIP_M, gx / safe, gy / safe], dtype=np.float64)


def build_obs(sect_now, sect_prev, goal_feat, vel):
    """Concatenate in OBS_LAYOUT order. torch: [N, *] tensors; numpy: 1-D arrays."""
    if isinstance(sect_now, torch.Tensor):
        return torch.cat([sect_now, sect_prev, goal_feat, vel], dim=-1)
    return np.concatenate([sect_now, sect_prev, goal_feat, np.asarray(vel, dtype=np.float64)]).astype(np.float32)


def build_obs_lagged(sect_now, sect_lags, goal_feat, vel):
    """Any obs version: scan_now, the lagged scans in OBS_SPECS order, goal, vel. build_obs is the one-lag case."""
    if isinstance(sect_now, torch.Tensor):
        return torch.cat([sect_now, *sect_lags, goal_feat, vel], dim=-1)
    return np.concatenate([sect_now, *sect_lags, goal_feat, np.asarray(vel, dtype=np.float64)]).astype(np.float32)


def scale_action(a):
    """Squashed action in [-1, 1]^3 -> (vx, vy, wz) command. torch [..., 3] or numpy [3]."""
    if isinstance(a, torch.Tensor):
        return torch.stack([(a[..., 0] + 1) * 0.5 * VX_MAX, a[..., 1] * VY_MAX, a[..., 2] * WZ_MAX], dim=-1)
    a = np.asarray(a, dtype=np.float64)
    return np.array([(a[0] + 1) * 0.5 * VX_MAX, a[1] * VY_MAX, a[2] * WZ_MAX])


def cast_rays(pos, yaw, centers, c_active, segs, s_active, return_person: bool = False):
    """Vectorised sim2d.lidar returning per-ray ranges. pos [N,2], yaw [N], centers [N,P,2] (people, radius
    PERSON_R), c_active [N,P] bool, segs [N,W,4] (ax, ay, bx, by), s_active [N,W] bool -> [N, 181] ranges (m).
    return_person: also return [N, 181] long, the person each ray's first return belongs to (-1: a wall or nothing)."""
    rel = torch.as_tensor(RAY_REL, dtype=pos.dtype, device=pos.device)
    ang = yaw[:, None] + rel[None, :]
    dx, dy = torch.cos(ang), torch.sin(ang)  # [N,R]
    best = torch.full_like(ang, MAX_RANGE)
    inf = torch.tensor(float("inf"), dtype=pos.dtype, device=pos.device)
    if centers.shape[1]:
        fx = (pos[:, 0:1] - centers[..., 0])[:, None, :]  # [N,1,P] robot minus centre
        fy = (pos[:, 1:2] - centers[..., 1])[:, None, :]
        b = dx[..., None] * fx + dy[..., None] * fy  # [N,R,P]
        qx, qy = fx - b * dx[..., None], fy - b * dy[..., None]  # closest-approach vector, cancellation-free
        disc = PERSON_R ** 2 - (qx * qx + qy * qy)
        t = -b - torch.sqrt(disc.clamp_min(0.0))
        ok = (disc >= 0) & (t > RAY_START) & c_active[:, None, :]
        tp, who = torch.where(ok, t, inf).min(dim=-1)
        best = torch.minimum(best, tp)
    else:
        tp, who = torch.full_like(best, float("inf")), torch.zeros(best.shape, dtype=torch.long, device=best.device)
    if segs.shape[1]:
        ax, ay, bx, by = (segs[..., i][:, None, :] for i in range(4))  # [N,1,W]
        ex, ey = bx - ax, by - ay
        rx, ry = ax - pos[:, 0, None, None], ay - pos[:, 1, None, None]
        den = dx[..., None] * ey - dy[..., None] * ex  # [N,R,W]
        good = den.abs() > 1e-9
        sden = torch.where(good, den, torch.ones_like(den))
        t = (rx * ey - ry * ex) / sden
        u = (rx * dy[..., None] - ry * dx[..., None]) / sden
        ok = good & (t > RAY_START) & (u >= 0) & (u <= 1) & s_active[:, None, :]
        best = torch.minimum(best, torch.where(ok, t, inf).amin(dim=-1))
    if return_person:  # the person is the first return iff its hit is the ray's range
        return best, torch.where((tp == best) & (tp < MAX_RANGE), who, torch.full_like(who, -1))  # tie -> person
    return best


def point_seg_dist(p, segs):
    """p [N,2], segs [N,W,4] -> [N,W] distance from p to each segment."""
    ax, ay, bx, by = segs.unbind(-1)
    ex, ey = bx - ax, by - ay
    px, py = p[:, 0:1], p[:, 1:2]
    t = (((px - ax) * ex + (py - ay) * ey) / (ex * ex + ey * ey).clamp_min(1e-12)).clamp(0, 1)
    return torch.hypot(px - (ax + t * ex), py - (ay + t * ey))


@dataclass
class EnvConfig:
    goal_min_m: float = 12.0
    goal_max_m: float = 25.0
    max_steps: int = 700  # 70 s, then truncated (bootstrapped, not a failure)
    event_slots: int = 3  # each event uses up to 2 person slots
    max_people: int = 4  # active people cap per env
    wall_slots: int = 2
    # event kind probabilities: none, head_on, pair_head_on, standing, crossing, walker
    event_probs: tuple = (0.40, 0.17, 0.12, 0.10, 0.17, 0.04)
    # wall kind probabilities: none, corridor, box, fence-behind-goal
    wall_probs: tuple = (0.5, 0.25, 0.15, 0.10)
    big_heading_err_p: float = 0.5  # else |heading error| < 0.5 rad
    range_noise_m: float = 0.0
    # reward
    w_progress: float = 1.0  # per metre of goal distance closed
    time_penalty: float = 0.01  # per step
    w_smooth: float = 0.01  # * ||a_t - a_{t-1}||^2 in squashed action units
    near_margin_m: float = NEAR_MISS_M
    w_near: float = 1.0  # per step at clearance 0, linear to 0 at near_margin_m
    contact_penalty: float = 10.0
    wall_penalty: float = 10.0
    success_bonus: float = 10.0
    # personal space (``personal_space_penalty``): a smooth per-step penalty w_space * ((R - c) / R)^2 while the
    # nearest person (ambient or hunter) is within clearance c < R = space_radius_m. 0 = off (legacy/base/hard)
    space_radius_m: float = 0.0
    w_space: float = 0.0
    # closed-loop people (rl/hunters.py); 0 slots = the legacy env
    hunter_slots: int = 0
    hunter_count_probs: tuple = (1.0,)  # P(0, 1, ..., hunter_slots hunters in an episode)
    wanderer_slots: int = 0
    wanderer_p: float = 0.5  # each wanderer slot is filled with this probability
    wanderer_start_min_m: float = 6.0  # wanderers start at least this far from the robot (hunters: always 6 m)
    # yield-early term (D46, RL v4 arm A2): a per-step penalty w_ttc * (1 - ttc / ttc_horizon_s)^2 for the smallest
    # time to collision with any person, from current positions and velocities (robot and person both held constant;
    # collision = centres within PERSON_R + SPOT_R + ttc_margin_m). Graded and early, unlike the distance-based
    # space / near terms: it rises seconds before contact, while braking or a sidestep still avoids it (D45: a stop
    # starting 2 s out avoids ~60 % of RL's wanderer contacts, 1 s out ~20 %). 0 = off (every earlier preset)
    w_ttc: float = 0.0
    ttc_horizon_s: float = 3.0
    ttc_margin_m: float = 0.1
    crowd_tier: str = "base"  # hunter parameters: pedestrians.CrowdConfig defaults + crowd.LIVE_TIERS[crowd_tier]
    obs_version: str = OBS_VERSION  # "avoid-v1" (134) | "avoid-v2" (198, scans at t, t-0.3, t-0.6) | "avoid-v2t" (326)
    # avoid-v2t only: Gaussian noise (m/s, per axis) on the true per-person velocity fed to the emulated tracker
    track_noise_mps: float = 0.0
    # ambient conflict cost (D49), reported as info["cost"] and never added to the reward; PPO-Lagrangian (train.py
    # --cost-limit) constrains it. "Ambient" = everyone the benchmark counts as ambient: the event people, wanderers
    # and hunters not currently locked on / committed. Per step, in [0, 3]:
    #   predicted intrusion  sum_k w^k depth_k / sum_k w^k over k = 1..cost_horizon_steps of cost_dt (w = cost_decay;
    #                        0.5 = CrowdNav++'s 2^-k, near-term heavy; closer to 1 = flatter, the far future counts),
    #                        robot and people at constant velocity; depth = (r - d) / cost_margin_m clipped to [0, 1],
    #                        r = the contact distance + cost_margin_m (prediction intrusion, D47 velocities)
    #   closing speed        within cost_close_m of clearance, the closing speed / 2 m/s, clipped to [0, 1]
    #   contact              1 on an ambient contact
    ambient_cost: bool = False
    cost_horizon_steps: int = 5
    cost_dt: float = 0.4
    cost_decay: float = 0.5
    cost_margin_m: float = 0.3
    cost_close_m: float = 0.4

    def crowd_config(self) -> CrowdConfig:
        return CrowdConfig.from_dict(LIVE_TIERS[self.crowd_tier])


_AMBIENT = dict(event_slots=5, max_people=8, event_probs=(0.15, 0.17, 0.10, 0.18, 0.25, 0.15), wanderer_slots=2,
                wanderer_p=0.5, hunter_slots=4)
PRESETS = {
    "legacy": {},
    # 1-4 hunters (mostly 1-2) + ~4-8 ambient people; crowd.LIVE_TIERS["base"] hunters
    "base": dict(_AMBIENT, hunter_count_probs=(0.05, 0.35, 0.35, 0.15, 0.10), crowd_tier="base"),
    # 1-4 hunters (mostly 2-4), faster and committing later (crowd.LIVE_TIERS["hard"])
    "hard": dict(_AMBIENT, hunter_count_probs=(0.0, 0.15, 0.35, 0.30, 0.20), crowd_tier="hard"),
    # "hard" + the benchmark's everyday crowd near the robot + a personal-space reward (D44). Density is matched where
    # it matters, near the robot, not by headcount: the sim2d hard tier's 56 people (30 wanderers, 8 loiterers, 18
    # hunters) share a 150 x 100 m site, so on average only 3.5 are within 12 m of the patrolling robot. What "hard"
    # lacks is wanderers (oblivious walkers from any direction) close by. Mean wanderers within 2 / 4 / 6 / 8 / 12 m
    # of the robot per 0.1 s step (sim2d: heuristic, seeds 1-32; env: turn-to-goal driver, 2048 envs x 600 steps):
    #   sim2d hard tier   0.025 / 0.176 / 0.393 / 0.673 / 1.401
    #   env "hard"        0.004 / 0.028 / 0.094 / 0.243 / 0.574   (2 slots at p 0.5, starting > 6 m away)
    #   env "hard-dense"  0.027 / 0.190 / 0.453 / 0.765 / 1.407   (5 slots at p 0.45, starting > 2.5 m away)
    # (starting > 6 m away plus 12-25 m legs left the near field empty; matching 4 m with the 6 m rule needed ~7
    # wanderers and tripled the 12 m count.) Hunters, the straight-line event people and walls are as in "hard".
    # Personal space R = 1.0 m of clearance (1.85 m centre to centre) sits between the heuristic planner's hard
    # safety margin (0.35 m) and its clearance cap (2.0 m). w_space = 0.2 makes the penalty at the near-miss line
    # (c = 0.3 m: 0.2 * 0.7^2 = 0.098 per step) about equal to full-speed progress (0.1 per step); inside that the
    # existing w_near ramp takes over.
    "hard-dense": dict(_AMBIENT, hunter_count_probs=(0.0, 0.15, 0.35, 0.30, 0.20), crowd_tier="hard",
                       wanderer_slots=5, wanderer_p=0.45, wanderer_start_min_m=2.5, space_radius_m=1.0, w_space=0.2),
}

# per-term reward bookkeeping: each term summed over the episode, reported with finished episodes as "r_<term>"
REWARD_TERMS = ("progress", "time", "smooth", "near", "space", "contact", "wall", "success", "ttc")


def ambient_cost(pos, wvel, people, people_vel, ambient, horizon_steps: int, dt: float, margin: float, close_m: float,
                 decay: float = 0.5):
    """D49 per-step ambient conflict cost (see EnvConfig.ambient_cost). pos/wvel [N,2] world frame, people/people_vel
    [N,P,2], ambient [N,P] bool -> (cost [N], ambient contact [N] bool)."""
    if people.shape[1] == 0:
        z = torch.zeros_like(pos[:, 0])
        return z, z.bool()
    contact_d = PERSON_R + SPOT_R
    neg = torch.zeros_like(people[..., 0])
    d0 = people - pos[:, None, :]
    dist0 = torch.linalg.norm(d0, dim=-1)
    pred, wsum = torch.zeros_like(pos[:, 0]), 0.0
    for k in range(1, horizon_steps + 1):
        dk = d0 + (people_vel - wvel[:, None, :]) * (k * dt)
        depth = ((contact_d + margin - torch.linalg.norm(dk, dim=-1)) / margin).clamp(0.0, 1.0)
        pred = pred + decay ** k * torch.where(ambient, depth, neg).amax(-1)
        wsum += decay ** k
    closing = -((d0 * (people_vel - wvel[:, None, :])).sum(-1)) / dist0.clamp_min(1e-6)
    close = torch.where(ambient & (dist0 - contact_d < close_m), (closing / 2.0).clamp(0.0, 1.0), neg).amax(-1)
    contact = (ambient & (dist0 < contact_d)).any(-1)
    return pred / wsum + close + contact.float(), contact


def personal_space_penalty(clearance, radius: float, weight: float):
    """Per-step personal-space penalty (>= 0; the env subtracts it) for the nearest person's clearance c (m, surface
    to surface): weight * ((radius - c) / radius)^2 for c < radius, 0 outside, c <= 0 counts as full depth.
    Smooth at the edge (value and slope 0 at c = radius) and monotone inside. radius <= 0 or weight 0 = off.
    Works on a torch tensor or a float."""
    if isinstance(clearance, torch.Tensor):
        if radius <= 0 or weight == 0:
            return torch.zeros_like(clearance)
        depth = ((radius - clearance) / radius).clamp(0.0, 1.0)
        return weight * depth * depth
    if radius <= 0 or weight == 0:
        return 0.0
    depth = min(max((radius - clearance) / radius, 0.0), 1.0)
    return weight * depth * depth


def min_time_to_collision(pos, vel, people, people_vel, active, radius: float):
    """Smallest time (s) until any active person's centre comes within `radius` of the robot's, both moving at
    constant velocity. pos/vel [N,2] (world frame), people/people_vel [N,P,2], active [N,P] -> [N]; 0 if already
    inside, inf if no one is on a collision course."""
    d = people - pos[:, None, :]
    w = people_vel - vel[:, None, :]
    a = (w * w).sum(-1)
    b = 2.0 * (d * w).sum(-1)
    c = (d * d).sum(-1) - radius * radius
    disc = b * b - 4.0 * a * c
    t = (-b - torch.sqrt(disc.clamp_min(0.0))) / (2.0 * a).clamp_min(1e-9)
    inf = torch.full_like(t, float("inf"))
    hit = (a > 1e-9) & (disc >= 0) & (t >= 0)
    ttc = torch.where(c <= 0, torch.zeros_like(t), torch.where(hit, t, inf))
    return torch.where(active, ttc, inf).amin(-1) if ttc.shape[-1] else torch.full_like(pos[:, 0], float("inf"))


def ttc_penalty(ttc, horizon_s: float, weight: float):
    """weight * (1 - ttc / horizon)^2 for ttc < horizon, else 0 (>= 0; the env subtracts it)."""
    if weight == 0 or horizon_s <= 0:
        return torch.zeros_like(ttc)
    depth = (1.0 - ttc / horizon_s).clamp(0.0, 1.0)
    return weight * depth * depth


def preset(name: str, **overrides) -> EnvConfig:
    """EnvConfig for a named preset (``PRESETS``: legacy | base | hard | hard-dense), with optional overrides."""
    return EnvConfig(**(PRESETS[name] | overrides))


KIND_NONE, KIND_HEAD, KIND_PAIR, KIND_STAND, KIND_CROSS, KIND_WALK = range(6)
WALL_NONE, WALL_CORRIDOR, WALL_BOX, WALL_FENCE = range(4)


class AvoidEnv:
    def __init__(self, num_envs: int, device: str | torch.device = "cuda", seed: int = 0, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.n, self.device = num_envs, torch.device(device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        n, P, W, dev = num_envs, 2 * self.cfg.event_slots, self.cfg.wall_slots, self.device
        f = dict(device=dev, dtype=torch.float32)
        self.P, self.W = P, W
        self.pos, self.yaw, self.vel = torch.zeros(n, 2, **f), torch.zeros(n, **f), torch.zeros(n, 3, **f)
        self.goal, self.t = torch.zeros(n, 2, **f), torch.zeros(n, **f)
        self.steps = torch.zeros(n, dtype=torch.long, device=dev)
        spec = obs_spec(self.cfg.obs_version)
        if spec.get("sim2d_only"):
            raise ValueError(f"obs_version {self.cfg.obs_version!r} is sim2d only (clone probe, D47)")
        self.obs_dim, self.lags = spec["dim"], spec["lag_steps"]
        self.track = spec.get("tracks") == "vel"
        # scan history: [:, 0] = now, [:, k] = k control steps ago (clamped to the episode's first scan)
        self.scan_hist = torch.zeros(n, max(self.lags) + 1, SECTORS, **f)
        self.prev_act = torch.zeros(n, 3, **f)
        self.prev_dist = torch.zeros(n, **f)
        self.p0, self.pvel = torch.zeros(n, P, 2, **f), torch.zeros(n, P, 2, **f)
        self.pactive = torch.zeros(n, P, dtype=torch.bool, device=dev)
        self.segs, self.sactive = torch.zeros(n, W, 4, **f), torch.zeros(n, W, dtype=torch.bool, device=dev)
        self.ep_ret, self.ep_len = torch.zeros(n, **f), torch.zeros(n, **f)
        self.ep_near = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ep_minclear = torch.zeros(n, **f)
        self.ep_terms = torch.zeros(n, len(REWARD_TERMS), **f)  # per-term episode sums (REWARD_TERMS order)
        self.ep_space_steps = torch.zeros(n, **f)
        self.ep_ttc_steps = torch.zeros(n, **f)  # steps with someone on a collision course inside the TTC horizon
        self.ep_cost = torch.zeros(n, **f)  # D49 ambient conflict cost, summed over the episode
        c = self.cfg
        self.HH, self.H = c.hunter_slots, c.hunter_slots + c.wanderer_slots
        if len(c.hunter_count_probs) != c.hunter_slots + 1:
            raise ValueError("hunter_count_probs needs hunter_slots + 1 entries")
        self.crowd = TorchCrowd(n, self.H, c.crowd_config(), dev, gen=self.gen, robot_r=SPOT_R) if self.H else None
        # emulated ObstacleTracker state per person slot (avoid-v2t): consecutive scans seen, smoothed world velocity
        self.tr_n = torch.zeros(n, P + self.H, dtype=torch.long, device=dev)
        self.tr_v = torch.zeros(n, P + self.H, 2, **f)
        self.crowd_prev = self.crowd.pos.clone() if self.crowd is not None else None

    # -- randomness helpers ----------------------------------------------------------------------------------------
    def _u(self, *shape, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(*shape, generator=self.gen, device=self.device)

    def _cat(self, probs, *shape):
        p = torch.as_tensor(probs, dtype=torch.float32, device=self.device)
        return torch.multinomial(p, int(np.prod(shape)), replacement=True, generator=self.gen).view(*shape)

    # -- episode sampling ------------------------------------------------------------------------------------------
    def _reset_idx(self, ids: torch.Tensor) -> None:
        cfg, m = self.cfg, len(ids)
        if m == 0:
            return
        E, P, W = cfg.event_slots, self.P, self.W
        D = self._u(m, lo=cfg.goal_min_m, hi=cfg.goal_max_m)
        th = self._u(m, lo=-math.pi, hi=math.pi)
        u = torch.stack([torch.cos(th), torch.sin(th)], -1)  # along the leg
        nrm = torch.stack([-u[:, 1], u[:, 0]], -1)  # left of the leg
        big = self._u(m) < cfg.big_heading_err_p
        err = torch.where(big, self._u(m, lo=-math.pi, hi=math.pi), self._u(m, lo=-0.5, hi=0.5))
        start = torch.zeros(m, 2, device=self.device)
        self.pos[ids], self.yaw[ids], self.goal[ids] = start, th + err, D[:, None] * u
        moving = (self._u(m) < 0.5).float()
        vel = torch.stack([self._u(m), self._u(m, lo=-0.2, hi=0.2), self._u(m, lo=-0.5, hi=0.5)], -1) * moving[:, None]
        self.vel[ids] = vel
        self.prev_act[ids] = torch.stack([2 * vel[:, 0] / VX_MAX - 1, vel[:, 1] / VY_MAX, vel[:, 2] / WZ_MAX], -1)
        self.t[ids], self.steps[ids] = 0.0, 0
        self.prev_dist[ids] = D

        # people: event e fills person slots 2e (and 2e+1 for the pair)
        kind = self._cat(cfg.event_probs, m, E)
        frac = self._u(m, E, lo=0.3, hi=0.7)
        s_meet = frac * D[:, None]
        v_nom = self._u(m, E, lo=0.8, hi=1.0)
        t_meet = err.abs()[:, None] * 1.0 + s_meet / v_nom + 0.7  # turn ~1 s/rad (crowd.py) + accel ramp
        v = self._u(m, E, lo=0.8, hi=1.5)
        lat = self._u(m, E, lo=-0.4, hi=0.4)
        uu, nn = u[:, None, :], nrm[:, None, :]
        meet = uu * s_meet[..., None]  # [m,E,2] the meeting point on the leg
        p0 = torch.zeros(m, E, 2, 2, device=self.device)
        pv = torch.zeros(m, E, 2, 2, device=self.device)
        act = torch.zeros(m, E, 2, dtype=torch.bool, device=self.device)
        # head-on / pair: walking toward the start along the leg, at the meeting point at t_meet
        head = (kind == KIND_HEAD) | (kind == KIND_PAIR)
        for k, off in enumerate((torch.where(kind == KIND_PAIR, lat - 0.6, lat), lat + 0.6)):
            pos_k = meet + uu * (v * t_meet)[..., None] + nn * off[..., None]
            p0[:, :, k] = torch.where(head[..., None], pos_k, p0[:, :, k])
            pv[:, :, k] = torch.where(head[..., None], -uu * v[..., None], pv[:, :, k])
        act[:, :, 0] |= head
        act[:, :, 1] |= kind == KIND_PAIR
        # standing on / near the line
        stand = kind == KIND_STAND
        p0[:, :, 0] = torch.where(stand[..., None], meet + nn * lat[..., None], p0[:, :, 0])
        act[:, :, 0] |= stand
        # crossing: perpendicular, reaching the leg at t_meet +- 2 s
        cross = kind == KIND_CROSS
        side = torch.where(self._u(m, E) < 0.5, -1.0, 1.0)
        t_cross = t_meet + self._u(m, E, lo=-2.0, hi=2.0)
        pos_c = meet + nn * (side * v * t_cross)[..., None]
        p0[:, :, 0] = torch.where(cross[..., None], pos_c, p0[:, :, 0])
        pv[:, :, 0] = torch.where(cross[..., None], -nn * (side * v)[..., None], pv[:, :, 0])
        act[:, :, 0] |= cross
        # random slow walker somewhere around the leg
        walk = kind == KIND_WALK
        pos_w = uu * (3.0 + (D[:, None] - 3.0) * self._u(m, E))[..., None] + nn * self._u(m, E, lo=-6.0, hi=6.0)[..., None]
        wdir = self._u(m, E, lo=-math.pi, hi=math.pi)
        wsp = self._u(m, E, lo=0.2, hi=0.8)
        p0[:, :, 0] = torch.where(walk[..., None], pos_w, p0[:, :, 0])
        pv[:, :, 0] = torch.where(walk[..., None], torch.stack([torch.cos(wdir), torch.sin(wdir)], -1) * wsp[..., None],
                                  pv[:, :, 0])
        act[:, :, 0] |= walk
        act = act.view(m, P)
        act &= torch.cumsum(act.long(), dim=1) <= cfg.max_people
        self.p0[ids], self.pvel[ids], self.pactive[ids] = p0.view(m, P, 2), pv.view(m, P, 2), act

        # walls
        wk = self._cat(cfg.wall_probs, m, W)
        uw, nw = u[:, None, :].expand(m, W, 2), nrm[:, None, :].expand(m, W, 2)
        Dw = D[:, None].expand(m, W)
        a = torch.zeros(m, W, 2, device=self.device)
        b = torch.zeros(m, W, 2, device=self.device)
        # corridor wall parallel to the leg
        side = torch.where(self._u(m, W) < 0.5, -1.0, 1.0)
        off = side * self._u(m, W, lo=1.5, hi=5.0)
        s0 = -6.0 + (0.5 * Dw + 6.0) * self._u(m, W)
        L = self._u(m, W, lo=8.0, hi=30.0)
        ca, cb = uw * s0[..., None] + nw * off[..., None], uw * (s0 + L)[..., None] + nw * off[..., None]
        sel = (wk == WALL_CORRIDOR)[..., None]
        a, b = torch.where(sel, ca, a), torch.where(sel, cb, b)
        # box edge: short segment near the leg, any orientation
        c = uw * (Dw * self._u(m, W, lo=0.25, hi=0.75))[..., None] + nw * self._u(m, W, lo=-3.0, hi=3.0)[..., None]
        ang = self._u(m, W, lo=-math.pi, hi=math.pi)
        half = self._u(m, W, lo=0.5, hi=2.0)
        dvec = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * half[..., None]
        sel = (wk == WALL_BOX)[..., None]
        a, b = torch.where(sel, c - dvec, a), torch.where(sel, c + dvec, b)
        # fence behind the goal, perpendicular to the leg
        c = uw * (Dw + self._u(m, W, lo=1.5, hi=8.0))[..., None] + nw * self._u(m, W, lo=-5.0, hi=5.0)[..., None]
        half = self._u(m, W, lo=5.0, hi=15.0)
        sel = (wk == WALL_FENCE)[..., None]
        a, b = torch.where(sel, c - nw * half[..., None], a), torch.where(sel, c + nw * half[..., None], b)
        segs = torch.cat([a, b], -1)
        # keep the start and the goal free: a wall that would touch either is dropped
        d_start = point_seg_dist(start, segs)
        d_goal = point_seg_dist(self.goal[ids], segs)
        self.segs[ids] = segs
        self.sactive[ids] = (wk != WALL_NONE) & (d_start > SPOT_R + 1.0) & (d_goal > SPOT_R + 0.5)

        # closed-loop people (drawn after everything else, so the legacy config samples exactly as before)
        if self.crowd is not None:
            HH, H = self.HH, self.H
            n_h = self._cat(cfg.hunter_count_probs, m)
            slot = torch.arange(H, device=self.device)[None, :]
            hunter_on = slot < n_h[:, None]
            wander_on = self._u(m, H) < cfg.wanderer_p
            self.crowd.reset_leg(ids, D, u, nrm, self.segs, self.sactive, hunter_on, wander_on, HH,
                                 cfg.wanderer_start_min_m)

        self.ep_ret[ids], self.ep_len[ids] = 0.0, 0.0
        self.ep_near[ids], self.ep_minclear[ids] = False, float("inf")
        self.ep_terms[ids], self.ep_space_steps[ids], self.ep_ttc_steps[ids] = 0.0, 0.0, 0.0
        self.tr_n[ids], self.tr_v[ids] = 0, 0.0
        self.ep_cost[ids] = 0.0

    # -- sensing ---------------------------------------------------------------------------------------------------
    def people_at(self, t=None):
        t = self.t if t is None else t
        return self.p0 + self.pvel * t[:, None, None]

    def people(self):
        """(centres [N, P + H, 2], active [N, P + H]): the ambient walkers now, then the closed-loop people."""
        pp, act = self.people_at(), self.pactive
        if self.crowd is not None:
            cp, ca = self.crowd.people()
            pp, act = torch.cat([pp, cp.to(pp.dtype)], 1), torch.cat([act, ca], 1)
        return pp, act

    def _sectors(self, ids=None, with_person: bool = False):
        sl = slice(None) if ids is None else ids
        pp, act = self.people()
        out = cast_rays(self.pos[sl], self.yaw[sl], pp[sl], act[sl], self.segs[sl], self.sactive[sl],
                        return_person=with_person)
        ranges, who = out if with_person else (out, None)
        if self.cfg.range_noise_m > 0:
            hit = ranges < MAX_RANGE
            noise = torch.randn(ranges.shape, generator=self.gen, device=self.device) * self.cfg.range_noise_m
            ranges = torch.where(hit, (ranges + noise).clamp(RAY_START, MAX_RANGE), ranges)
        return (sectorize(ranges), ranges, who) if with_person else sectorize(ranges)

    def _sense(self, ids=None):
        """(sectors [n, 64], track features [n, 128] or None). avoid-v2t: one scan of the emulated ObstacleTracker
        (robots/spot/local_planner.py), from the true velocities instead of scan-to-scan cluster centroids:
          - a person is seen when >= 2 rays return on them (the tracker ignores single-point clusters)
          - velocity = alpha 0.5 smoothing from 0 on the second consecutive scan (its first match), clipped at 3 m/s
          - trusted from the third consecutive scan (2 matches) and at >= 0.25 m/s, else 0
          - feature per sector = the trusted velocity of the person owning the sector's nearest return (0 for a wall
            or nothing), in robot-frame axes / TRACK_VEL_SCALE: rl/track_features.py's layout exactly"""
        if not self.track:
            return self._sectors(ids), None
        sl = slice(None) if ids is None else ids
        sect, ranges, who = self._sectors(ids, with_person=True)
        m, NP = who.shape[0], self.tr_n.shape[1]
        hits = torch.zeros(m, NP, device=self.device).scatter_add_(1, who.clamp_min(0), (who >= 0).float())
        raw = self.pvel
        if self.crowd is not None:
            raw = torch.cat([raw, ((self.crowd.pos - self.crowd_prev) / DT).to(raw.dtype)], 1)
        raw = raw[sl]
        if self.cfg.track_noise_mps > 0:
            raw = raw + torch.randn(raw.shape, generator=self.gen, device=self.device) * self.cfg.track_noise_mps
        n_new = torch.where(hits >= 2, self.tr_n[sl] + 1, torch.zeros_like(self.tr_n[sl]))
        v = torch.where((n_new >= 2)[..., None], 0.5 * raw + 0.5 * self.tr_v[sl], torch.zeros_like(raw))
        speed = torch.linalg.norm(v, dim=-1, keepdim=True)
        v = v * (TRACK_MAX_MPS / speed.clamp_min(TRACK_MAX_MPS))
        self.tr_n[sl], self.tr_v[sl] = n_new, v
        trusted = torch.where(((n_new >= 3) & (speed[..., 0] >= TRACK_MIN_MPS))[..., None], v, torch.zeros_like(v))
        tab = torch.as_tensor(SECTOR_TABLE, device=self.device)
        k = ranges[:, tab].argmin(-1)  # [m, 64]: the nearest ray within each sector's table row
        ray = tab[torch.arange(SECTORS, device=self.device)[None, :], k]
        owner = who.gather(1, ray)
        sv = trusted.gather(1, owner.clamp_min(0)[..., None].expand(-1, -1, 2)) * (owner >= 0)[..., None]
        c, s = torch.cos(self.yaw[sl])[:, None], torch.sin(self.yaw[sl])[:, None]
        rx, ry = c * sv[..., 0] + s * sv[..., 1], -s * sv[..., 0] + c * sv[..., 1]
        return sect, torch.cat([rx, ry], -1) / TRACK_VEL_SCALE

    def _goal_feat(self, ids=None):
        sl = slice(None) if ids is None else ids
        d = self.goal[sl] - self.pos[sl]
        c, s = torch.cos(self.yaw[sl]), torch.sin(self.yaw[sl])
        return goal_features(c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1])

    @property
    def prev_sect(self):
        """v1's scan_prev: the scan one control step ago."""
        return self.scan_hist[:, 1]

    @prev_sect.setter
    def prev_sect(self, sect):  # a hand-made "previous" scan: the whole history before now
        self.scan_hist[:, 1:] = sect[:, None]

    def _obs(self, hist, goal_feat, vel, extra=None):
        obs = build_obs_lagged(hist[:, 0], [hist[:, k] for k in self.lags], goal_feat, vel)
        return obs if extra is None else torch.cat([obs, extra], -1)

    def reset(self) -> torch.Tensor:
        ids = torch.arange(self.n, device=self.device)
        self._reset_idx(ids)
        if self.crowd is not None:
            self.crowd_prev = self.crowd.pos.clone()
        sect, extra = self._sense()
        self.scan_hist = sect[:, None].repeat(1, self.scan_hist.shape[1], 1)
        return self._obs(self.scan_hist, self._goal_feat(), self.vel, extra)

    def clearances(self):
        """(people clearance [N], wall clearance [N]); inf where there is nothing."""
        pp, act = self.people()
        dp = torch.hypot(pp[..., 0] - self.pos[:, 0:1], pp[..., 1] - self.pos[:, 1:2]) - PERSON_R - SPOT_R
        cp = torch.where(act, dp, torch.full_like(dp, float("inf"))).amin(-1)
        dw = point_seg_dist(self.pos, self.segs) - SPOT_R
        cw = torch.where(self.sactive, dw, torch.full_like(dw, float("inf"))).amin(-1) if self.W else torch.full_like(cp, float("inf"))
        return cp, cw

    # -- stepping --------------------------------------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        """action: squashed [N, 3] in [-1, 1]. Returns obs, reward, done, info (info['final_obs'] holds the
        pre-reset observation for bootstrapping truncated envs; info['episodes'] the stats of finished ones)."""
        cfg = self.cfg
        a = action.clamp(-1.0, 1.0)
        cmd = scale_action(a)
        lim = torch.tensor([ACC_VX * DT, ACC_VY * DT, ACC_WZ * DT], device=self.device)
        self.vel = self.vel + torch.maximum(-lim, torch.minimum(lim, cmd - self.vel))
        self.yaw = self.yaw + self.vel[:, 2] * DT
        c, s = torch.cos(self.yaw), torch.sin(self.yaw)
        vx, vy = self.vel[:, 0], self.vel[:, 1]
        self.pos = self.pos + torch.stack([vx * c - vy * s, vx * s + vy * c], -1) * DT
        self.yaw = torch.remainder(self.yaw + math.pi, 2 * math.pi) - math.pi
        self.t = self.t + DT
        self.steps += 1
        crowd_prev = (self.crowd.pos.clone() if (self.crowd is not None and (cfg.w_ttc or self.track or cfg.ambient_cost))
                      else None)
        self.crowd_prev = crowd_prev
        if self.crowd is not None:  # the crowd sees the robot where it is now (sim2d: crowd.step(t, robot at t))
            self.crowd.step(self.pos, self.steps.double() / 10.0, self.segs, self.sactive)

        cp, cw = self.clearances()
        dist = torch.linalg.norm(self.goal - self.pos, dim=-1)
        depth = ((cfg.near_margin_m - cp) / cfg.near_margin_m).clamp(0.0, 1.0)
        contact = cp < 0
        wall = (cw < 0) & ~contact
        success = (dist <= ARRIVAL_M) & ~contact & ~wall
        progress = cfg.w_progress * (self.prev_dist - dist)
        smooth = cfg.w_smooth * ((a - self.prev_act) ** 2).sum(-1)
        space = personal_space_penalty(cp, cfg.space_radius_m, cfg.w_space)
        if cfg.w_ttc:
            pp, pact = self.people()
            pv = self.pvel
            if crowd_prev is not None:  # closed-loop people: velocity from this step's displacement
                pv = torch.cat([pv, ((self.crowd.pos - crowd_prev) / DT).to(pv.dtype)], 1)
            wvel = torch.stack([vx * c - vy * s, vx * s + vy * c], -1)
            ttc = min_time_to_collision(self.pos, wvel, pp, pv, pact, PERSON_R + SPOT_R + cfg.ttc_margin_m)
            ttc_pen = ttc_penalty(ttc, cfg.ttc_horizon_s, cfg.w_ttc)
            self.ep_ttc_steps += (ttc < cfg.ttc_horizon_s).float()
        else:
            ttc_pen = torch.zeros_like(cp)
        if cfg.ambient_cost:
            pp, pact = self.people()
            pv = self.pvel
            amb = self.pactive
            if self.crowd is not None:
                pv = torch.cat([pv, ((self.crowd.pos - crowd_prev) / DT).to(pv.dtype)], 1)
                amb = torch.cat([amb, self.crowd.active & ~self.crowd.open_lock()], 1)
            wv = torch.stack([vx * c - vy * s, vx * s + vy * c], -1)
            cost, amb_contact = ambient_cost(self.pos, wv, pp, pv, amb, cfg.cost_horizon_steps, cfg.cost_dt,
                                             cfg.cost_margin_m, cfg.cost_close_m, cfg.cost_decay)
        else:
            cost, amb_contact = torch.zeros_like(cp), torch.zeros_like(contact)
        self.ep_cost += cost
        # summed in the pre-D44 order, so presets without personal space give bit-identical rewards
        r = progress - cfg.time_penalty
        r = r - smooth
        r = r - cfg.w_near * depth
        if cfg.w_space and cfg.space_radius_m > 0:
            r = r - space
        if cfg.w_ttc:
            r = r - ttc_pen
        r = r - cfg.contact_penalty * contact.float() - cfg.wall_penalty * wall.float() + cfg.success_bonus * success.float()
        self.ep_terms += torch.stack([  # signed contributions, in REWARD_TERMS order
            progress, torch.full_like(r, -cfg.time_penalty), -smooth, -cfg.w_near * depth, -space,
            -cfg.contact_penalty * contact.float(), -cfg.wall_penalty * wall.float(),
            cfg.success_bonus * success.float(), -ttc_pen,
        ], -1)
        if cfg.space_radius_m > 0:
            self.ep_space_steps += (cp < cfg.space_radius_m).float()
        terminated = contact | wall | success
        truncated = (self.steps >= cfg.max_steps) & ~terminated
        done = terminated | truncated
        self.prev_dist, self.prev_act = dist, a

        self.ep_ret += r
        self.ep_len += 1
        self.ep_near |= cp < NEAR_MISS_M
        self.ep_minclear = torch.minimum(self.ep_minclear, cp)

        sect, extra = self._sense()
        hist = torch.cat([sect[:, None], self.scan_hist[:, :-1]], 1)
        obs = self._obs(hist, self._goal_feat(), self.vel, extra)
        info = {"final_obs": obs, "truncated": truncated, "terminated": terminated, "cost": cost}
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        info["episodes"] = {
            "ret": self.ep_ret[ids].clone(), "len": self.ep_len[ids].clone(), "success": success[ids],
            "contact": contact[ids], "wall": wall[ids], "timeout": truncated[ids], "near": self.ep_near[ids].clone(),
            "space_frac": self.ep_space_steps[ids] / self.ep_len[ids],  # share of steps with someone inside R
            "ttc_frac": self.ep_ttc_steps[ids] / self.ep_len[ids],  # share of steps on a collision course (w_ttc > 0)
            "cost": self.ep_cost[ids] / self.ep_len[ids],  # D49: mean ambient cost per step (0 unless ambient_cost)
            "ambient_contact": amb_contact[ids],  # D49: the episode ended on an ambient contact (ambient_cost only)
        }
        info["episodes"] |= {f"r_{k}": self.ep_terms[ids, i].clone() for i, k in enumerate(REWARD_TERMS)}
        if self.crowd is not None:  # closed-loop stats: locks, commits, and locks that ended in contact
            info["episodes"] |= {"locks": self.crowd.n_locks[ids].clone(), "commits": self.crowd.n_commits[ids].clone(),
                                 "hits": self.crowd.n_hits[ids].clone()}
        else:
            z = torch.zeros(len(ids), dtype=torch.long, device=self.device)
            info["episodes"] |= {"locks": z, "commits": z, "hits": z}
        if len(ids):
            self._reset_idx(ids)
            fresh, fextra = self._sense(ids)  # the reset legs' people are new to the tracker (_reset_idx cleared it)
            obs = obs.clone()
            hist[ids] = fresh[:, None].expand(-1, hist.shape[1], -1)
            obs[ids] = self._obs(hist[ids], self._goal_feat(ids), self.vel[ids], fextra)
        self.scan_hist = hist
        return obs, r, done, info

    def config_dict(self) -> dict:
        return asdict(self.cfg)
