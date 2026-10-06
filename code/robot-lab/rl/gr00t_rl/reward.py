"""Reward and STRICT-success bookkeeping for the SO-101 put-the-object-in-the-bowl task (pure torch, no Isaac).

Every quantity comes from simulator state readings taken once per control tick (30 Hz), with the same thresholds as
tools/gr00t_eval.py (lift 3 cm, bowl shift 3 cm, bowl tilt 20 deg, blow-up limits). The tracker is fed the reading
AFTER each env.step; a reading from a tick on which the env terminated is never used (Isaac Lab auto-resets inside
env.step, so that reading already belongs to the next layout) -- the criteria use the last pre-terminal reading,
exactly like gr00t_eval.py:231-243.

Reward per tick (summed over the 8 ticks of a chunk = one RL step). Stated exactly in rl/DESIGN.md section 2:
  milestones, each paid ONCE per episode (latched):
    reach        +0.25  TCP within 2 cm of the object centre
    lift         +1.0   object centre risen >= 3 cm above its settled start height (the strict lift criterion)
    over_bowl    +1.0   after lift: object centre within the bowl radius (xy) and above the bowl root
    release      +2.0   after lift: jaw opened past the success threshold (0.5 rad) while the object centre is within
                        the bowl radius (xy)  -- the behaviour night 1 never produced
  terminal:
    strict       +10.0  the task's success term fires AND lifted AND bowl ok AND no blow-up (= gr00t_eval STRICT)
    blowup       -2.0   (episode ends, as in the eval)
    dropped      -1.0   the env's object_dropping termination (object fell off the table)
  once-only penalty:
    bowl_bad     -2.0   first tick the bowl is tilted > 20 deg or shifted > 3 cm (the strict bowl criterion fails)
  potential-based shaping (Ng et al. 1999, so it cannot change which policy is optimal and cannot be farmed):
    F = gamma_tick * Phi(s') - Phi(s),  Phi = W_PHI * (0.3 - min(d_xy(object, bowl), 0.3))  if the object is "carried"
                                        Phi = 0                                              otherwise
    (Phi >= 0 and 0 when idle: a constant NEGATIVE potential would pay (1 - gamma) * |Phi| per tick for merely staying
    alive; with Phi >= 0, hovering while carrying costs (1 - gamma) * Phi per tick instead)
    carried = lifted (latched) AND (object held: rise >= 2 cm and TCP-object <= 5 cm
                                    OR object centre inside the bowl radius)
    so carrying toward the bowl pays, releasing INSIDE the bowl footprint costs nothing, dropping outside it pays the
    potential back; pushing the object without a lift never enters the carried branch.
The headline metric is never this reward: it is the STRICT success count, reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

LIFT_RISE = 0.03
BOWL_SHIFT = 0.03
BOWL_TILT_DEG = 20.0
BOWL_RADIUS = 0.055          # robot_lab/tasks/so101_mug_bowl.py BOWL_RADIUS (success-term footprint)
JAW_OPEN_RAD = 0.5           # object_in_bowl "opened" threshold
VEL_MAX, OBJ_SPEED_MAX, Z_FLOOR = 30.0, 3.0, -0.03   # gr00t_eval.py:79
REACH_DIST = 0.02
HELD_RISE, HELD_DIST = 0.02, 0.05
PHI_CAP = 0.3


@dataclass
class RewardWeights:
    reach: float = 0.25
    lift: float = 1.0
    over_bowl: float = 1.0
    release: float = 2.0
    strict: float = 10.0
    blowup: float = -2.0
    dropped: float = -1.0
    bowl_bad: float = -2.0
    w_phi: float = 2.0
    gamma_tick: float = 0.99 ** (1.0 / 8.0)   # per-tick discount consistent with gamma = 0.99 per 8-tick RL step


@dataclass
class Reading:
    """One tick of simulator state for all envs (env frame, metres / radians)."""
    obj: torch.Tensor        # (N,3) object geometric centre
    bowl: torch.Tensor       # (N,3) bowl root
    bowl_tilt_deg: torch.Tensor  # (N,)
    tcp: torch.Tensor        # (N,3)
    jaw: torch.Tensor        # (N,) gripper joint [rad]
    joint_vel_max: torch.Tensor  # (N,)
    obj_speed: torch.Tensor  # (N,)


class RewardTracker:
    KEYS = ("reach", "lift", "over_bowl", "release", "strict", "blowup", "dropped", "bowl_bad", "shaping")

    def __init__(self, n: int, device, w: RewardWeights | None = None):
        self.n, self.dev, self.w = n, torch.device(device), w or RewardWeights()
        z = lambda dt=torch.bool: torch.zeros(n, dtype=dt, device=self.dev)  # noqa: E731
        self.z0, self.max_rise = z(torch.float32), z(torch.float32)
        self.bowl0 = torch.zeros(n, 3, device=self.dev)
        self.reached, self.lifted, self.over, self.released, self.bowl_bad = z(), z(), z(), z(), z()
        self.done, self.strict, self.task, self.blowup, self.dropped = z(), z(), z(), z(), z()
        self.phi = z(torch.float32)
        self.bowl_last_shift, self.bowl_last_tilt = z(torch.float32), z(torch.float32)
        self.lifted_step = torch.full((n,), -1, dtype=torch.long, device=self.dev)
        self.end_step = torch.full((n,), -1, dtype=torch.long, device=self.dev)

    def start(self, r: Reading):
        """New episode for ALL envs (synchronous waves), from the settled reading."""
        self.__init__(self.n, self.dev, self.w)
        self.z0 = r.obj[:, 2].clone()
        self.bowl0 = r.bowl.clone()
        self.phi = self._phi(r)

    # ---- pieces ----
    def _geom(self, r: Reading):
        rise = r.obj[:, 2] - self.z0
        d_xy = torch.linalg.norm(r.obj[:, :2] - r.bowl[:, :2], dim=-1)
        d_tcp = torch.linalg.norm(r.tcp - r.obj, dim=-1)
        in_fp = d_xy < BOWL_RADIUS
        return rise, d_xy, d_tcp, in_fp

    def _phi(self, r: Reading, lifted=None):
        rise, d_xy, d_tcp, in_fp = self._geom(r)
        lifted = self.lifted if lifted is None else lifted
        held = (rise >= HELD_RISE) & (d_tcp <= HELD_DIST)
        carried = lifted & (held | in_fp)
        return torch.where(carried, self.w.w_phi * (PHI_CAP - d_xy.clamp(max=PHI_CAP)), torch.zeros_like(d_xy))

    def step(self, r: Reading, terminated: torch.Tensor, task_success: torch.Tensor, dropped: torch.Tensor,
             tick: int) -> dict[str, torch.Tensor]:
        """One tick. terminated/task_success/dropped: the env's termination flags for THIS step (pre-reset). The
        reading r is post-step and is ignored for envs that terminated on this step (it is the next layout).
        Returns per-env reward components for this tick (zeros for envs already done)."""
        w, out = self.w, {k: torch.zeros(self.n, device=self.dev) for k in self.KEYS}
        live = ~self.done
        # -- terminal tick from the env (success term / object dropped / time-out): judge on the last valid reading
        term = live & terminated
        strict_now = term & task_success & self.lifted & ~self.bowl_bad_last() & ~self.blowup
        out["strict"] += w.strict * strict_now.float()
        out["dropped"] += w.dropped * (term & dropped & ~task_success).float()
        self.task |= term & task_success
        self.strict |= strict_now
        self.dropped |= term & dropped
        self.end_step = torch.where(term, torch.full_like(self.end_step, tick), self.end_step)
        self.done |= term
        # -- live, non-terminal tick: the reading is valid
        ok = live & ~terminated
        rise, d_xy, d_tcp, in_fp = self._geom(r)
        self.max_rise = torch.where(ok, torch.maximum(self.max_rise, rise), self.max_rise)
        new_reach = ok & ~self.reached & (d_tcp <= REACH_DIST)
        new_lift = ok & ~self.lifted & (rise >= LIFT_RISE)
        lifted = self.lifted | new_lift
        new_over = ok & lifted & ~self.over & in_fp & (r.obj[:, 2] > r.bowl[:, 2])
        new_rel = ok & lifted & ~self.released & in_fp & (r.jaw > JAW_OPEN_RAD)
        shift = torch.linalg.norm(r.bowl[:, :2] - self.bowl0[:, :2], dim=-1)
        self.bowl_last_shift = torch.where(ok, shift, self.bowl_last_shift)
        self.bowl_last_tilt = torch.where(ok, r.bowl_tilt_deg, self.bowl_last_tilt)
        bad_now = ok & ((shift > BOWL_SHIFT) | (r.bowl_tilt_deg > BOWL_TILT_DEG))
        new_bad = bad_now & ~self.bowl_bad
        blow = ok & ((r.joint_vel_max > VEL_MAX) | (r.obj[:, 2] < Z_FLOOR) | (r.bowl[:, 2] < Z_FLOOR)
                     | (r.obj_speed > OBJ_SPEED_MAX))
        out["reach"] += w.reach * new_reach.float()
        out["lift"] += w.lift * new_lift.float()
        out["over_bowl"] += w.over_bowl * new_over.float()
        out["release"] += w.release * new_rel.float()
        out["bowl_bad"] += w.bowl_bad * new_bad.float()
        out["blowup"] += w.blowup * blow.float()
        self.lifted_step = torch.where(new_lift, torch.full_like(self.lifted_step, tick), self.lifted_step)
        self.reached |= new_reach
        self.lifted = lifted
        self.over |= new_over
        self.released |= new_rel
        self.bowl_bad |= new_bad
        phi_new = self._phi(r)
        out["shaping"] += torch.where(ok, w.gamma_tick * phi_new - self.phi, torch.zeros_like(phi_new))
        self.phi = torch.where(ok, phi_new, self.phi)
        # a blow-up ends the episode as a failure (gr00t_eval.py:246-249)
        self.blowup |= blow
        self.end_step = torch.where(blow, torch.full_like(self.end_step, tick), self.end_step)
        self.done |= blow
        return out

    def bowl_bad_last(self):
        """The strict bowl criterion on the LAST valid reading (gr00t_eval uses bowl_last, not 'ever')."""
        return (self.bowl_last_shift > BOWL_SHIFT) | (self.bowl_last_tilt > BOWL_TILT_DEG)

    def finish(self, tick: int):
        """End of the wave's time budget: every still-live env ends as a time-out (failure)."""
        self.end_step = torch.where(self.done, self.end_step, torch.full_like(self.end_step, tick))
        self.done[:] = True

    def summary(self) -> dict[str, torch.Tensor]:
        return {"strict": self.strict, "task": self.task, "lifted": self.lifted, "reached": self.reached,
                "over_bowl": self.over, "released_over": self.released, "bowl_bad": self.bowl_bad,
                "blowup": self.blowup, "dropped": self.dropped, "max_rise": self.max_rise,
                "end_step": self.end_step, "lifted_step": self.lifted_step}
