"""Record the heuristic planner as a teacher for behaviour cloning (D46, RL v4 arm A3).

Runs sim2d patrols (live crowd) driven by the heuristic planner and records, at every control step, the observation
the RL controller would build at that moment (RLController.observe: same sectorisation, lagged scans, goal and
velocity features) and the planner's command as a policy action in [-1, 1]^3 (the inverse of env.scale_action).
Arrival steps are skipped, as RLController skips them.

    PYTHONPATH=. python tools/collect_teacher.py --seeds 1000-1399 --live hard --workers 96 \\
        --out /mnt/weights/ai/patrol-lab-data/rl/teacher-heuristic-hard-s1000-1399

Seeds must stay disjoint from the selection seeds (21-60) and the sealed test seeds (121-200): the script refuses
any overlap. Writes one ep_<seed>.npz (obs float32 [T, obs width], act float32 [T, 3], t, plus the episode row) per seed
and summary.json (episodes, steps, completion, contact counts, action clip rate).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FORBIDDEN = set(range(21, 61)) | set(range(121, 201))  # selection + sealed test seeds (D46)
_RECORDS: list = []


_DRIVER = None  # DAgger: (model, obs_version) of a learner that drives while the planner only labels
_OBS_VERSION = "avoid-v2"  # the observation recorded without a driver (D47: avoid-v2t / avoid-v2ts)


class TeacherController:
    """The heuristic planner, plus a shadow RLController that only builds observations. With a driver (DAgger), the
    learner's RLController drives and builds the observations, and the planner, fed the same scans every step, only
    supplies the labels: the data then covers the states the learner actually reaches."""
    name = "teacher"

    def __init__(self):
        import torch
        from benchmarks.avoidance.rl.controller import RLController
        from benchmarks.avoidance.rl.env import obs_spec
        from benchmarks.avoidance.rl.ppo import ActorCritic
        from benchmarks.avoidance.sim2d import Heuristic
        torch.set_num_threads(1)
        self.teacher = Heuristic()
        if _DRIVER is not None:
            self.driver = RLController(None, "command", model=_DRIVER[0], obs_version=_DRIVER[1])
        else:
            self.driver = None
            dummy = ActorCritic(obs_spec(_OBS_VERSION)["dim"], 3, (8,))  # never called: observe() needs the width
            self.shadow = RLController(None, "command", model=dummy, obs_version=_OBS_VERSION)
        self.obs, self.act, self.ts = [], [], []
        _RECORDS.append(self)

    def step(self, t, pose, goal, scan):
        from benchmarks.avoidance.rl.env import ARRIVAL_M, VX_MAX, VY_MAX, WZ_MAX
        if self.driver is not None:
            lvx, lvy, lwz, lstatus = self.driver.step(t, pose, goal, scan)  # builds obs from the same inputs
            tvx, tvy, twz, tstatus = self.teacher.step(t, pose, goal, scan)  # the label; its tracker sees every scan
            if lstatus != "arrived" and tstatus != "arrived":
                self.obs.append(self.driver.last_obs.astype(np.float32))
                self.act.append(np.array([2.0 * tvx / VX_MAX - 1.0, tvy / VY_MAX, twz / WZ_MAX], dtype=np.float32))
                self.ts.append(t)
            return lvx, lvy, lwz, lstatus
        obs, dist = self.shadow.observe(t, pose, goal, scan)
        vx, vy, wz, status = self.teacher.step(t, pose, goal, scan)
        if status == "arrived" or dist <= ARRIVAL_M:
            return vx, vy, wz, status
        self.obs.append(obs.astype(np.float32))
        self.act.append(np.array([2.0 * vx / VX_MAX - 1.0, vy / VY_MAX, wz / WZ_MAX], dtype=np.float32))
        self.ts.append(t)
        # the shadow's velocity feature integrates the commands sim2d will apply, exactly as RLController.step does
        lim = self.shadow_lim()
        self.shadow.vel = self.shadow.vel + np.clip(np.array([vx, vy, wz]) - self.shadow.vel, -lim, lim)
        return vx, vy, wz, status

    @staticmethod
    def shadow_lim():
        from benchmarks.avoidance.rl.controller import _LIM
        return _LIM


def _init(driver_ckpt=None, obs_version="avoid-v2"):
    global _DRIVER, _OBS_VERSION
    _OBS_VERSION = obs_version
    from benchmarks.avoidance import sim2d
    if driver_ckpt:
        import torch
        from benchmarks.avoidance.rl.controller import load_policy
        torch.set_num_threads(1)
        model, ck = load_policy(driver_ckpt)
        _DRIVER = (model, ck["obs_version"])
    sim2d.register("teacher", TeacherController)


def _job(args):
    seed, live, out = args
    from benchmarks.avoidance import sim2d
    _RECORDS.clear()
    row = sim2d.run_episode(seed, "teacher", live=live)
    rec = _RECORDS[-1]
    width = (rec.driver if rec.driver is not None else rec.shadow).model.obs_rms.mean.shape[0]
    obs = np.stack(rec.obs) if rec.obs else np.zeros((0, int(width)), np.float32)
    act = np.stack(rec.act) if rec.act else np.zeros((0, 3), np.float32)
    clipped = int((np.abs(act) > 1.0 + 1e-6).any(-1).sum())
    np.savez_compressed(Path(out) / f"ep_{seed:05d}.npz", obs=obs, act=np.clip(act, -1, 1), t=np.array(rec.ts),
                        row=json.dumps(row))
    return {"seed": seed, "steps": int(len(obs)), "clipped": clipped, "row": row}


def main(argv=None) -> int:
    from benchmarks.avoidance.crowd import generate_live, parse_seeds, to_scenario_file
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--live", default="hard")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--out", required=True)
    ap.add_argument("--driver", default=None, help="DAgger: a checkpoint that drives while the planner labels")
    ap.add_argument("--obs-version", default="avoid-v2", help="without --driver (with one, the driver's own)")
    a = ap.parse_args(argv)
    seeds = parse_seeds(a.seeds)
    bad = sorted(set(seeds) & FORBIDDEN)
    if bad:
        raise SystemExit(f"refusing: seeds {bad[:5]}... overlap the selection / sealed test seeds")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        to_scenario_file(generate_live(s, a.live), REPO / "data/scenarios/crowd")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"), initializer=_init,
                             initargs=(a.driver, a.obs_version)) as ex:
        res = list(ex.map(_job, [(s, a.live, str(out)) for s in seeds], chunksize=1))
    rows = [r["row"] for r in res]
    summ = {"seeds": a.seeds, "live": a.live, "episodes": len(res), "steps": sum(r["steps"] for r in res),
            "clipped_steps": sum(r["clipped"] for r in res), "wall_s": round(time.time() - t0, 1),
            "completed": sum(1 for r in rows if r.get("completed")),
            "episodes_with_ambient_contact": sum(1 for r in rows if (r.get("locks") or {}).get("ambient_contacts")),
            "obs_version": a.obs_version if a.driver is None else "driver's", "driver": a.driver}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
