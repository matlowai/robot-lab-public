"""Evaluate an RL checkpoint on the sim2d patrol benchmark, side by side with the other contenders.

    PYTHONPATH=. python -m benchmarks.avoidance.rl.eval --ckpt <run>/latest.pt --seeds 1-200 \
        --controllers rl,control,heuristic --workers 48

    ... --live hard   # closed-loop crowds (SCN-LC-*, pedestrians.py) of that crowd.LIVE_TIERS tier instead

Writes data/benchmarks/rl-eval-<stamp>/{episodes.jsonl, summary.json} (summary via sim2d.summarize; live runs add
a live_crowd block per controller: locks, commits, hits, hit_per_commit).
--trace-dir DIR (opt-in, off by default) also writes one per-step trace per episode, DIR/<controller>-<seed>.npz
(sim2d.Trace), for contact forensics (tools/contact_forensics.py). Episodes and summary are unchanged by it.
Each worker process registers the checkpoint itself (spawned workers don't inherit the registration).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmarks.avoidance import sim2d
from benchmarks.avoidance.crowd import LIVE_TIERS, generate, generate_live, parse_seeds, to_scenario_file

REPO = Path(__file__).resolve().parents[3]


def _init(ckpt: str, velocity_source: str) -> None:
    from benchmarks.avoidance.rl.controller import register_rl
    from benchmarks.avoidance.rl.hybrid import register_shield
    register_rl(ckpt, velocity_source=velocity_source)
    register_shield(ckpt, velocity_source=velocity_source)  # "rl_shield": the named hybrid arm (D46), opt-in by name


def _job(args):
    seed, ctrl, live, trace_dir = args
    trace = str(Path(trace_dir) / f"{ctrl}-{seed:03d}.npz") if trace_dir else None
    return sim2d.run_episode(seed, ctrl, live=live, trace_path=trace)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", default="1-200")
    ap.add_argument("--controllers", default="rl,control,heuristic")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--velocity-source", default="command", choices=("command", "pose"))
    ap.add_argument("--out", default=str(REPO / "data/benchmarks"))
    ap.add_argument("--live", default=None, choices=sorted(LIVE_TIERS),
                    help="closed-loop crowd tier (SCN-LC-*) instead of the keyframed crowds")
    ap.add_argument("--trace-dir", default=None,
                    help="opt-in: write a per-step trace per episode here (<controller>-<seed>.npz), for forensics")
    a = ap.parse_args(argv)
    workers = min(a.workers, 48)
    seeds, ctrls = parse_seeds(a.seeds), a.controllers.split(",")
    for s in seeds:  # write scenario files once, before workers race to create them
        to_scenario_file(generate_live(s, a.live) if a.live else generate(s), REPO / "data/scenarios/crowd")
    trace_dir = str(Path(a.trace_dir).resolve()) if a.trace_dir else None
    jobs = [(s, c, a.live, trace_dir) for s in seeds for c in ctrls]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init,
                             initargs=(str(Path(a.ckpt).resolve()), a.velocity_source)) as ex:
        rows = list(ex.map(_job, jobs, chunksize=1))
    out = Path(a.out) / f"rl-eval-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {"ckpt": str(Path(a.ckpt).resolve()), "seeds": a.seeds, "controllers": ctrls,
               "velocity_source": a.velocity_source, "live": a.live, "wall_s": round(time.time() - t0, 1),
               "by_controller": sim2d.summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"OUT {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
