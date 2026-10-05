"""Pre-registered checkpoint selection for RL v4 (D46). Uses validation seeds only; the sealed test seeds never
enter this choice.

For each checkpoint's validation run (rl only, seeds 21-60, live hard tier), against the planner's run on the same
seeds:

    score = ambient_contact_runs
          + 40 * max(0, hit_per_commit - planner_hit_per_commit)
          + 40 * max(0, median_time / planner_median_time - 1.05)
          + 10 * (episodes - completed)

Lowest score wins; ties go to the later checkpoint (more training). The same function ranks arms (each arm's selected
score) for the hybrid and for "which pure arm is the demo claim".

    python tools/v4_select.py --scan <bench>/A1-scan --ref <bench>/refs-val --out <arm>/selected.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


def metrics(summary_path: str, controller: str) -> dict:
    s = json.load(open(summary_path))
    r = s["by_controller"][controller]
    lc = r.get("live_crowd", {})
    return {"episodes": r["episodes"], "completed": r["completed"], "time_s_median": r["time_s_median"],
            "hit_per_commit": lc.get("hit_per_commit"), "locks_per_run": lc.get("locks_per_run"),
            "ambient_runs": lc.get("episodes_with_ambient_contact"), "summary": summary_path}


def score(m: dict, ref: dict) -> float:
    hpc = m["hit_per_commit"] if m["hit_per_commit"] is not None else 1.0
    return (m["ambient_runs"] + 40 * max(0.0, hpc - ref["hit_per_commit"])
            + 40 * max(0.0, m["time_s_median"] / ref["time_s_median"] - 1.05) + 10 * (m["episodes"] - m["completed"]))


def latest_summary(d: str) -> str:
    found = sorted(glob.glob(f"{d}/rl-eval-*/summary.json"))
    if not found:
        raise FileNotFoundError(f"no rl-eval-*/summary.json under {d}")
    return found[-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True, help="dir of <ckpt_name>/rl-eval-*/summary.json")
    ap.add_argument("--ref", required=True, help="dir holding the planner's validation rl-eval-*/summary.json")
    ap.add_argument("--run", default=None, help="training run dir with the checkpoints (default: <arm>/run)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ref = metrics(latest_summary(a.ref), "heuristic")
    run = Path(a.run) if a.run else Path(a.out).parent / "run"
    rows = []
    for d in sorted(Path(a.scan).iterdir()):
        if not d.is_dir():
            continue
        m = metrics(latest_summary(str(d)), "rl")
        upd = int(re.search(r"(\d+)", d.name).group(1))
        rows.append({"ckpt": str(run / f"{d.name}.pt"), "name": d.name, "update": upd, **m, "score": round(score(m, ref), 3)})
    if not rows:
        raise SystemExit("no checkpoints scanned")
    best = min(rows, key=lambda r: (r["score"], -r["update"]))
    out = {"rule": "D46 validation score, ties -> later checkpoint", "ref_planner_val": ref, "selected": best["name"],
           "ckpt": best["ckpt"], "score": best["score"], "candidates": rows}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("selected", "ckpt", "score")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
