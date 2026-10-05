"""RL v4 (D46) morning report: every arm, the planner and the hybrid on the same tables, the pre-registered win check,
and the speed-channel diagnostics. Writes <root>/MORNING_REPORT.md and <root>/report.json. Safe to run at any time
(missing pieces are reported as missing, never guessed).

    python tools/v4_report.py [--root /mnt/weights/ai/patrol-lab-data/rl/v4]
    python tools/v4_report.py --root .../rl/v5 --arms .../rl/v5/arms.json --protocol D48 --title "RL v5: report"
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from pathlib import Path

ARMS = {
    "A0": "baseline: tanh Gaussian, hard-dense, no space term (matched to A1 except the distribution)",
    "A1": "speed-channel fix only: clipped Gaussian + bounds loss + std ceiling 1",
    "A2": "A1 + yield-early reward (time to collision, 3 s horizon)",
    "A3": "A2 settings, warm-started from a behaviour clone of the planner (seeds 1000-1599)",
}
WIN = "completed >= 79/80, ambient-contact runs < 27/80, hit/commit <= 0.385, median time <= 396 s"


def summ(path_glob: str, ctrl: str):
    found = sorted(glob.glob(path_glob))
    if not found:
        return None
    s = json.load(open(found[-1]))
    r = s["by_controller"].get(ctrl)
    if not r:
        return None
    lc = r.get("live_crowd", {})
    out = {"episodes": r["episodes"], "completed": r["completed"], "time": r["time_s_median"],
           "hpc": lc.get("hit_per_commit"), "locks": lc.get("locks_per_run"),
           "ambient": lc.get("episodes_with_ambient_contact"), "closest": r.get("closest_m_median"),
           "statuses": r.get("statuses"), "src": found[-1]}
    return out


def win(m) -> str:
    if not m:
        return "n/a"
    ok = (m["completed"] >= 79 and m["ambient"] is not None and m["ambient"] < 27 and m["hpc"] is not None
          and m["hpc"] <= 0.385 and m["time"] <= 396.0)
    return "**WIN**" if ok else "no"


def fmt(m):
    if not m:
        return "– | – | – | – | –"
    hpc = f"{m['hpc']:.3f}" if m["hpc"] is not None else "–"
    locks = f"{m['locks']:.1f}" if m["locks"] is not None else "–"
    return f"{m['completed']}/{m['episodes']} | {m['ambient']}/{m['episodes']} | {hpc} | {locks} | {m['time']:.0f}"


def train_tail(run: Path) -> dict:
    p = run / "train.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return {}
    last = rows[-50:]
    avg = lambda k: round(sum(float(r[k]) for r in last if r.get(k) not in (None, "", "nan")) / len(last), 4)  # noqa: E731
    out = {"updates": len(rows), "env_steps_B": round(int(rows[-1]["env_steps"]) / 1e9, 2)}
    for k in ("success_rate", "contact_rate", "hit_per_commit", "std_vx", "vx_mid_frac", "mu_vx_sat", "ttc_frac", "fps"):
        if k in rows[-1]:
            try:
                out[k] = avg(k)
            except (ValueError, ZeroDivisionError):
                pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/weights/ai/patrol-lab-data/rl/v4")
    ap.add_argument("--arms", default=None, help="JSON file {arm: description} (default: the D46 arms)")
    ap.add_argument("--protocol", default="D46")
    ap.add_argument("--title", default="RL v4 overnight: morning report")
    a = ap.parse_args(argv)
    arms = json.load(open(a.arms)) if a.arms else ARMS
    root = Path(a.root)
    bench = root / "bench"
    rep = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "win_rule": WIN, "arms": {}}
    planner_val = summ(f"{bench}/refs-val/rl-eval-*/summary.json", "heuristic")
    planner_ho = summ(f"{bench}/refs-heldout/rl-eval-*/summary.json", "heuristic")
    control_ho = summ(f"{bench}/refs-heldout/rl-eval-*/summary.json", "control")
    L = [f"# {a.title}", "", f"Generated {rep['generated']}. Protocol: DECISIONS.md {a.protocol}.", "",
         f"**Win rule (pre-registered, sealed seeds 121-200):** {WIN}.", "",
         "Columns: completed | runs with an ordinary-person contact | hits per committed approach | lock-ons per run | "
         "median patrol time (s).", ""]
    L += ["## Sealed test seeds 121-200", "", "| contender | what | completed | ambient-contact runs | hit/commit | "
          "locks/run | median s | win? |", "|---|---|---|---|---|---|---|---|"]
    L.append(f"| planner | hand-written | {fmt(planner_ho)} | (the bar) |")
    L.append(f"| control | no avoidance | {fmt(control_ho)} | – |")
    best_pure = None
    for arm, what in arms.items():
        sel = root / arm / "selected.json"
        selj = json.load(open(sel)) if sel.exists() else None
        ho = summ(f"{bench}/{arm}-heldout/rl-eval-*/summary.json", "rl")
        st = json.load(open(root / arm / "status.json")) if (root / arm / "status.json").exists() else None
        rep["arms"][arm] = {"what": what, "selected": selj and {k: selj[k] for k in ("selected", "ckpt", "score")},
                            "heldout": ho, "status": st, "train": train_tail(root / arm / "run")}
        L.append(f"| {arm} | {what} | {fmt(ho)} | {win(ho)} |")
        if selj and ho and (best_pure is None or selj["score"] < best_pure[0]):
            best_pure = (selj["score"], arm)
    h_ho = summ(f"{bench}/H-heldout/rl-eval-*/summary.json", "rl_shield")
    hsel = json.load(open(root / "H" / "selected.json")) if (root / "H" / "selected.json").exists() else None
    L.append(f"| H (hybrid) | RL + planner-tracker safety shield on {hsel['from_arm'] if hsel else '?'} | {fmt(h_ho)} | "
             f"{win(h_ho)} (hybrid) |")
    rep["hybrid"] = {"selected": hsel, "heldout": h_ho}
    L += ["", f"**Demo claim (pre-registered: the pure arm with the best validation score):** "
          f"{best_pure[1] + ' → ' + win(rep['arms'][best_pure[1]]['heldout']) if best_pure else 'not available yet'}.", ""]
    if h_ho and h_ho.get("statuses"):
        st = h_ho["statuses"]
        tot = sum(st.values()) or 1
        L.append(f"Shield interventions on the sealed seeds: {st.get('shield', 0)} of {tot} steps "
                 f"({100 * st.get('shield', 0) / tot:.1f}%).")
        L.append("")
    L += ["## Validation seeds 21-60 (selection data)", "", "| contender | selected ckpt | score | completed | ambient | "
          "hit/commit | locks | median s |", "|---|---|---|---|---|---|---|---|"]
    L.append(f"| planner | – | – | {fmt(planner_val)} |")
    for arm in arms:
        selj = rep["arms"][arm]["selected"]
        if selj:
            sv = json.load(open(root / arm / "selected.json"))
            c = next(x for x in sv["candidates"] if x["name"] == sv["selected"])
            m = {"episodes": c["episodes"], "completed": c["completed"], "ambient": c["ambient_runs"],
                 "hpc": c["hit_per_commit"], "locks": c["locks_per_run"], "time": c["time_s_median"]}
            L.append(f"| {arm} | {sv['selected']} | {sv['score']} | {fmt(m)} |")
        else:
            L.append(f"| {arm} | – | – | – | – | – | – | – |")
    hv = summ(f"{bench}/H-val/rl-eval-*/summary.json", "rl_shield")
    L.append(f"| H (hybrid) | (from best arm) | – | {fmt(hv)} |")
    bc = summ(f"{bench}/A3-bc-val/rl-eval-*/summary.json", "rl")
    if bc:
        L.append(f"| A3's BC clone, before PPO | bc.pt | – | {fmt(bc)} |")
    L += ["", "## Training diagnostics (mean of the last 50 updates)", "",
          "| arm | updates | env steps (B) | success | contact | std vx | vx intermediate | vx mean saturated | "
          "ttc frac | fps | status |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        t, st = rep["arms"][arm]["train"], rep["arms"][arm]["status"]
        g = lambda k: t.get(k, "–")  # noqa: E731
        L.append(f"| {arm} | {g('updates')} | {g('env_steps_B')} | {g('success_rate')} | {g('contact_rate')} | "
                 f"{g('std_vx')} | {g('vx_mid_frac')} | {g('mu_vx_sat')} | {g('ttc_frac')} | {g('fps')} | "
                 f"{(st or {}).get('stage', '–')}: {(st or {}).get('state', '–')} |")
    L += ["", "D45 diagnostic gate: std off the clamp and >= 5 % intermediate vx. Behavioural success needs the held-out "
          "table above as well (review 2026-09-26).", ""]
    (root / "MORNING_REPORT.md").write_text("\n".join(L) + "\n")
    (root / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
