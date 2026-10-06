"""MORNING_REPORT.md for an overnight GR00T run (tools/overnight_gr00t.sh). Runs in the Isaac-GR00T venv.

  /mnt/weights/ai/nvidia-action/Isaac-GR00T/.venv/bin/python tools/gr00t_morning_report.py --run <run dir>

Headline = STRICT real successes (lift + upright, unmoved bowl + no blow-up + the task's own success term; see
tools/gr00t_eval.py). The FLUX-comparable "task success" (the task term alone, what the FLUX morning report counted)
is reported beside it, with the FLUX night's numbers from the same objects / split / seeds / 12 x 20 s for reference.
Every task-success episode is listed with its strip image: look at them before believing any number.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
args = ap.parse_args()
RUN = Path(args.run).resolve()
REP = RUN / "report"
REP.mkdir(exist_ok=True)
HELDOUT = {"mustard bottle", "cracker box"}
# FLUX night full-20260923-2207 (course Lessons 6.2 / 7.1): reported (task-term) vs real successes after watching
# every "success" episode. Same objects, split, seed 1000 + ep, 12 episodes x 20 s per object.
FLUX_REF = [("FLUX base, training objects", "3/48", "0/48"), ("FLUX base, held-out", "1/24", "0/24"),
            ("FLUX LoRA mid (EMA), training", "2/48", "0/48"), ("FLUX LoRA final (raw), training", "1/48", "0/48"),
            ("FLUX LoRA final (EMA), all 6", "0/72", "0/72")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def frac(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({k / max(1, n):.0%}, CI {lo:.0%}-{hi:.0%})"


cfg = (RUN / "config.txt").read_text().strip() if (RUN / "config.txt").exists() else ""
res = {}  # label -> list of result dicts (one per object)
for f in sorted((RUN / "eval").glob("*/results_*.json")):
    res.setdefault(f.parent.name, []).append(json.loads(f.read_text()))

policy_labels = [k for k in res if k != "replay_expert"]
tot = {lab: (sum(r["summary"]["strict_successes"] for r in res[lab]), sum(r["summary"]["episodes"] for r in res[lab]),
             sum(r["summary"]["successes"] for r in res[lab])) for lab in policy_labels}

L = ["# Morning report: GR00T N1.7 on SO-101 (sim)", "", f"Run: `{RUN}`", "", f"Config: `{cfg}`", ""]
L += ["## Headline: STRICT real successes", ""]
if policy_labels:
    for lab in policy_labels:
        s, n, t = tot[lab]
        L.append(f"- **{lab}: {frac(s, n)} strict real successes** (task-term only: {t}/{n})")
else:
    L.append("- No policy evaluations ran (see Stages).")
L += ["", "Strict = the task's success term fires AND the object was lifted >= 3 cm AND the bowl stayed upright "
      "(<= 20 deg) and within 3 cm of its spawn AND no physics blow-up; layouts with the object starting within "
      "10 cm of the bowl are re-seeded (counted below). Look at every strip before quoting a number.", ""]

L += ["## Stages", "", "| Stage | Status | Note |", "|---|---|---|"]
for f in sorted((RUN / "status").glob("*.json")):
    s = json.loads(f.read_text())
    L.append(f"| {s['stage']} | {s['status']} | {s['note']} |")
L.append("")

if "replay_expert" in res:
    L += ["## S0 · Harness check (recorded expert demos replayed through the strict eval path)", "",
          "| Object | Strict | Task term | Episodes |", "|---|---|---|---|"]
    for r in res["replay_expert"]:
        s = r["summary"]
        L.append(f"| {s['object']} | {s['strict_successes']} | {s['successes']} | {s['episodes']} |")
    L += ["", "Expected: nearly all strict. If not, the criteria or the controller path are wrong and every "
          "policy number below is suspect.", ""]

conv = RUN / "logs" / "convert.log"
if conv.exists():
    keep = [ln for ln in conv.read_text().splitlines() if ln.startswith(("CONVERT", "TIMING", "RAWCHECK"))]
    L += ["## S1 · Dataset", ""] + [f"- `{ln}`" for ln in keep] + [""]

# ---- training curve ----
ft = RUN / "finetune"
states = sorted(ft.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
if states:
    hist = json.loads(states[-1].read_text())["log_history"]
    pts = [(h["step"], h["loss"]) for h in hist if "loss" in h]
    L += ["## S2 · Post-training", ""]
    if pts:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(*zip(*pts), lw=1)
        ax.set_xlabel("step")
        ax.set_ylabel("train loss")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(REP / "training_loss.png", dpi=120)
        first, last = pts[0], pts[-1]
        L += [f"{len(pts)} logged points; loss {first[1]:.4f} (step {first[0]}) -> {last[1]:.4f} (step {last[0]}). "
              f"Checkpoints: {', '.join(p.parent.name for p in states)}.", "", "![training loss](report/training_loss.png)", ""]
    rt = [h for h in hist if "train_runtime" in h]
    if rt:
        L += [f"Train runtime {rt[-1]['train_runtime'] / 3600:.2f} h, {rt[-1].get('train_steps_per_second', 0):.3f} steps/s.", ""]

# ---- per object ----
L += ["## S3 · Closed-loop evaluation (Isaac Lab, GR00T in the loop)", "",
      "| Policy | Object | Split | Strict real | Task term (FLUX-comparable) | Rejected resets | Plan latency median | Start jaw |",
      "|---|---|---|---|---|---|---|---|"]
for lab in policy_labels:
    for r in sorted(res[lab], key=lambda r: (r["summary"]["object"] in HELDOUT, r["summary"]["object"])):
        s = r["summary"]
        ms = f"{s['plan_ms_median']:.0f} ms" if s.get("plan_ms_median") else "-"
        L.append(f"| {lab} | {s['object']} | {'held-out' if s['object'] in HELDOUT else 'train'} | "
                 f"{frac(s['strict_successes'], s['episodes'])} | {s['successes']}/{s['episodes']} | "
                 f"{s['rejected_resets']} | {ms} | {s['start_jaw_units_median']:.1f} |")
L.append("")
L += ["### Split totals", "", "| Policy | Split | Strict real | Task term |", "|---|---|---|---|"]
for lab in policy_labels:
    for split, pick in (("train", lambda o: o not in HELDOUT), ("held-out", lambda o: o in HELDOUT)):
        rs = [r["summary"] for r in res[lab] if pick(r["summary"]["object"])]
        if rs:
            n = sum(s["episodes"] for s in rs)
            L.append(f"| {lab} | {split} | {frac(sum(s['strict_successes'] for s in rs), n)} | "
                     f"{sum(s['successes'] for s in rs)}/{n} |")
L += ["", "### FLUX 3 Action reference (same objects, split, seeds, 12 x 20 s per object)", "",
      "| Policy | Reported (task term) | Real (after watching every success) |", "|---|---|---|"]
L += [f"| {a} | {b} | {c} |" for a, b, c in FLUX_REF]
L += ["", "FLUX's one real success came later, in a diagnostic with the jaw pre-opened (1/6, mug). GR00T evals start "
      "from the demos' start state (jaw ~16 units) by default.", ""]

# ---- every success, for eyes ----
L += ["## Every task-success episode (look at each)", ""]
any_s = False
for lab in policy_labels + (["replay_expert"] if "replay_expert" in res else []):
    for r in res[lab]:
        for e in r["episodes"]:
            if e["success"]:
                any_s = True
                tagname = e["object"].replace(" ", "_")
                kind = "strict" if e["strict_success"] else "task"
                strip = Path("eval") / lab / f"{tagname}_ep{e['episode']:02d}_{kind}_strip.png"
                L.append(f"- {lab} / {e['object']} ep {e['episode']} (seed {e['seed']}): **{kind.upper()}** "
                         f"checks={e['checks']} rise={e['max_rise_m']} m, {e['seconds_sim']} s - `{strip}`")
if not any_s:
    L.append("- none")
L.append("")

# ---- every action dimension, commanded vs measured (the check the FLUX night skipped: Lesson 7.3) ----
import numpy as np  # noqa: E402

JN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
plots = []
for lab in policy_labels:
    diags = sorted((RUN / "eval" / lab).glob("diag_*_ep00.npz"))
    if not diags:
        continue
    d = np.load(diags[0])
    fig, axs = plt.subplots(2, 3, figsize=(11, 5), sharex=True)
    for j, ax in enumerate(axs.flat):
        ax.plot(d["cmd_units"][:, j], lw=1, label="commanded")
        ax.plot(d["meas_units"][:, j], lw=1, label="measured")
        ax.set_title(JN[j], fontsize=9)
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"{lab}: {diags[0].stem} (dataset units, 30 Hz ticks)", fontsize=10)
    fig.tight_layout()
    fig.savefig(REP / f"cmd_vs_meas_{lab}.png", dpi=110)
    plt.close(fig)
    g = d["cmd_units"][:, 5]
    plots.append(f"- {lab}: `report/cmd_vs_meas_{lab}.png`; jaw command p10/p50/p90 = "
                 f"{np.percentile(g, 10):.1f}/{np.percentile(g, 50):.1f}/{np.percentile(g, 90):.1f}, "
                 f"ticks commanding open (>30): {np.mean(g > 30):.0%}; min TCP-object distance "
                 f"{d['tcp_obj_dist'].min() * 100:.1f} cm")
if plots:
    L += ["## Every action dimension (first episode per policy)", ""] + plots + [""]

# ---- latency ----
lat = [(lab, r["summary"]["object"], e["plan_ms_median"], e["plan_ms_p90"]) for lab in policy_labels
       for r in res[lab] for e in r["episodes"] if e.get("plan_ms_median")]
if lat:
    import statistics

    L += ["## Latency", "", f"GR00T plan (one 16-step chunk, {len(lat)} episodes): median of episode medians "
          f"{statistics.median(x[2] for x in lat):.1f} ms, worst episode p90 {max(x[3] for x in lat):.1f} ms. "
          "One plan per exec_horizon ticks (see config).", ""]

(RUN / "MORNING_REPORT.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
