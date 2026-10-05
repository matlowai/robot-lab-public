"""Build MORNING_REPORT.md + charts + videos from an overnight run directory (flux-action venv: numpy/matplotlib/ffmpeg).

  python tools/morning_report.py --run /mnt/weights/ai/robot-lab-data/overnight/latest-full
Outputs go to <run>/report/ and are also copied to /mnt/work/AI/robot-lab/course/media/overnight/<run-name>/.
"""

import argparse
import glob
import json
import math
import re
import shutil
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
args = ap.parse_args()
RUN = Path(args.run).resolve()
REP = RUN / "report"
REP.mkdir(exist_ok=True)
COURSE = Path("/mnt/work/AI/robot-lab/course/media/overnight") / RUN.name


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


lines = ["# Morning report", "", f"Run: `{RUN}`", ""]

# ---- stage status ----
lines += ["## Stages", "", "| Stage | Status | Note |", "|---|---|---|"]
for f in sorted((RUN / "status").glob("*.json")):
    s = json.loads(f.read_text())
    lines.append(f"| {s['stage']} | {s['status']} | {s['note']} |")
lines.append("")

# ---- demos ----
lines += ["## S1 · Scripted demos", "", "| Object | Successes | Episodes | Rate |", "|---|---|---|---|"]
for log in sorted((RUN / "logs").glob("demos_*.log")):
    eps = [ln for ln in log.read_text(errors="ignore").splitlines() if ln.startswith("EPISODE")]
    ok = sum("success=True" in ln for ln in eps)
    lines.append(f"| {log.stem[6:].replace('_', ' ')} | {ok} | {len(eps)} | {ok / max(1, len(eps)):.0%} |")
lines.append("")

# ---- dataset ----
units = RUN / "lerobot_ds" / "units.json"
if units.exists():
    u = json.loads(units.read_text())
    lines += ["## S2 · Dataset", "", f"{u['episodes']} episodes, {u['frames']} frames at 30 Hz. Tasks:", ""]
    lines += [f"- {k}: {v}" for k, v in u["tasks"].items()]
    lines.append("")

# ---- training curves ----
lora_log = RUN / "logs" / "lora.log"
if lora_log.exists():
    txt = lora_log.read_text(errors="ignore").replace("\r", "\n")
    rows = re.findall(r"step:(\d+)\S* .*?loss:([\d.]+).*?video_mse:([\d.]+) action_mse:([\d.]+)", txt)
    evals = re.findall(r"step (\d+): eval_loss=([\d.]+)", txt)
    mem = re.findall(r"mem_gb:([\d.]+)", txt)
    spd = re.findall(r"updt_s:([\d.]+)", txt)
    if rows:
        a = np.array([[float(x) for x in r] for r in rows])
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        def smooth(y, k=15):
            return np.convolve(y, np.ones(k) / k, mode="valid") if len(y) > k else y

        for i, (name, col) in enumerate([("loss", 1), ("action_mse", 3)]):
            ax[i].plot(a[:, 0], a[:, col], alpha=0.25, color="#1B7373")
            s = smooth(a[:, col])
            ax[i].plot(a[len(a) - len(s):, 0], s, color="#1B7373", lw=2, label=f"train {name}")
            ax[i].set_title(name)
            ax[i].set_xlabel("micro-step")
            ax[i].grid(alpha=0.3)
        ax[0].plot(a[:, 0], a[:, 2], alpha=0.25, color="#D9A40E")
        s = smooth(a[:, 2])
        ax[0].plot(a[len(a) - len(s):, 0], s, color="#D9A40E", lw=2, label="train video_mse")
        if evals:
            e = np.array([[float(x) for x in r] for r in evals])
            ax[0].plot(e[:, 0], e[:, 1], "ko-", label="held-out eval_loss")
        ax[0].legend()
        fig.tight_layout()
        fig.savefig(REP / "training_curves.png", dpi=90)
        plt.close(fig)
        lines += ["## S3 · LoRA training", "",
                  f"{int(a[-1, 0])} micro-steps logged; final smoothed loss {smooth(a[:, 1])[-1]:.3f}, "
                  f"action_mse {smooth(a[:, 3])[-1]:.3f}, video_mse {smooth(a[:, 2])[-1]:.3f}. "
                  f"Update time median {np.median([float(x) for x in spd]):.2f} s/micro-step; peak mem {max(float(m) for m in mem):.1f} GB.",
                  "", "![training curves](report/training_curves.png)", ""]
        if evals:
            lines += ["Held-out eval loss: " + ", ".join(f"step {s}: {v}" for s, v in evals), ""]

# ---- closed-loop evals ----
table, per_label = [], {}
for d in sorted((RUN / "eval").iterdir()) if (RUN / "eval").exists() else []:
    for f in sorted(d.glob("results_*.json")):
        r = json.loads(f.read_text())
        s = r["summary"]
        lat = [e["plan_ms_median"] for e in r["episodes"] if e.get("plan_ms_median")]
        k, n = s["successes"], s["episodes"]
        lo, hi = wilson(k, n)
        table.append((d.name, s["object"], k, n, lo, hi, np.median(lat) if lat else float("nan")))
        per_label.setdefault(d.name, [0, 0])
        per_label[d.name][0] += k
        per_label[d.name][1] += n
if table:
    lines += ["## S4 · Closed-loop evaluation (Isaac Lab, FLUX policy in the loop)", "",
              "| Policy | Object | Success | 95% CI (Wilson) | Plan latency (median) |", "|---|---|---|---|---|"]
    for lab, obj, k, n, lo, hi, lat in table:
        lines.append(f"| {lab} | {obj} | {k}/{n} | {lo:.0%}–{hi:.0%} | {lat / 1000:.2f} s |")
    lines += ["", "| Policy | Overall |", "|---|---|"]
    for lab, (k, n) in per_label.items():
        lo, hi = wilson(k, n)
        lines.append(f"| {lab} | {k}/{n} ({k / max(1, n):.0%}, CI {lo:.0%}–{hi:.0%}) |")
    lines.append("")
    # chart
    labs = list(per_label)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    rates = [per_label[x][0] / max(1, per_label[x][1]) for x in labs]
    cis = [wilson(*per_label[x]) for x in labs]
    ax.barh(labs, rates, color="#1B7373", xerr=[[r - c[0] for r, c in zip(rates, cis)], [c[1] - r for r, c in zip(rates, cis)]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("success rate (95% CI)")
    fig.tight_layout()
    fig.savefig(REP / "eval_success.png", dpi=90)
    plt.close(fig)
    lines += ["![eval success](report/eval_success.png)", ""]

# ---- videos: one success + one failure per policy label ----
vids = []
for d in sorted((RUN / "eval").iterdir()) if (RUN / "eval").exists() else []:
    for kind in ("ok", "fail"):
        cands = sorted(d.glob(f"*_{kind}.npy")) + sorted(d.glob(f"*_{kind}.mp4"))
        if not cands:
            continue
        src = cands[0]
        dst = REP / f"{d.name}__{src.stem}.mp4"
        if src.suffix == ".npy":
            iio.imwrite(dst, np.load(src), fps=30, codec="libx264")
        else:
            shutil.copy(src, dst)
        vids.append(dst)
if vids:
    lines += ["## Rollout videos (scene | wrist)", ""] + [f"- `{v.name}`" for v in vids] + [""]

lines += ["## What to look at first", "",
          "1. The overall success table: does the LoRA beat the base policy on training objects? On held-out objects?",
          "2. Training curves: did action_mse keep falling, and does held-out eval loss agree?",
          "3. A failure video per policy: what goes wrong (reach, grasp, release, drift)?", ""]
(RUN / "MORNING_REPORT.md").write_text("\n".join(lines))
COURSE.mkdir(parents=True, exist_ok=True)
for f in REP.iterdir():
    shutil.copy(f, COURSE / f.name)
shutil.copy(RUN / "MORNING_REPORT.md", COURSE / "MORNING_REPORT.md")
print("REPORT", RUN / "MORNING_REPORT.md")
