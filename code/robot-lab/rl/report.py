"""Report for one RL run: training curves (strict / task / lifted / released per wave, return, PPO stats, throughput)
and the paired sealed-eval table. Strict successes are the only headline.

  /mnt/weights/ai/isaac/IsaacLab/.venv/bin/python /mnt/work/AI/robot-lab/rl/report.py <run_dir>
Writes <run_dir>/REPORT.md and <run_dir>/curves.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # frozen copies import their own gr00t_rl
from gr00t_rl.common import wilson  # noqa: E402

RUN = Path(sys.argv[1])
rows = [json.loads(x) for x in (RUN / "train_log.jsonl").read_text().splitlines()] if (RUN / "train_log.jsonl").exists() else []
lines = [f"# GR00T + residual RL run report\n", f"Run: `{RUN}`\n"]
cfg = RUN / "config.txt"
if cfg.exists():
    lines.append("```\n" + cfg.read_text().strip() + "\n```\n")

if rows:
    gw = [r["global_wave"] for r in rows]
    n = [r["valid_layouts"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for k, c in (("strict_valid", "C0"), ("task", "C1"), ("lifted", "C2"), ("released_over", "C3"), ("over_bowl", "C4")):
        ax[0, 0].plot(gw, [r[k] / max(1, r["n_envs"]) for r in rows], ".-", color=c, label=k)
    ax[0, 0].set_title("per training wave (stochastic policy), fraction of envs")
    ax[0, 0].set_xlabel("global wave")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(gw, [r["return_mean"] for r in rows], ".-", label="return mean")
    ax[0, 1].plot(gw, [r["p_release_mean"] * 10 for r in rows], ".-", label="p(release) x10, all decisions")
    ax[0, 1].plot(gw, [r.get("p_release_over_bowl", float("nan")) * 10 for r in rows], ".-",
                  label="p(release) x10, lifted object over bowl")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(gw, [r["ppo"]["kl"] for r in rows], ".-", label="kl")
    ax[1, 0].plot(gw, [r["ppo"]["clipfrac"] for r in rows], ".-", label="clipfrac")
    ax[1, 0].plot(gw, [sum(r["log_std"]) / len(r["log_std"]) for r in rows], ".-", label="mean log_std")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(gw, [r["timing"]["env_ticks_per_s"] for r in rows], ".-", label="env ticks/s")
    ax[1, 1].plot(gw, [r["timing"]["rl_steps_per_s"] * 10 for r in rows], ".-", label="RL steps/s x10")
    ax[1, 1].plot(gw, [r["mem"]["rss_gb"] for r in rows], ".-", label="trainer RSS GB")
    ax[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RUN / "curves.png", dpi=90)
    lines.append("## Training (stochastic exploration policy; NOT the headline)\n")
    lines.append(f"{len(rows)} waves, {sum(r['transitions'] for r in rows)} RL transitions, "
                 f"{sum(r['n_envs'] for r in rows)} episodes. ![curves](curves.png)\n")
    lines.append("| global wave | object | strict/valid | task | lifted | over bowl | released over | bowl bad | return | p(rel) all / over bowl | wave s |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['global_wave']} | {r['object']} | {r['strict_valid']}/{r['valid_layouts']} | {r['task']} | "
                     f"{r['lifted']} | {r['over_bowl']} | {r['released_over']} | {r['bowl_bad']} | "
                     f"{r['return_mean']:.2f} | {r['p_release_mean']:.3f} / {r.get('p_release_over_bowl', float('nan')):.3f} | "
                     f"{r['timing']['wave_s']} |")
    lines.append("")

ev = sorted((RUN / "eval").glob("results_*.json")) if (RUN / "eval").exists() else []
if ev:
    lines.append("## Evaluation (paired: identical layouts + GR00T noise per arm; RL arm deterministic) -- HEADLINE\n")
    lines.append("| object | split | seeds | base strict (; of which object rests in bowl 1 s later) | RL strict (; rested) | paired both / base-only / RL-only / neither | "
                 "base released-over | RL released-over | invalid layouts |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    tot = {"train": {"base": [0, 0], "rl": [0, 0]}, "held-out": {"base": [0, 0], "rl": [0, 0]}}
    for f in ev:
        s = json.loads(f.read_text())["summary"]
        split = "held-out" if s["object"] in ("mustard bottle", "cracker box") else "train"
        a = s["arms"]
        cell = {}
        for arm in ("base", "rl"):
            if arm in a:
                k, v = a[arm]["strict"], a[arm]["valid"]
                lo, hi = wilson(k, v)
                cell[arm] = (f"{k}/{v} ({100 * k / max(1, v):.0f}%, CI {max(0.0, 100 * lo):.0f}-{100 * hi:.0f}%); "
                             f"rested {a[arm].get('strict_rested', '-')}")
                tot[split][arm][0] += k
                tot[split][arm][1] += v
            else:
                cell[arm] = "-"
        p = s.get("paired", {})
        lines.append(f"| {s['object']} | {split} | {s['seeds']} | {cell['base']} | {cell['rl']} | "
                     f"{p.get('both', '-')} / {p.get('base_only', '-')} / {p.get('rl_only', '-')} / {p.get('neither', '-')} | "
                     f"{a.get('base', {}).get('released_over', '-')} | {a.get('rl', {}).get('released_over', '-')} | "
                     f"{a.get('base', a.get('rl', {})).get('invalid_layouts', '-')} |")
    lines.append("")
    for split, d in tot.items():
        parts = []
        for arm, (k, v) in d.items():
            if v:
                lo, hi = wilson(k, v)
                parts.append(f"{arm} {k}/{v} ({100 * k / v:.0f}%, CI {max(0.0, 100 * lo):.0f}-{100 * hi:.0f}%)")
        if parts:
            lines.append(f"- **{split}**: " + "; ".join(parts))
    lines.append("\nLook at the videos/strips in eval/ before quoting any number.\n")
(RUN / "REPORT.md").write_text("\n".join(lines) + "\n")
print(f"wrote {RUN / 'REPORT.md'}")
