"""Night-2 vs night-1 GR00T comparison; appends a section to <run>/MORNING_REPORT.md (GR00T venv).

  .venv/bin/python tools/gr00t_compare_v2.py --run <night-2 run> --night1 <night-1 run>

Per run x policy (gr00t_mid / gr00t_final) x split, from eval/<policy>/results_*.json + diag_*.npz:
  strict, task-term, lifted (object centre rose >= 3 cm), lifted-not-placed (lifted and no task success),
  open-after-lift (gripper command > 30 units at some tick after the lift), and lifted-not-placed episodes that
  NEVER commanded open after the lift (night 1: 48 of 59, the failure this night targets).
Also the paired subset: episodes 0-11 (seeds 1000-1011, the only seeds night 1 ran), and demo release-frame counts
(frames after the last closed gripper command whose command is open) for both training sets.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--night1", required=True)
args = ap.parse_args()
RUN, N1 = Path(args.run), Path(args.night1)
HELDOUT = {"mustard bottle", "cracker box"}
OPEN = 30.0


def episodes(run, label, max_ep=None):
    rows = []
    for f in sorted((run / "eval" / label).glob("results_*.json")):
        for e in json.loads(f.read_text())["episodes"]:
            if max_ep is not None and e["episode"] >= max_ep:
                continue
            tagname = e["object"].replace(" ", "_")
            dpath = run / "eval" / label / f"diag_{tagname}_ep{e['episode']:02d}.npz"
            opened = None
            if e["lifted_step"] is not None and dpath.exists():
                cmd = np.load(dpath)["cmd_units"][:, 5]
                opened = bool((cmd[e["lifted_step"] + 1:] > OPEN).any())
            rows.append({"object": e["object"], "strict": e["strict_success"], "task": e["success"],
                         "lifted": e["checks"]["lifted"], "open_after_lift": opened})
    return rows


def summarize(rows):
    n = len(rows)
    lnp = [r for r in rows if r["lifted"] and not r["task"]]
    return {"n": n, "strict": sum(r["strict"] for r in rows), "task": sum(r["task"] for r in rows),
            "lifted": sum(r["lifted"] for r in rows), "lnp": len(lnp),
            "open_after_lift": sum(bool(r["open_after_lift"]) for r in rows if r["lifted"]),
            "lnp_never_open": sum(r["open_after_lift"] is False for r in lnp)}


def table(title, max_ep=None):
    L = [f"### {title}", "", "| Night | Policy | Split | Strict | Task term | Lifted | Lifted-not-placed | "
         "Open after lift (of lifted) | LNP that never opened |", "|---|---|---|---|---|---|---|---|---|"]
    for night, run in (("1", N1), ("2", RUN)):
        for label in ("gr00t_mid", "gr00t_final"):
            rows = episodes(run, label, max_ep)
            for split, pick in (("train", lambda o: o not in HELDOUT), ("held-out", lambda o: o in HELDOUT)):
                s = summarize([r for r in rows if pick(r["object"])])
                if s["n"]:
                    L.append(f"| {night} | {label} | {split} | {s['strict']}/{s['n']} | {s['task']}/{s['n']} | "
                             f"{s['lifted']}/{s['n']} | {s['lnp']} | {s['open_after_lift']}/{s['lifted']} | "
                             f"{s['lnp_never_open']}/{s['lnp']} |")
    return L + [""]


def per_object(run, label):
    L = []
    rows = episodes(run, label)
    for obj in sorted({r["object"] for r in rows}, key=lambda o: (o in HELDOUT, o)):
        s = summarize([r for r in rows if r["object"] == obj])
        L.append(f"| {label} | {obj} | {s['strict']}/{s['n']} | {s['task']}/{s['n']} | {s['lifted']}/{s['n']} | "
                 f"{s['lnp']} | {s['lnp_never_open']}/{s['lnp']} |")
    return L


def release_frames_night1(ds):
    out = []
    for f in sorted(glob.glob(f"{ds}/data/chunk-*/episode_*.parquet")):
        g = np.stack(pd.read_parquet(f, columns=["action"])["action"].to_numpy())[:, 5]
        closed = np.where(g < OPEN)[0]
        out.append(int((g[closed.max() + 1:] > OPEN).sum()) if len(closed) else 0)
    return out


L = ["", "## Night 2 vs night 1", ""]
L += table("All evaluated episodes (night 1: 12/object, night 2: 24/object)")
L += table("Paired seeds only (episodes 0-11 = seeds 1000-1011 per object)", max_ep=12)
L += ["### Night 2 per object", "", "| Policy | Object | Strict | Task | Lifted | LNP | LNP never opened |",
      "|---|---|---|---|---|---|---|"]
for label in ("gr00t_mid", "gr00t_final"):
    L += per_object(RUN, label)
L.append("")
r1 = release_frames_night1(N1 / "gr00t_ds")
rs2 = RUN / "gr00t_ds" / "release_stats.json"
L += ["### Release frames in the training demos", "",
      "Frames after the last closed gripper command whose command is open (> 30 units).", "",
      "| Training set | Episodes | Median | Min | Max |", "|---|---|---|---|---|",
      f"| night 1 (FLUX-era demos, final frame dropped) | {len(r1)} | {np.median(r1):.0f} | {min(r1)} | {max(r1)} |"]
if rs2.exists():
    s2 = json.loads(rs2.read_text())
    a = s2["all"]
    L.append(f"| night 2 (release tail) | {a['episodes']} | {a['median']:.0f} | {a['min']} | {a['max']} |")
    L += ["", "Night-2 per object: " + ", ".join(f"{k} median {v['median']:.0f} ({v['episodes']} demos)"
                                               for k, v in s2["per_object"].items())]
L.append("")
with open(RUN / "MORNING_REPORT.md", "a") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))
