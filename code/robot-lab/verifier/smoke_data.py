"""Tiny MECHANICS-ONLY dataset for smoke-testing train_qlora.py / eval_verifier.py before the sim data exists.

Source: night-2 expert demos (overnight-gr00t-v2 raw npz), episodes NOT used in set_v2.json. Labels are coarse
(phase-based: last frame of a strict-success demo = in bowl; a CARRY-phase frame = held, not in bowl). The adapter
trained on this is discarded; no metric from it is reported as a result.
usage: smoke_data.py <out_dir> [n_eps]
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import QUESTIONS  # noqa: E402

out = Path(sys.argv[1])
n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 12
(out / "img").mkdir(parents=True, exist_ok=True)
used = {x["id"] for x in json.load(open("/mnt/weights/ai/robot-lab-data/vlm-verifier-test/set_v2.json"))}
RAW = Path("/mnt/weights/ai/robot-lab-data/overnight-gr00t-v2/full-20261005-2209/raw")
rng = random.Random(0)
rows, evals = [], []
for obj in ["mug", "blue_block", "soup_can", "sugar_box"]:
    files = sorted(RAW.glob(f"{obj}/ep_*.npz"))
    rng.shuffle(files)
    k = 0
    for f in files:
        if f"demo_{obj}_{f.stem}" in used:
            continue
        d = np.load(f, allow_pickle=True)
        if not bool(d["success"]):
            continue
        ph = d["phase"]
        carry = np.nonzero(ph == 4)[0]
        if len(carry) == 0:
            continue
        o = obj.replace("_", " ")
        for tag, t, lab in (("final", -1, True), ("carry", int(carry[len(carry) // 2]), False)):
            p = out / "img" / f"{obj}_{f.stem}_{tag}.png"
            Image.fromarray(np.concatenate([d["scene"][t], d["wrist"][t]], axis=1)).resize((512, 256)).save(p)
            q = "in_bowl_placed"
            ans = json.dumps({"in_bowl": lab, "reason": f"the {o} is resting inside the bowl and the gripper has let go of it"
                              if lab else f"the gripper is still holding the {o}, which is not in the bowl"})
            rows.append(dict(id=f"{p.stem}|{q}", image=str(p), object=o, question=q, prompt=QUESTIONS[q](o), answer=ans,
                             label=lab, scenario="demo", kind=tag))
            evals.append(dict(id=p.stem, image=str(p), object=o, in_bowl=lab, lifted=True))
        k += 1
        if k >= n_eps // 4:
            break
rng.shuffle(rows)
n_val = max(4, len(rows) // 6)
(out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[n_val:]))
(out / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:n_val]))
json.dump(evals[:8], open(out / "smoke_eval.json", "w"), indent=1)
print(f"smoke rows train={len(rows) - n_val} val={n_val} eval={min(8, len(evals))} -> {out}")
