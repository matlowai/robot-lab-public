"""Label the generator's raw snapshots from sim state and build the verifier datasets (CPU only).

usage: build_dataset.py <raw_root> <out_dir> [--val_frac 0.1]
  raw_root/<object>/episodes.jsonl + img/*.png + meta.json   (from gen_verifier_data.py)
Outputs in out_dir:
  frames.jsonl            every snapshot with derived labels + raw state (audit trail)
  train.jsonl / val.jsonl one record per (image, question) training sample: messages-ready prompt + JSON answer
  test_heldout_final.json run_test.py-compatible set (held-out objects, FINAL frames after retreat + settle)
  test_heldout_release.json  same objects, frames at the instant of release (the ambiguous set.json-v1 style)
  val_final.json          run_test-compatible in-distribution set (train objects, held-out episodes)
  pairs.json              before/after image pairs per episode with the episode outcome
  stats.json              class balance per split / question / scenario / object; label thresholds
  contact_*.png           thumbnail grids per final-frame class, for eyeballing labels

Label rules (thresholds are module constants and are copied into stats.json):
  held         grip commanded closed AND object centre within HELD_DIST of the TCP AND (jaw blocked by the object
               OR object raised above its rest height). jaw blocked = jaw angle > (empty-closed jaw angle + margin),
               the empty-closed angle measured from this data (closed frames with the object far from the TCP).
  in_bowl_geom object centre within BOWL_RADIUS of the bowl axis (xy) and < IN_BOWL_DZ above the bowl root
               (the env's success geometry, robot_lab.tasks.so101_pick_place.object_in_bowl).
  bowl_upright bowl tilt <= 20 deg (tools/gr00t_eval.py strict threshold).
  in_bowl      ("placed") in_bowl_geom AND bowl upright AND NOT held AND object speed < SETTLED_V.
  over_bowl    held AND within BOWL_RADIUS + 1 cm of the bowl axis AND centre > OVER_DZ above the bowl root.
  on_table     NOT held AND NOT in_bowl_geom AND centre within [-5 cm, +1.5 cm] of its rest height.
  ambiguous    object in the bowl footprint but still moving (> SETTLED_V) -> excluded from in_bowl questions.
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BOWL_RADIUS = 0.055
IN_BOWL_DZ = 0.07
BOWL_TILT_MAX = 20.0
SETTLED_V = 0.10
HELD_DIST = 0.05
JAW_MARGIN = 0.08
RAISED = 0.015
OVER_DZ = 0.04
HELDOUT = {"mustard bottle", "cracker box"}

HEAD = ("Robot task: put the {o} in the yellow bowl. The image shows the end of the attempt: left half is the "
        "scene camera, right half is the robot's wrist camera. ")
# The two in_bowl wordings are copied verbatim from vlm-verifier-test/run_test.py so training matches the eval.
Q_PLACED = ("Question: has the {o} been placed into the yellow bowl, i.e. it is down inside the bowl and the "
            "gripper has opened, even if the gripper fingers are still around it? Answer false if it is on the table, "
            "lifted above the bowl, outside the bowl, or the bowl is knocked over. ")
Q_VISUAL = "Question: is the {o} now inside the yellow bowl (not held above it, not on the table)? "
TAIL_IN_BOWL = 'Reply with only JSON: {"in_bowl": true or false, "reason": "<short>"}'
SKILL_HEAD = ("Robot skill: {s}. The image shows the end of the skill: left half is the scene camera, right half is "
              "the robot's wrist camera. ")
QUESTIONS = {
    "in_bowl_placed": lambda o: HEAD.format(o=o) + Q_PLACED.format(o=o) + TAIL_IN_BOWL,
    "in_bowl_visual": lambda o: HEAD.format(o=o) + Q_VISUAL.format(o=o) + TAIL_IN_BOWL,
    "held": lambda o: SKILL_HEAD.format(s=f"pick up the {o}") + f"Question: is the {o} now held in the gripper and "
    "lifted clear of the table? " + 'Reply with only JSON: {"held": true or false, "reason": "<short>"}',
    "over_bowl": lambda o: SKILL_HEAD.format(s=f"move the {o} over the yellow bowl") + f"Question: is the {o} held "
    "in the gripper directly above the yellow bowl? " + 'Reply with only JSON: {"over_bowl": true or false, '
    '"reason": "<short>"}',
    "bowl_upright": lambda o: HEAD.format(o=o) + "Question: is the yellow bowl still upright (not tipped or knocked "
    "over)? " + 'Reply with only JSON: {"bowl_upright": true or false, "reason": "<short>"}',
    "on_table": lambda o: HEAD.format(o=o) + f"Question: is the {o} resting on the table (not held by the gripper and "
    "not in the bowl)? " + 'Reply with only JSON: {"on_table": true or false, "reason": "<short>"}',
}
ANSWER_KEY = {"in_bowl_placed": "in_bowl", "in_bowl_visual": "in_bowl", "held": "held_lifted",
              "over_bowl": "over_bowl", "bowl_upright": "bowl_upright", "on_table": "on_table"}
JSON_KEY = {"in_bowl_placed": "in_bowl", "in_bowl_visual": "in_bowl", "held": "held", "over_bowl": "over_bowl",
            "bowl_upright": "bowl_upright", "on_table": "on_table"}


def tilt_deg(q):
    x, y, z, w = q
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def label(snap, ep, jaw_empty):
    s = snap["state"]
    o, b, tcp = np.array(s["obj"]), np.array(s["bowl"]), np.array(s["tcp"])
    rest_z = ep["rest"]["obj"][2]
    hd = float(np.hypot(*(o[:2] - b[:2])))
    dz = float(o[2] - b[2])
    dist_tcp = float(np.linalg.norm(o - tcp))
    speed = float(np.linalg.norm(s["obj_v"]))
    closed = snap["grip_cmd"] >= 0.9
    jaw_blocked = s["jaw"] > jaw_empty + JAW_MARGIN
    raised = o[2] - rest_z > RAISED
    held = bool(closed and dist_tcp < HELD_DIST and (jaw_blocked or raised))
    tilt = tilt_deg(s["bowl_q"])
    upright = tilt <= BOWL_TILT_MAX
    geom = hd < BOWL_RADIUS and dz < IN_BOWL_DZ
    in_bowl = bool(geom and upright and not held and speed < SETTLED_V)
    ambiguous = bool(geom and upright and not held and speed >= SETTLED_V)
    fell_off = bool(o[2] < -0.05)
    on_table = bool(not held and not geom and -0.05 < o[2] - rest_z < RAISED)
    over = bool(held and hd < BOWL_RADIUS + 0.01 and dz > OVER_DZ)
    shift = float(np.hypot(*(b[:2] - np.array(ep["rest"]["bowl"][:2]))))
    lab = dict(in_bowl=in_bowl, in_bowl_geom=bool(geom), held=held, held_lifted=bool(held and raised),
               over_bowl=over, bowl_upright=bool(upright), on_table=on_table, fell_off_table=fell_off,
               ambiguous=ambiguous, bowl_moved=bool(shift > 0.03))
    aux = dict(hd=round(hd, 4), dz=round(dz, 4), dist_tcp=round(dist_tcp, 4), speed=round(speed, 4),
               jaw=round(s["jaw"], 4), jaw_blocked=bool(jaw_blocked), raised_m=round(float(o[2] - rest_z), 4),
               bowl_tilt_deg=round(tilt, 1), bowl_shift_m=round(shift, 4))
    return lab, aux


def reason(lab, aux, o):
    if not lab["bowl_upright"]:
        return f"the yellow bowl is tipped over" + (f" and the gripper is holding the {o}" if lab["held"] else "")
    if lab["in_bowl"]:
        return f"the {o} is resting inside the bowl and the gripper has let go of it"
    if lab["held"] and lab["over_bowl"]:
        return f"the gripper is still holding the {o} above the bowl; it has not been released"
    if lab["held"] and lab["in_bowl_geom"]:
        return f"the gripper is still closed on the {o}; it has not been released"
    if lab["held"]:
        return f"the gripper is still holding the {o}, which is not in the bowl"
    if lab["fell_off_table"]:
        return f"the {o} fell off the table"
    if lab["on_table"]:
        return (f"the {o} is on the table next to the bowl, not inside it" if aux["hd"] < 0.16
                else f"the {o} is on the table, not in the bowl")
    return f"the {o} is not inside the bowl"


def reason_for(q, lab, aux, o):
    if q in ("in_bowl_placed", "in_bowl_visual"):
        return reason(lab, aux, o)
    if q == "held":
        if lab["held_lifted"]:
            return f"the gripper is holding the {o} clear of the table"
        if lab["held"]:
            return f"the gripper is closed on the {o} but it is not lifted clear of the table"
        return (f"the {o} is not in the gripper" + ("; it is on the table" if lab["on_table"] else ""))
    if q == "over_bowl":
        if lab["over_bowl"]:
            return f"the {o} is held in the gripper directly above the bowl"
        if lab["held"]:
            return f"the {o} is held but not above the bowl"
        return f"the gripper is not holding the {o}"
    if q == "bowl_upright":
        return "the bowl stands upright" if lab["bowl_upright"] else "the bowl is tipped over"
    if q == "on_table":
        if lab["on_table"]:
            return f"the {o} rests on the table"
        return reason(lab, aux, o)
    raise ValueError(q)


def answer(q, lab, aux, o):
    return json.dumps({JSON_KEY[q]: bool(lab[ANSWER_KEY[q]]), "reason": reason_for(q, lab, aux, o)})


def split_of(obj, ep, val_frac):
    if obj in HELDOUT:
        return "test"
    h = int(hashlib.sha1(f"{obj}|{ep['round']}|{ep['env']}|{ep.get('seed_tag','')}".encode()).hexdigest()[:8], 16)
    return "val" if (h % 1000) < val_frac * 1000 else "train"


def questions_for(kind, rng):
    if kind == "final":
        qs = ["in_bowl_placed"]
        if rng.random() < 0.5:
            qs.append("in_bowl_visual")
        if rng.random() < 0.3:
            qs.append(str(rng.choice(["bowl_upright", "on_table"])))
        return qs
    if kind == "release_instant":
        return ["in_bowl_placed"] if rng.random() < 0.6 else []
    if kind == "after_lift":
        return ["held"]
    if kind == "after_move":
        return ["over_bowl"]
    if kind == "mid":
        return [str(rng.choice(["held", "over_bowl", "bowl_upright", "on_table", "in_bowl_placed"]))]
    if kind == "before":
        r = rng.random()
        return ["on_table"] if r < 0.3 else (["in_bowl_placed"] if r < 0.5 else [])
    return []


def contact_sheet(frames, path, title, n=30):
    frames = frames[:n]
    if not frames:
        return
    cols, tw, th = 5, 256, 128
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * (th + 14) + 16), "white")
    d = ImageDraw.Draw(sheet)
    d.text((4, 2), title, fill="black")
    for k, f in enumerate(frames):
        im = Image.open(f["image"]).resize((tw, th))
        x, y = (k % cols) * tw, 16 + (k // cols) * (th + 14)
        sheet.paste(im, (x, y))
        d.text((x + 2, y + th), f"{f['object'][:12]} {f['scenario'][:14]}", fill="black")
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_root")
    ap.add_argument("out_dir")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--neg_pos_cap", type=float, default=1.5, help="train in_bowl questions: max negatives per positive")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eps = []
    for f in sorted(Path(a.raw_root).glob("*/episodes.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                e["seed_tag"] = f.parent.name
                eps.append(e)
    assert eps, f"no episodes under {a.raw_root}"
    # empty-closed jaw angle: closed frames with the object far (> 8 cm) from the TCP
    jaws = [s["state"]["jaw"] for e in eps for s in e["snaps"]
            if s["grip_cmd"] >= 0.9 and np.linalg.norm(np.array(s["state"]["obj"]) - np.array(s["state"]["tcp"])) > 0.08]
    all_closed = [s["state"]["jaw"] for e in eps for s in e["snaps"] if s["grip_cmd"] >= 0.9]
    jaw_empty = float(np.median(jaws)) if len(jaws) >= 5 else (float(np.min(all_closed)) if all_closed else 0.0)
    frames, pairs = [], []
    for e in eps:
        if "rest" not in e:
            continue
        o = e["object"]
        split = split_of(o, e, a.val_frac)
        final = None
        for s in e["snaps"]:
            lab, aux = label(s, e, jaw_empty)
            fr = dict(id=f"{e['seed_tag']}_r{e['round']}_e{e['env']:02d}_{s['kind']}", image=s["image"], object=o,
                      split=split, scenario=e["scenario"], kind=s["kind"], step=s["step"], truncated=e["truncated"],
                      labels=lab, aux=aux, max_rise_m=e["max_rise_m"], mean_px=s.get("mean_px"),
                      stalls=len(e.get("stalls", [])), dr=e.get("dr"), light=e.get("light"))
            frames.append(fr)
            if s["kind"] == "final":
                final = fr
        before = next((f for f in frames[::-1] if f["id"].startswith(f"{e['seed_tag']}_r{e['round']}_e{e['env']:02d}_")
                       and f["kind"] == "before"), None)
        if final and before:
            pairs.append(dict(id=final["id"][:-6], before=before["image"], after=final["image"], object=o, split=split,
                              scenario=e["scenario"], in_bowl=final["labels"]["in_bowl"], labels=final["labels"]))
    with open(out / "frames.jsonl", "w") as fh:
        for fr in frames:
            fh.write(json.dumps(fr) + "\n")
    json.dump(pairs, open(out / "pairs.json", "w"), indent=1)

    def rt_item(fr):  # run_test.py-compatible record (+ our labels)
        L = fr["labels"]
        return dict(id=fr["id"], image=fr["image"], object=fr["object"], in_bowl=L["in_bowl"],
                    strict=bool(L["in_bowl"] and fr["max_rise_m"] >= 0.03 and not L["bowl_moved"]),
                    lifted=bool(fr["max_rise_m"] >= 0.03), bowl_ok=bool(L["bowl_upright"] and not L["bowl_moved"]),
                    scenario=fr["scenario"], kind=fr["kind"], labels=L)

    usable = [f for f in frames if not f["labels"]["ambiguous"] and (f["mean_px"] or 0) > 3]
    test_final = [rt_item(f) for f in usable if f["split"] == "test" and f["kind"] == "final" and not f["truncated"]]
    test_rel = [rt_item(f) for f in usable if f["split"] == "test" and f["kind"] == "release_instant"]
    val_final = [rt_item(f) for f in usable if f["split"] == "val" and f["kind"] == "final" and not f["truncated"]]
    json.dump(test_final, open(out / "test_heldout_final.json", "w"), indent=1)
    json.dump(test_rel, open(out / "test_heldout_release.json", "w"), indent=1)
    json.dump(val_final, open(out / "val_final.json", "w"), indent=1)

    samples = defaultdict(list)
    for f in usable:
        if f["split"] == "test":
            continue
        for q in questions_for(f["kind"], rng):
            if f["truncated"] and q.startswith("in_bowl"):
                continue
            L, X = f["labels"], f["aux"]
            hard = bool(not L["in_bowl"] and ((L["held"] and (L["over_bowl"] or L["in_bowl_geom"] or X["hd"] < 0.10))
                                              or (not L["held"] and X["hd"] < 0.10)))
            samples[f["split"]].append(dict(id=f"{f['id']}|{q}", image=f["image"], object=f["object"], question=q,
                                            prompt=QUESTIONS[q](f["object"]), answer=answer(q, L, X, f["object"]),
                                            label=bool(L[ANSWER_KEY[q]]), scenario=f["scenario"], kind=f["kind"],
                                            hard_negative=hard))
    # cap in_bowl negatives in TRAIN (positives are the rarer class); val keeps its natural balance
    tr = samples["train"]
    ib = [s for s in tr if s["question"].startswith("in_bowl")]
    pos = [s for s in ib if s["label"]]
    neg = [s for s in ib if not s["label"]]
    cap = int(len(pos) * a.neg_pos_cap)
    if len(neg) > cap > 0:  # keep the hard negatives (held over/near the bowl, on the rim, next to it) first
        neg.sort(key=lambda s: (not s["hard_negative"], rng.random()))
        neg = neg[:cap]
    # coordinator 2026-10-06: weight the mix toward the hard negative and near-misses -> each hard in_bowl negative
    # is also asked with the other in_bowl wording (a second sample of the same frame)
    extra = []
    for s in neg:
        if s["hard_negative"]:
            q2 = "in_bowl_visual" if s["question"] == "in_bowl_placed" else "in_bowl_placed"
            extra.append({**s, "id": s["id"].rsplit("|", 1)[0] + "|" + q2, "question": q2,
                          "prompt": QUESTIONS[q2](s["object"])})
    seen = {s["id"] for s in tr}
    extra = [s for s in extra if s["id"] not in seen]
    tr = [s for s in tr if not s["question"].startswith("in_bowl")] + pos + neg + extra
    rng.shuffle(tr)
    samples["train"] = tr
    for sp in ("train", "val"):
        with open(out / f"{sp}.jsonl", "w") as fh:
            for s in samples[sp]:
                fh.write(json.dumps(s) + "\n")

    def bal(rows, key):
        c = Counter((r[key], r["label"]) for r in rows)
        return {f"{k}={'T' if v else 'F'}": n for (k, v), n in sorted(c.items())}

    fin = [f for f in frames if f["kind"] == "final"]
    stats = dict(
        episodes=len(eps), frames=len(frames), usable_frames=len(usable),
        ambiguous_frames=sum(f["labels"]["ambiguous"] for f in frames),
        dark_frames=sum((f["mean_px"] or 0) <= 3 for f in frames),
        truncated_episodes=sum(e["truncated"] for e in eps),
        objects={o: dict(episodes=sum(e["object"] == o for e in eps), split=split_of(o, {"round": 0, "env": 0}, 0)
                         if o in HELDOUT else "train/val") for o in sorted({e["object"] for e in eps})},
        jaw_empty_closed_rad=round(jaw_empty, 4), jaw_empty_n=len(jaws),
        thresholds=dict(BOWL_RADIUS=BOWL_RADIUS, IN_BOWL_DZ=IN_BOWL_DZ, BOWL_TILT_MAX=BOWL_TILT_MAX, SETTLED_V=SETTLED_V,
                        HELD_DIST=HELD_DIST, JAW_MARGIN=JAW_MARGIN, RAISED=RAISED, OVER_DZ=OVER_DZ),
        final_in_bowl_by_scenario={sc: dict(Counter(str(f["labels"]["in_bowl"]) for f in fin if f["scenario"] == sc))
                                   for sc in sorted({f["scenario"] for f in fin})},
        final_in_bowl_by_object={ob: dict(Counter(str(f["labels"]["in_bowl"]) for f in fin if f["object"] == ob))
                                 for ob in sorted({f["object"] for f in fin})},
        label_rates_by_kind={k: {lk: round(float(np.mean([f["labels"][lk] for f in frames if f["kind"] == k])), 3)
                                 for lk in ("in_bowl", "held", "held_lifted", "over_bowl", "bowl_upright", "on_table",
                                            "fell_off_table", "ambiguous")}
                             for k in sorted({f["kind"] for f in frames})},
        train_samples=len(samples["train"]), val_samples=len(samples["val"]),
        train_hard_negative_in_bowl=sum(s["hard_negative"] and s["question"].startswith("in_bowl")
                                        for s in samples["train"]),
        train_balance=bal(samples["train"], "question"), val_balance=bal(samples["val"], "question"),
        test_heldout_final=dict(n=len(test_final), pos=sum(t["in_bowl"] for t in test_final),
                                neg_lifted=sum((not t["in_bowl"]) and t["lifted"] for t in test_final)),
        test_heldout_release=dict(n=len(test_rel), pos=sum(t["in_bowl"] for t in test_rel)),
        val_final=dict(n=len(val_final), pos=sum(t["in_bowl"] for t in val_final)),
        pairs=len(pairs),
    )
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    rng2 = random.Random(1)
    groups = {
        "final_in_bowl": [f for f in fin if f["labels"]["in_bowl"]],
        "final_held_not_placed": [f for f in fin if f["labels"]["held"]],
        "final_on_table": [f for f in fin if f["labels"]["on_table"]],
        "final_bowl_tipped": [f for f in fin if not f["labels"]["bowl_upright"]],
        "after_lift_held": [f for f in frames if f["kind"] == "after_lift" and f["labels"]["held_lifted"]],
        "after_lift_not_held": [f for f in frames if f["kind"] == "after_lift" and not f["labels"]["held_lifted"]],
        "after_move_over_bowl": [f for f in frames if f["kind"] == "after_move" and f["labels"]["over_bowl"]],
    }
    for g, fs in groups.items():
        rng2.shuffle(fs)
        contact_sheet(fs, out / f"contact_{g}.png", f"{g} (n={len(fs)})")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
