"""Verifier dataset from EXISTING sim recordings (CPU only; operator decision 2026-10-06: no new blackforge GPU runs).

Sources (all labels from sim state / the sim's own success checks, never from a VLM):
  D  night-2 expert demos  overnight-gr00t-v2/full-20261005-2209/raw/<obj>/ep_*.npz  (1920 episodes, 4 train objects)
     per-frame scene/wrist RGB, phase, joint state (jaw = state[:, 5] rad), object root pose, bowl pose,
     episode strict json (in_bowl_end, bowl tilt/shift, max rise) computed in-sim with the object's true centre.
  E  GR00T policy evals    overnight-gr00t*/full-*/eval/{gr00t_final,gr00t_mid,replay_expert}/  mp4 (scene|wrist
     512x256) + results_<obj>.json (task/strict success, lifted, bowl_ok). Episodes end at the success term (the
     release instant) or at timeout.
  R  RL smoke evals        gr00t-rl/smoke/*/eval*/<obj>_<arm>_w<wave>_env<k>_<fail|task|strict>.mp4 (outcome in name)
Frames per demo episode (success episodes):
  final       last frame (released, retreated ~0.4 s, settled ~0.5 s)        in_bowl = in_bowl_end & tilt <= 20
  retreat     a RETREAT-phase frame (object in bowl, gripper rising)           in_bowl = same as final
  hover_low   a LOWER-phase frame: held low over the bowl, jaw closed          in_bowl = False  (THE hard negative)
  hover_high  end of CARRY: held above the bowl                                 in_bowl = False, over_bowl = True
  lift        LIFT/early CARRY with the object raised: held, not over the bowl in_bowl = False
  before      frame 0: object on the table                                      in_bowl = False
Failure demos: final frame, labeled from state (held / on table / bowl tipped); never-released episodes whose
object nevertheless ended inside the bowl footprint (spawn artefacts, 30 of 1920) are excluded from in_bowl questions.
Exclusions (test hygiene): demo episodes in vlm-verifier-test/set_v2.json and in skills-v0 cpu-tests verifier frames;
eval episodes in vlm-verifier-test/set.json (v1, which also supplies set_v2's negatives).
Splits: objects mustard bottle + cracker box -> test_heldout only (negatives only exist in the recordings: every
held-out-object episode on disk is a failure). Train objects: 10% of episodes (hash) -> val, rest -> train.
Also writes test_sugar.json (all sugar-box frames that are not in v1/v2) for the leave-one-object-out probe arm.

usage: build_from_recordings.py <out_dir> [--workers 12]
"""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import ANSWER_KEY, QUESTIONS, contact_sheet  # noqa: E402

DATA = "/mnt/weights/ai/robot-lab-data"
DEMO = f"{DATA}/overnight-gr00t-v2/full-20261005-2209/raw"
HELDOUT = {"mustard bottle", "cracker box"}
PH = {n: i for i, n in enumerate(["PREGRASP", "DESCEND", "CLOSE", "LIFT", "CARRY", "LOWER", "RELEASE", "RETREAT",
                                    "SETTLE", "DONE"])}
JAW_CLOSED = 0.6     # rad: demo jaw ~0.785 open (pre-shape), ~0.28 closed on an object
RISE_HELD = 0.015    # m: object root above its start height
OVER_BOWL_XY = 0.06  # m: object root within this of the bowl axis (root, not centre: mug offset ~1.1 cm)
TABLE_XY = 0.08


def tilt_deg(q):
    x, y, z, w = q
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def h01(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def state_labels(d, t, z0):
    op, bp = d["object_pose"][t], d["bowl_pose"][t]
    rise = float(op[2] - z0)
    hd = float(np.hypot(*(op[:2] - bp[:2])))
    jaw = float(d["state"][t, 5])
    held = jaw < JAW_CLOSED and rise > RISE_HELD
    tilt = tilt_deg(bp[3:7])
    return dict(held=bool(held), held_lifted=bool(held and rise > 0.02), over_bowl=bool(held and hd < OVER_BOWL_XY),
                bowl_upright=bool(tilt <= 20), on_table=bool((not held) and rise < 0.01 and hd > TABLE_XY),
                fell_off_table=bool(op[2] < -0.05)), dict(rise=round(rise, 4), hd=round(hd, 4), jaw=round(jaw, 3),
                                                          bowl_tilt=round(tilt, 1))


def demo_worker(job):
    path, obj, out_img, excluded = job
    d = np.load(path, allow_pickle=True)
    arr = {k: d[k] for k in ("phase", "state", "object_pose", "bowl_pose")}
    sj = json.loads(str(d["strict"]))
    success = bool(d["success"])
    ph = arr["phase"]
    n = len(ph)
    z0 = float(arr["object_pose"][0, 2])
    stem = f"demo_{obj.replace(' ', '_')}_{Path(path).stem}"
    rng = random.Random(stem)
    picks = []  # (kind, t, in_bowl label or None)
    in_end = bool(sj["in_bowl_end"] and sj["bowl_tilt_deg"] <= 20)
    released = int(ph.max()) >= PH["RELEASE"]
    if success:
        picks.append(("final", n - 1, True))
        rt = np.nonzero(ph == PH["RETREAT"])[0]
        if len(rt) and rng.random() < 0.5:
            picks.append(("retreat", int(rt[len(rt) // 2]), True))
        lw = np.nonzero(ph == PH["LOWER"])[0]
        if len(lw):
            picks.append(("hover_low", int(lw[rng.randrange(len(lw))]), False))
        cy = np.nonzero(ph == PH["CARRY"])[0]
        if len(cy):
            picks.append(("hover_high", int(cy[-1 - rng.randrange(min(5, len(cy)))]), False))
            raised = [t for t in np.nonzero((ph == PH["LIFT"]) | (ph == PH["CARRY"]))[0]
                      if arr["object_pose"][t, 2] - z0 > 0.03 and t < cy[len(cy) // 2]]
            if raised:
                picks.append(("lift", int(raised[rng.randrange(len(raised))]), False))
        if rng.random() < 0.25:
            picks.append(("before", 0, False))
    else:
        lab_end = None if (in_end and not released) else (in_end if released else False)
        never_grasped = int(ph.max()) <= PH["DESCEND"]
        if not never_grasped or rng.random() < 0.25 or sj["bowl_tilt_deg"] > 20:
            picks.append(("final", n - 1, lab_end))
    if excluded:
        picks = []
    if not picks:
        return []
    sc, wr = d["scene"], d["wrist"]  # one decompression each (NpzFile indexing re-decompresses per access)
    rows = []
    for kind, t, ib in picks:
        img = np.concatenate([sc[t], wr[t]], axis=1)
        p = f"{out_img}/{stem}_{kind}.png"
        Image.fromarray(img).resize((512, 256)).save(p)
        lab, aux = state_labels(arr, t, z0)
        lab["in_bowl"] = ib
        if ib:
            lab["held"] = lab["held_lifted"] = lab["over_bowl"] = False
        rows.append(dict(id=f"{stem}_{kind}", image=p, object=obj, source="demo", episode=stem, kind=kind, t=t,
                         labels=lab, aux=aux, lifted=bool(sj["max_rise_m"] >= 0.03), success_episode=success,
                         strict_json=sj))
    return rows


def release_open_worker(job):
    """run2 supplement: a RELEASE-phase frame where the jaw is already open past the env's success threshold (0.5 rad,
    robot_lab.tasks.so101_pick_place.object_in_bowl) and the object already rests at its final in-bowl pose -- the
    'fingers still around it, gripper opened' look of the set.json v1 positives (episodes end at that instant)."""
    path, obj, out_img, excluded = job
    if excluded:
        return []
    d = np.load(path, allow_pickle=True)
    if not bool(d["success"]):
        return []
    ph, st, op = d["phase"], d["state"], d["object_pose"]
    cand = [t for t in np.nonzero(ph == PH["RELEASE"])[0]
            if st[t, 5] > 0.5 and np.linalg.norm(op[t, :3] - op[-1, :3]) < 0.006]
    if not cand:
        return []
    stem = f"demo_{obj.replace(' ', '_')}_{Path(path).stem}"
    t = int(cand[random.Random(stem + "ro").randrange(len(cand))])
    img = np.concatenate([d["scene"][t], d["wrist"][t]], axis=1)  # two single decompressions
    p = f"{out_img}/{stem}_release_open.png"
    Image.fromarray(img).resize((512, 256)).save(p)
    z0 = float(op[0, 2])
    lab, aux = state_labels({"state": st, "object_pose": op, "bowl_pose": d["bowl_pose"]}, t, z0)
    lab.update(in_bowl=True, held=False, held_lifted=False, over_bowl=False)
    sj = json.loads(str(d["strict"]))
    return [dict(id=f"{stem}_release_open", image=p, object=obj, source="demo", episode=stem, kind="release_open", t=t,
                 labels=lab, aux=aux, lifted=bool(sj["max_rise_m"] >= 0.03), success_episode=True, strict_json=sj)]


def extract_frame(mp4, png, sseof):
    if not os.path.exists(png):
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-sseof", str(sseof), "-i", mp4, "-frames:v", "1",
                        png], check=True)
    return png


def eval_jobs(v1_ids):
    jobs = []
    for f in sorted(glob.glob(f"{DATA}/overnight-gr00t*/full-*/eval/*/results_*.json")):
        night = "n1" if "overnight-gr00t/" in f else "n2"
        sub = f.split("/")[-2]
        tag = {"gr00t_final": "final", "gr00t_mid": "mid", "replay_expert": "replay"}[sub]
        for e in json.load(open(f))["episodes"]:
            o = e["object"]
            vids = glob.glob(os.path.join(os.path.dirname(f), f"{o.replace(' ', '_')}_ep{e['episode']:02d}_*.mp4"))
            if not vids:
                continue
            eid = f"{night}_{tag}_{o.replace(' ', '_')}_ep{e['episode']:02d}"
            jobs.append(dict(eid=eid, mp4=vids[0], object=o, task=bool(e["success"]), strict=bool(e["strict_success"]),
                             lifted=bool(e["checks"]["lifted"]), bowl_ok=bool(e["checks"]["bowl_ok"]),
                             in_v1=eid in v1_ids, source=f"eval_{night}"))
    for mp4 in sorted(glob.glob(f"{DATA}/gr00t-rl/smoke/*/eval*/*.mp4")):
        m = re.match(r"(.+)_(base|rl)_w(\d+)_env(\d+)_(fail|task|strict)\.mp4", os.path.basename(mp4))
        if not m:
            continue
        o = m.group(1).replace("_", " ")
        rel = os.path.relpath(mp4, f"{DATA}/gr00t-rl/smoke").replace("/", "_")[:-4]
        jobs.append(dict(eid=f"rl_{rel}", mp4=mp4, object=o, task=m.group(5) != "fail", strict=m.group(5) == "strict",
                         lifted=None, bowl_ok=None, in_v1=False, source="rl_smoke"))
    return jobs


def eval_worker(job_out):
    j, out_img = job_out
    rows = []
    if j["in_v1"]:
        return rows
    lab_final = dict(in_bowl=j["task"])
    ambiguous = j["task"] and j["bowl_ok"] is False  # task success with a knocked bowl: excluded from training
    p = extract_frame(j["mp4"], f"{out_img}/{j['eid']}_final.png", -0.2)
    rows.append(dict(id=f"{j['eid']}_final", image=p, object=j["object"], source=j["source"], episode=j["eid"],
                     kind="eval_final", labels=lab_final, aux={}, lifted=j["lifted"], ambiguous=ambiguous,
                     task=j["task"], strict=j["strict"], bowl_ok=j["bowl_ok"]))
    if not j["task"]:  # a failure is a failure one second earlier too (still hovering / still on the table)
        p2 = extract_frame(j["mp4"], f"{out_img}/{j['eid']}_m1s.png", -1.2)
        rows.append(dict(id=f"{j['eid']}_m1s", image=p2, object=j["object"], source=j["source"], episode=j["eid"],
                         kind="eval_m1s", labels=dict(in_bowl=False), aux={}, lifted=j["lifted"], ambiguous=False,
                         task=False, strict=False, bowl_ok=j["bowl_ok"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--neg_pos_cap", type=float, default=1.6)
    ap.add_argument("--hard_frac", type=float, default=0.65, help="share of the capped in_bowl negatives that are hard")
    ap.add_argument("--resample_only", action="store_true", help="reuse frames.jsonl, redo splits/samples/stats only")
    ap.add_argument("--heldout_only", default="", help="write only the held-out-object test set to this file name")
    ap.add_argument("--add_release_open", default="", help="src dataset dir: copy its frames.jsonl + add release_open frames")
    ap.add_argument("--visual_frac", type=float, default=0.3, help="share of in_bowl samples asked with the visual wording")
    a = ap.parse_args()
    out = Path(a.out_dir)
    (out / "img").mkdir(parents=True, exist_ok=True)
    if a.add_release_open:
        src = Path(a.add_release_open)
        base_rows = [json.loads(line) for line in open(src / "frames.jsonl")]
        meta = json.load(open(src / "stats.json"))
        v2_demo = {x["id"] for x in json.load(open(f"{DATA}/vlm-verifier-test/set_v2.json")) if x["id"].startswith("demo_")}
        skills_src = {r["src"] for r in json.load(open(f"{DATA}/skills-v0/cpu-tests/verifier_frames_gemma.json"))["results"]}
        jobs = []
        for objdir in sorted(Path(DEMO).iterdir()):
            if objdir.is_dir():
                for f in sorted(objdir.glob("ep_*.npz")):
                    jobs.append((str(f), objdir.name.replace("_", " "), str(out / "img"),
                                 f"demo_{objdir.name}_{f.stem}" in v2_demo or str(f) in skills_src))
        with ProcessPoolExecutor(a.workers) as ex:
            new = [r for rs in ex.map(release_open_worker, jobs, chunksize=4) for r in rs]
        for r in new:
            r["sha1"] = hashlib.sha1(open(r["image"], "rb").read()).hexdigest()
        rows = base_rows + new
        for r in rows:
            r["split"] = "test" if r["object"] in HELDOUT else ("val" if h01(r["episode"]) < a.val_frac else "train")
        with open(out / "frames.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"release_open frames added: {len(new)}", flush=True)
        return finish(a, out, rows, meta["demo_jobs"], meta["demo_excluded_v2_or_skills"], meta["eval_jobs"],
                      meta["eval_excluded_v1"])
    if a.heldout_only:  # refresh ONLY the held-out-object test set (new eval episodes); training data untouched
        v1_ids = {x["id"] for x in json.load(open(f"{DATA}/vlm-verifier-test/set.json"))}
        hj = [j for j in eval_jobs(v1_ids) if j["object"] in HELDOUT]
        with ProcessPoolExecutor(a.workers) as ex:
            hr = [r for rs in ex.map(eval_worker, [(j, str(out / "img")) for j in hj], chunksize=2) for r in rs]
        test = [dict(id=r["id"], image=r["image"], object=r["object"], in_bowl=bool(r["labels"]["in_bowl"]),
                     lifted=bool(r.get("lifted")), kind=r["kind"], source=r["source"], strict=bool(r.get("strict")),
                     bowl_ok=r.get("bowl_ok")) for r in hr if not r.get("ambiguous")]
        json.dump(test, open(out / a.heldout_only, "w"), indent=1)
        print(json.dumps(dict(heldout_jobs=len(hj), in_v1=sum(j["in_v1"] for j in hj), items=len(test),
                              pos=sum(t["in_bowl"] for t in test), by=dict(Counter(f"{t['object']}|{t['source']}|{t['kind']}|{t['in_bowl']}" for t in test))), indent=1))
        return
    if a.resample_only:
        rows = [json.loads(line) for line in open(out / "frames.jsonl")]
        meta = json.load(open(out / "stats.json"))
        return finish(a, out, rows, meta["demo_jobs"], meta["demo_excluded_v2_or_skills"], meta["eval_jobs"],
                      meta["eval_excluded_v1"])
    v1 = json.load(open(f"{DATA}/vlm-verifier-test/set.json"))
    v2 = json.load(open(f"{DATA}/vlm-verifier-test/set_v2.json"))
    v1_ids = {x["id"] for x in v1}
    v2_demo = {x["id"] for x in v2 if x["id"].startswith("demo_")}
    skills = json.load(open(f"{DATA}/skills-v0/cpu-tests/verifier_frames_gemma.json"))["results"]
    skills_src = {r["src"] for r in skills}
    jobs = []
    for objdir in sorted(Path(DEMO).iterdir()):
        if not objdir.is_dir():
            continue
        obj = objdir.name.replace("_", " ")
        for f in sorted(objdir.glob("ep_*.npz")):
            stem = f"demo_{objdir.name}_{f.stem}"
            jobs.append((str(f), obj, str(out / "img"), stem in v2_demo or str(f) in skills_src))
    n_excl = sum(j[3] for j in jobs)
    with ProcessPoolExecutor(a.workers) as ex:
        demo_rows = [r for rs in ex.map(demo_worker, jobs, chunksize=4) for r in rs]
    ejobs = eval_jobs(v1_ids)
    with ProcessPoolExecutor(a.workers) as ex:
        eval_rows = [r for rs in ex.map(eval_worker, [(j, str(out / "img")) for j in ejobs], chunksize=2) for r in rs]
    # dedupe identical eval frames (RL eval/eval_video dirs re-render the same seeds)
    seen, rows = set(), []
    for r in demo_rows + eval_rows:
        h = hashlib.sha1(open(r["image"], "rb").read()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        r["sha1"] = h
        rows.append(r)
    for r in rows:
        if r["object"] in HELDOUT:
            r["split"] = "test"
        else:
            r["split"] = "val" if h01(r["episode"]) < a.val_frac else "train"
    with open(out / "frames.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return finish(a, out, rows, len(jobs), n_excl, len(ejobs), sum(j["in_v1"] for j in ejobs))


def finish(a, out, rows, n_jobs, n_excl, n_ejobs, n_ev1):

    def rt(r):
        return dict(id=r["id"], image=r["image"], object=r["object"], in_bowl=bool(r["labels"]["in_bowl"]),
                    lifted=bool(r.get("lifted")), kind=r["kind"], source=r["source"],
                    strict=bool(r.get("strict", r["labels"]["in_bowl"])), bowl_ok=r.get("bowl_ok"))

    ib_ok = [r for r in rows if r["labels"].get("in_bowl") is not None and not r.get("ambiguous")]
    test = [rt(r) for r in ib_ok if r["split"] == "test"]
    val = [rt(r) for r in ib_ok if r["split"] == "val"]
    val_final = [v for v in val if v["kind"] in ("final", "eval_final")]
    sugar = [rt(r) for r in ib_ok if r["object"] == "sugar box"]
    json.dump(test, open(out / "test_heldout.json", "w"), indent=1)
    json.dump(val, open(out / "val_all.json", "w"), indent=1)
    json.dump(val_final, open(out / "val_final.json", "w"), indent=1)
    json.dump(sugar, open(out / "test_sugar.json", "w"), indent=1)

    rng = random.Random(0)
    samples = defaultdict(list)
    for r in rows:
        if r["split"] == "test":
            continue
        L, o = r["labels"], r["object"]
        qs = []
        if L.get("in_bowl") is not None and not r.get("ambiguous"):
            qs.append("in_bowl_visual" if rng.random() < a.visual_frac else "in_bowl_placed")
        if r["source"] == "demo":
            if r["kind"] in ("lift", "before", "hover_high") or (r["kind"] == "final" and not r["success_episode"]):
                qs.append("held")
            if r["kind"] in ("hover_high", "hover_low", "lift"):
                qs.append("over_bowl")
            if not L["bowl_upright"] or rng.random() < 0.1:
                qs.append("bowl_upright")
        for q in qs:
            lab = L[ANSWER_KEY[q]]
            hard = bool(q.startswith("in_bowl") and not lab and (r["kind"] in ("hover_low", "hover_high")
                                                                 or (r["kind"] in ("eval_final", "eval_m1s") and r.get("lifted"))))
            samples[r["split"]].append(dict(id=f"{r['id']}|{q}", image=r["image"], object=o, question=q,
                                            prompt=QUESTIONS[q](o), answer=answer_for(q, L, o, r), label=bool(lab),
                                            kind=r["kind"], source=r["source"], hard_negative=hard))
    tr = samples["train"]
    ib = [s for s in tr if s["question"].startswith("in_bowl")]
    pos = [s for s in ib if s["label"]]
    neg = [s for s in ib if not s["label"]]
    # negatives capped at neg_pos_cap x positives; of those, hard_frac are hard negatives (held over the bowl /
    # policy lifted-not-placed), the rest easy/other (on table, never grasped, tipped bowl) -- coordinator 2026-10-06:
    # weight toward the hard negative without starving the positives (missed-success must stay <= ~2/45)
    cap = int(len(pos) * a.neg_pos_cap)
    hardn = [s for s in neg if s["hard_negative"]]
    easyn = [s for s in neg if not s["hard_negative"]]
    rng.shuffle(hardn)
    rng.shuffle(easyn)
    nh = min(len(hardn), int(cap * a.hard_frac))
    neg = hardn[:nh] + easyn[:cap - nh]
    tr = [s for s in tr if not s["question"].startswith("in_bowl")] + pos + neg
    rng.shuffle(tr)
    samples["train"] = tr
    for sp in ("train", "val"):
        with open(out / f"{sp}.jsonl", "w") as fh:
            for s in samples[sp]:
                fh.write(json.dumps(s) + "\n")
    bal = lambda rows_: {f"{q}={'T' if v else 'F'}": n for (q, v), n in sorted(Counter((s["question"], s["label"]) for s in rows_).items())}  # noqa: E731
    stats = dict(
        demo_jobs=n_jobs, demo_excluded_v2_or_skills=n_excl, eval_jobs=n_ejobs,
        eval_excluded_v1=n_ev1, frames=len(rows),
        frames_by_source_kind=dict(Counter(f"{r['source']}:{r['kind']}" for r in rows)),
        frames_by_object_split=dict(Counter(f"{r['object']}|{r['split']}" for r in rows)),
        in_bowl_by_kind={k: dict(Counter(str(r["labels"].get("in_bowl")) for r in rows if r["kind"] == k))
                         for k in sorted({r["kind"] for r in rows})},
        train_samples=len(samples["train"]), val_samples=len(samples["val"]),
        train_balance=bal(samples["train"]), val_balance=bal(samples["val"]),
        train_hard_negative_in_bowl=sum(s["hard_negative"] and s["question"].startswith("in_bowl") for s in samples["train"]),
        test_heldout=dict(n=len(test), pos=sum(t["in_bowl"] for t in test), lifted_neg=sum((not t["in_bowl"]) and t["lifted"] for t in test)),
        val_final=dict(n=len(val_final), pos=sum(t["in_bowl"] for t in val_final)),
        test_sugar=dict(n=len(sugar), pos=sum(t["in_bowl"] for t in sugar)),
        thresholds=dict(JAW_CLOSED=JAW_CLOSED, RISE_HELD=RISE_HELD, OVER_BOWL_XY=OVER_BOWL_XY, TABLE_XY=TABLE_XY),
        image_bytes=sum(os.path.getsize(r["image"]) for r in rows),
    )
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    rng2 = random.Random(1)
    for name, sel in {"pos_release_open": lambda r: r["kind"] == "release_open",
                      "pos_final": lambda r: r["kind"] == "final" and r["labels"]["in_bowl"],
                      "neg_hover_low": lambda r: r["kind"] == "hover_low",
                      "neg_hover_high": lambda r: r["kind"] == "hover_high",
                      "neg_demo_fail_final": lambda r: r["kind"] == "final" and r["labels"]["in_bowl"] is False,
                      "eval_final_pos": lambda r: r["kind"] == "eval_final" and r["labels"]["in_bowl"],
                      "eval_final_neg_lifted": lambda r: r["kind"] == "eval_final" and not r["labels"]["in_bowl"] and r.get("lifted"),
                      "test_heldout": lambda r: r["split"] == "test"}.items():
        fs = [dict(image=r["image"], object=r["object"], scenario=r["kind"]) for r in rows if sel(r)]
        rng2.shuffle(fs)
        contact_sheet(fs, out / f"contact_{name}.png", f"{name} (n={len(fs)})")
    print(json.dumps(stats, indent=1))


def answer_for(q, L, o, r):
    k = r["kind"]
    if q.startswith("in_bowl"):
        if L["in_bowl"] and k == "release_open":
            why = f"the {o} is down inside the bowl and the gripper has opened around it"
        elif L["in_bowl"]:
            why = f"the {o} is resting inside the bowl and the gripper has let go of it"
        elif r["source"] == "demo" and not L.get("bowl_upright", True):
            why = "the yellow bowl is tipped over"
        elif k == "hover_low":
            why = f"the gripper is still closed on the {o} over the bowl; it has not been released"
        elif k == "hover_high":
            why = f"the gripper is still holding the {o} above the bowl; it has not been released"
        elif k == "lift" or L.get("held"):
            why = f"the gripper is still holding the {o}, which is not in the bowl"
        elif k in ("eval_final", "eval_m1s"):
            why = (f"the {o} has not been put down inside the bowl" if r.get("lifted")
                   else f"the {o} was never picked up; it is not in the bowl")
        elif L.get("on_table"):
            why = f"the {o} is on the table, not in the bowl"
        else:
            why = f"the {o} is not inside the bowl"
        return json.dumps({"in_bowl": bool(L["in_bowl"]), "reason": why})
    if q == "held":
        v = L["held_lifted"]
        why = (f"the gripper is holding the {o} clear of the table" if v else
               (f"the {o} is on the table, not in the gripper" if L.get("on_table") else f"the {o} is not held in the gripper"))
        return json.dumps({"held": bool(v), "reason": why})
    if q == "over_bowl":
        v = L["over_bowl"]
        why = (f"the {o} is held in the gripper directly above the bowl" if v else
               (f"the {o} is held but not above the bowl" if L["held"] else f"the gripper is not holding the {o}"))
        return json.dumps({"over_bowl": bool(v), "reason": why})
    if q == "bowl_upright":
        v = L["bowl_upright"]
        return json.dumps({"bowl_upright": bool(v), "reason": "the bowl stands upright" if v else "the bowl is tipped over"})
    raise ValueError(q)


if __name__ == "__main__":
    main()
