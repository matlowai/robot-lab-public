"""Score base / fine-tuned Gemma 4 12B (4-bit, Unsloth, local HF generate) as a post-skill verifier.

Same prompts and metrics as vlm-verifier-test/run_test.py (accuracy, false-success rate = predicted done on a
failure, false success on lifted-but-not-placed failures, missed-success rate, latency), on any run_test-format set.
Greedy decoding; max_new_tokens 32768 (thinking-model floor per repo rule; answers stop at EOS after ~30 tokens).

usage: eval_verifier.py --name base|A|B --model <hf id | adapter dir> --sets name=path.json,... --out <dir>
       [--variants placed,visual] [--bs 8] [--limit 0]
"""

import argparse
import json
import re
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--sets", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--variants", default="placed,visual")
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

import unsloth  # noqa: E402,F401
from unsloth import FastVisionModel  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

HEAD = ("Robot task: put the {o} in the yellow bowl. The image shows the end of the attempt: left half is the "
        "scene camera, right half is the robot's wrist camera. ")
QQ = {"visual": "Question: is the {o} now inside the yellow bowl (not held above it, not on the table)? ",
      "placed": ("Question: has the {o} been placed into the yellow bowl, i.e. it is down inside the bowl and the "
                 "gripper has opened, even if the gripper fingers are still around it? Answer false if it is on the "
                 "table, lifted above the bowl, outside the bowl, or the bowl is knocked over. ")}
TAIL = 'Reply with only JSON: {"in_bowl": true or false, "reason": "<short>"}'

t0 = time.time()
model, processor = FastVisionModel.from_pretrained(args.model, load_in_4bit=True)
FastVisionModel.for_inference(model)
load_s = time.time() - t0
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


# skills-v0 verifier prompt (robot-lab/skills/vlm.py VLMVerifier, combined image, reply="json"), used verbatim for
# items that carry a "skill" field (skills-v0 cpu-test frames); answer key "success" instead of "in_bowl".
SK_SYSTEM = ("You check whether a robot arm's last action succeeded, by looking at camera images taken right "
             "after it. Be strict: answer yes only if the images clearly show it.")
SK_Q = {"pick_up": "Is the {object} held in the robot's gripper and lifted clearly above the table?",
        "move_to": "Is the robot gripper holding the {object} directly above the {target}?",
        "release": "Is the {object} inside the {target}?"}


def skills_text(it):
    views = ("The image shows two views side by side: left half is a fixed camera looking at the table, right half "
             "is the camera on the gripper.")
    q = SK_Q[it["skill"]].format(object=it["object"], target="yellow bowl")
    return (f"{views} The robot just ran the skill \"\" and then held still.\nQuestion: {q}\n"
            'Reply with only JSON: {"success": true or false, "reason": "<short>"}')


def render(prompt, system=None):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if system:
        msgs.insert(0, {"role": "system", "content": [{"type": "text", "text": system}]})
    return processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


@torch.inference_mode()
def generate(batch):
    tok.padding_side = "left"
    texts = [render(p, SK_SYSTEM if sk else None) for p, _, sk in batch]
    imgs = [[Image.open(i).convert("RGB")] for _, i, _ in batch]
    enc = processor(text=texts, images=imgs, return_tensors="pt", padding=True).to(model.device)
    gen = model.generate(**enc, max_new_tokens=32768, do_sample=False, use_cache=True)
    new = gen[:, enc["input_ids"].shape[1]:]
    return [tok.decode(x, skip_special_tokens=True) for x in new], [int((x != tok.pad_token_id).sum()) for x in new]


def parse(txt, key="in_bowl"):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0)).get(key)
        return v if isinstance(v, bool) else None
    except Exception:
        return None


def summarize(res, variant, set_name):
    ok = [r for r in res if r["pred"] is not None]
    tp = sum(r["pred"] and r["in_bowl"] for r in ok)
    tn = sum((not r["pred"]) and (not r["in_bowl"]) for r in ok)
    fp = sum(r["pred"] and not r["in_bowl"] for r in ok)
    fn = sum((not r["pred"]) and r["in_bowl"] for r in ok)
    fpl = sum(r["pred"] and not r["in_bowl"] and r.get("lifted", False) for r in ok)
    neg = sum(not r["in_bowl"] for r in ok)
    pos = sum(r["in_bowl"] for r in ok)
    negl = sum((not r["in_bowl"]) and r.get("lifted", False) for r in ok)
    lat = sorted(r["latency_s"] for r in ok)
    return {"model": args.name, "set": set_name, "prompt": variant, "n": len(res), "answered": len(ok),
            "accuracy": round((tp + tn) / max(1, len(ok)), 3), "false_success_rate": f"{fp}/{neg}",
            "false_success_frac": round(fp / max(1, neg), 3),
            "false_success_on_lifted_not_placed": f"{fpl}/{negl}", "missed_success_rate": f"{fn}/{pos}",
            "missed_success_frac": round(fn / max(1, pos), 3),
            "latency_median_s_per_item": lat[len(lat) // 2] if lat else None,
            "median_completion_tokens": sorted(r["completion_tokens"] for r in ok)[len(ok) // 2] if ok else None,
            "batch_size": args.bs, "model_path": args.model, "load_s": round(load_s, 1)}


all_summ = []
for spec in args.sets.split(","):
    set_name, path = spec.split("=", 1)
    items = json.load(open(path))
    if args.limit:
        items = items[:args.limit]
    is_sk = bool(items) and "skill" in items[0]
    for variant in (["skills"] if is_sk else args.variants.split(",")):
        res = []
        for k in range(0, len(items), args.bs):
            chunk = items[k:k + args.bs]
            prompts = [((skills_text(it) if is_sk else HEAD.format(o=it["object"]) + QQ[variant].format(o=it["object"])
                         + TAIL), it["image"], is_sk) for it in chunk]
            t = time.time()
            try:
                txts, ntoks = generate(prompts)
            except Exception as ex:  # noqa: BLE001 -- recorded per item, counted as unanswered (loud in summary)
                txts, ntoks = [f"ERROR {type(ex).__name__}: {ex}"] * len(chunk), [0] * len(chunk)
            dt = (time.time() - t) / len(chunk)
            for it, txt, nt in zip(chunk, txts, ntoks):
                res.append({**{k2: it[k2] for k2 in ("id", "image", "object", "in_bowl", "skill") if k2 in it},
                            "lifted": it.get("lifted", False), "scenario": it.get("scenario"), "kind": it.get("kind"),
                            "source": it.get("source"),
                            "pred": parse(txt, "success" if is_sk else "in_bowl"), "latency_s": round(dt, 3),
                            "completion_tokens": nt, "raw": txt[-300:]})
        s = summarize(res, variant, set_name)
        if is_sk:  # per-skill breakdown in the skills-v0 report's terms (yes_but_false = false success)
            for sk in ("pick_up", "move_to", "release"):
                r2 = [r for r in res if r.get("skill") == sk and r["pred"] is not None]
                s[sk] = dict(n=len(r2), matched=sum(r["pred"] == r["in_bowl"] for r in r2),
                             yes_but_false=sum(r["pred"] and not r["in_bowl"] for r in r2),
                             no_but_true=sum((not r["pred"]) and r["in_bowl"] for r in r2))
        all_summ.append(s)
        json.dump({"summary": s, "results": res}, open(out / f"eval_{args.name}_{set_name}_{variant}.json", "w"), indent=1)
        print("EVAL", json.dumps(s), flush=True)
json.dump(all_summ, open(out / f"summary_{args.name}.json", "w"), indent=1)
