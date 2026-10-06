"""Score an OpenAI-compatible VLM endpoint (e.g. vLLM serving the QAT base + our LoRA) as a post-skill verifier.

Same request as vlm-verifier-test/run_test.py (image as PNG data URL, temperature 0, max_tokens 32768) and the same
metrics as eval_verifier.py; per-request wall latency at the given concurrency. Writes to --out (never into
vlm-verifier-test/). Path prefixes in the set files can be remapped with --map old=new (box0 vs blackforge paths).

usage: eval_endpoint.py --name qat_A --base http://10.77.0.1:8012/v1 --model verifierA --sets v2=...,v1=... --out <dir>
       [--variants placed,visual] [--conc 1] [--map /media/...=/mnt/...]
"""
import argparse
import base64
import concurrent.futures as cf
import json
import re
import time
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--sets", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--variants", default="placed,visual")
ap.add_argument("--conc", type=int, default=1)
ap.add_argument("--map", default="")
args = ap.parse_args()
HEAD = ("Robot task: put the {o} in the yellow bowl. The image shows the end of the attempt: left half is the "
        "scene camera, right half is the robot's wrist camera. ")
QQ = {"visual": "Question: is the {o} now inside the yellow bowl (not held above it, not on the table)? ",
      "placed": ("Question: has the {o} been placed into the yellow bowl, i.e. it is down inside the bowl and the "
                 "gripper has opened, even if the gripper fingers are still around it? Answer false if it is on the "
                 "table, lifted above the bowl, outside the bowl, or the bowl is knocked over. ")}
TAIL = 'Reply with only JSON: {"in_bowl": true or false, "reason": "<short>"}'
mp = tuple(args.map.split("=", 1)) if args.map else None


def ask(it, variant):
    path = it["image"].replace(*mp) if mp else it["image"]
    img = base64.b64encode(open(path, "rb").read()).decode()
    q = HEAD.format(o=it["object"]) + QQ[variant].format(o=it["object"]) + TAIL
    body = {"model": args.model, "max_tokens": 32768, "temperature": 0,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
                                                      {"type": "text", "text": q}]}]}
    t = time.time()
    req = urllib.request.Request(args.base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        txt = r["choices"][0]["message"].get("content") or ""
        m = re.search(r"\{.*\}", txt, re.S)
        pred = json.loads(m.group(0)).get("in_bowl") if m else None
        pred = pred if isinstance(pred, bool) else None
        toks = r.get("usage", {}).get("completion_tokens")
    except Exception as ex:  # noqa: BLE001 -- counted as unanswered, visible in the summary
        txt, pred, toks = f"ERROR {ex}", None, None
    return {**{k: it[k] for k in ("id", "object", "in_bowl") if k in it}, "lifted": it.get("lifted", False),
            "kind": it.get("kind"), "pred": pred, "latency_s": round(time.time() - t, 3), "completion_tokens": toks,
            "raw": txt[-300:]}


out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
allsum = []
for spec in args.sets.split(","):
    sname, path = spec.split("=", 1)
    items = json.load(open(path.replace(*mp) if mp else path))
    for variant in args.variants.split(","):
        with cf.ThreadPoolExecutor(args.conc) as ex:
            res = list(ex.map(lambda it: ask(it, variant), items))
        ok = [r for r in res if r["pred"] is not None]
        fp = sum(r["pred"] and not r["in_bowl"] for r in ok)
        fn = sum((not r["pred"]) and r["in_bowl"] for r in ok)
        fpl = sum(r["pred"] and not r["in_bowl"] and r["lifted"] for r in ok)
        neg, pos = sum(not r["in_bowl"] for r in ok), sum(r["in_bowl"] for r in ok)
        negl = sum((not r["in_bowl"]) and r["lifted"] for r in ok)
        lat = sorted(r["latency_s"] for r in ok)
        s = {"model": args.name, "served_model": args.model, "endpoint": args.base, "set": sname, "prompt": variant,
             "n": len(res), "answered": len(ok), "accuracy": round((len(ok) - fp - fn) / max(1, len(ok)), 3),
             "false_success_rate": f"{fp}/{neg}", "false_success_on_lifted_not_placed": f"{fpl}/{negl}",
             "missed_success_rate": f"{fn}/{pos}", "latency_median_s": lat[len(lat) // 2] if lat else None,
             "latency_p90_s": lat[int(len(lat) * 0.9)] if lat else None, "concurrency": args.conc,
             "median_completion_tokens": sorted(r["completion_tokens"] or 0 for r in ok)[len(ok) // 2] if ok else None}
        allsum.append(s)
        json.dump({"summary": s, "results": res}, open(out / f"ep_{args.name}_{sname}_{variant}.json", "w"), indent=1)
        print("EVAL", json.dumps(s), flush=True)
json.dump(allsum, open(out / f"ep_summary_{args.name}.json", "w"), indent=1)
