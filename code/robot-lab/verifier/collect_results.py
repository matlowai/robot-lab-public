"""Markdown tables from eval_verifier.py / eval_endpoint.py summaries (copied back from box0).

usage: collect_results.py <runs_dir> [> RESULTS_TABLES.md]
Reads <runs_dir>/run*/eval/summary_*.json, run*/eval_post/summary_*.json and any ep_summary_*.json.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for f in sorted(root.glob("run*/eval*/summary_*.json")) + sorted(root.glob("**/ep_summary_*.json")):
    run = f.parts[len(root.parts)] if f.is_relative_to(root) else "?"
    for s in json.load(open(f)):
        s = dict(s)
        s["run"] = run
        s["where"] = f.parent.name
        rows.append(s)
by = defaultdict(list)
for s in rows:
    by[(s["set"], s["prompt"])].append(s)
order = ["v2", "v1", "heldout2", "heldout", "val", "skills72", "sugar"]
print("| set | prompt | run/model | n | acc | false-success (pred done on failure) | FS on lifted-not-placed | "
      "missed success | latency/item s | bs/conc |")
print("|---|---|---|---|---|---|---|---|---|---|")
for key in sorted(by, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1])):
    for s in sorted(by[key], key=lambda s: (s["run"], s["model"])):
        lat = s.get("latency_median_s_per_item", s.get("latency_median_s"))
        extra = ""
        if "pick_up" in s:
            extra = " ; " + ", ".join(f"{k}: yes_but_false {s[k]['yes_but_false']}, no_but_true {s[k]['no_but_true']}"
                                      for k in ("pick_up", "move_to", "release"))
        print(f"| {key[0]} | {key[1]} | {s['run']}/{s['model']} | {s['n']} | {s['accuracy']:.3f} | "
              f"{s['false_success_rate']} | {s['false_success_on_lifted_not_placed']} | {s['missed_success_rate']}{extra} | "
              f"{lat} | {s.get('batch_size', s.get('concurrency'))} |")
