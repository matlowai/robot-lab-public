#!/usr/bin/env bash
# Train a PPO avoidance policy, then evaluate it with the same live hard-tier protocol as RL v1 / v2 (D42, D44),
# all in one process tree, so one systemd unit covers the whole job:
#
#   systemd-run --user --unit=patrol-rl-hard-v3 -p MemoryMax=24G --working-directory=/mnt/work/AI/patrol-lab \
#       tools/rl_train_eval.sh <out_dir> <tag> <train.py args...>
#
#   1. train      CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-1} python -m benchmarks.avoidance.rl.train --out <out_dir> <args>
#   2. scan       every checkpoint on the validation seeds 21-60 (live hard tier, rl only)
#                 -> data/benchmarks/rl-<tag>-scan/<ckpt>/rl-eval-*/summary.json   (informational)
#   3. held out   the FINAL checkpoint (the rule v1 and v2 ended up with; fixed before seeing any v3 number) on
#                 seeds 121-200, live hard tier, controllers rl,heuristic,control
#                 -> data/benchmarks/rl-<tag>-final/rl-eval-*/summary.json
# Eval runs on CPU (CUDA_VISIBLE_DEVICES=""), 24 workers, exactly like the v1 / v2 evaluations. The held-out command
# is echoed into the log so it can be rerun by hand. <out_dir>/EVAL_DONE lists the result directories.
set -euo pipefail

OUT=${1:?usage: rl_train_eval.sh <out_dir> <tag> <train args...>}
TAG=${2:?usage: rl_train_eval.sh <out_dir> <tag> <train args...>}
shift 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${ISAAC_PY:-/mnt/weights/ai/isaac/IsaacLab/.venv/bin/python}
BENCH="$REPO/data/benchmarks"
VAL_SEEDS=${VAL_SEEDS:-21-60}     # overridable only to test this script; the protocol is 21-60 / 121-200
TEST_SEEDS=${TEST_SEEDS:-121-200}
cd "$REPO"
export PYTHONPATH="$REPO"

echo "[$(date '+%F %T')] train -> $OUT (GPU ${TRAIN_GPU:-1})"
CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-1} "$PY" -m benchmarks.avoidance.rl.train --out "$OUT" "$@"
test -f "$OUT/DONE"

for ck in "$OUT"/ckpt_*.pt; do
    name=$(basename "$ck" .pt)
    echo "[$(date '+%F %T')] validation scan $name (seeds $VAL_SEEDS)"
    CUDA_VISIBLE_DEVICES="" "$PY" -m benchmarks.avoidance.rl.eval --ckpt "$ck" --seeds "$VAL_SEEDS" --controllers rl \
        --live hard --workers 24 --out "$BENCH/rl-$TAG-scan/$name" | tail -n 1
done

FINAL=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['final'])" "$OUT/DONE")
CMD=(env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" "$PY" -m benchmarks.avoidance.rl.eval --ckpt "$FINAL"
     --seeds "$TEST_SEEDS" --controllers rl,heuristic,control --live hard --workers 24 --out "$BENCH/rl-$TAG-final")
echo "[$(date '+%F %T')] held-out eval: ${CMD[*]}"
"${CMD[@]}" | tail -n 1

"$PY" - "$BENCH/rl-$TAG-scan" "$BENCH/rl-$TAG-final" "$OUT/EVAL_DONE" <<'EOF'
import glob, json, sys
scan_dir, final_dir, done = sys.argv[1:4]
def row(summary):
    out = {}
    for c, r in json.load(open(summary))["by_controller"].items():
        lc = r.get("live_crowd", {})
        out[c] = {"episodes": r["episodes"], "completed": r["completed"], "time_s_median": r["time_s_median"],
                  "hit_per_commit": lc.get("hit_per_commit"), "locks_per_run": lc.get("locks_per_run"),
                  "episodes_with_ambient_contact": lc.get("episodes_with_ambient_contact"),
                  "closest_m_median": r["closest_m_median"]}
    return out
scan = {p.split("/")[-3]: row(p)["rl"] for p in sorted(glob.glob(f"{scan_dir}/*/*/summary.json"))}
final = sorted(glob.glob(f"{final_dir}/*/summary.json"))[-1]
res = {"validation_scan": scan, "held_out": row(final), "final_summary": final}
open(done, "w").write(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
EOF
echo "[$(date '+%F %T')] all done"
