#!/usr/bin/env bash
# RL v4 (D46) references: the hand-written planner and the no-avoidance control on the validation seeds (selection
# baseline for tools/v4_select.py) and on the sealed test seeds (the bar the arms are judged against), same protocol.
# eval.py needs a checkpoint to register "rl"; these controllers never use it.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${ISAAC_PY:-/mnt/weights/ai/isaac/IsaacLab/.venv/bin/python}
ROOT=${V4_ROOT:-/mnt/weights/ai/patrol-lab-data/rl/v4}
BENCH="$ROOT/bench"; LOCK="$ROOT/eval.lock"
ANY_CKPT=${ANY_CKPT:-/mnt/weights/ai/patrol-lab-data/rl/avoid-ppo-hard-v2-20260925-1603/ckpt_16549.pt}
cd "$REPO"; export PYTHONPATH="$REPO"; mkdir -p "$BENCH"
for pair in "val:${VAL_SEEDS:-21-60}" "heldout:${TEST_SEEDS:-121-200}"; do
    tag=${pair%%:*}; seeds=${pair#*:}
    [[ -f "$BENCH/refs-$tag/DONE" ]] && continue
    flock "$LOCK" env CUDA_VISIBLE_DEVICES="" "$PY" -m benchmarks.avoidance.rl.eval --ckpt "$ANY_CKPT" --seeds "$seeds" \
        --controllers heuristic,control --live hard --workers "${WORKERS:-24}" --out "$BENCH/refs-$tag" | tail -n 1
    touch "$BENCH/refs-$tag/DONE"
done
echo "[$(date '+%F %T')] refs done"
