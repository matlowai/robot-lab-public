#!/usr/bin/env bash
# D46 amendment: DAgger for arm A3's warm start, then launch A3 from the round-3 clone (fixed in advance: no picking).
# Round k: the round-(k-1) clone drives sim2d live-hard patrols on fresh seeds while the planner labels every step;
# retrain BC on all data so far; score the clone on the validation seeds (record only).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${ISAAC_PY:-/mnt/weights/ai/isaac/IsaacLab/.venv/bin/python}
ROOT=${V4_ROOT:-/mnt/weights/ai/patrol-lab-data/rl/v4}
D="$ROOT/dagger"; BENCH="$ROOT/bench"; LOCK="$ROOT/eval.lock"
cd "$REPO"; export PYTHONPATH="$REPO"; mkdir -p "$D"
DATA="$ROOT/teacher-heuristic-hard-s1000-1599"
DRIVER="$ROOT/A3/bc.pt"
SEEDS=("1600-2199" "2200-2799" "2800-3399")
for k in 1 2 3; do
    echo "[$(date '+%F %T')] DAgger round $k: driver $DRIVER, seeds ${SEEDS[$((k-1))]}"
    "$PY" tools/collect_teacher.py --seeds "${SEEDS[$((k-1))]}" --live hard --workers 64 --driver "$DRIVER" \
        --out "$D/round$k" | tail -n 12
    DATA="$DATA,$D/round$k"
    CUDA_VISIBLE_DEVICES=0 "$PY" tools/bc_train.py --data "$DATA" --out "$D/bc$k.pt" --epochs 12 | tail -n 2
    flock "$LOCK" env CUDA_VISIBLE_DEVICES="" "$PY" -m benchmarks.avoidance.rl.eval --ckpt "$D/bc$k.pt" --seeds 21-60 \
        --controllers rl --live hard --workers 24 --out "$BENCH/A3-dagger$k-val" | tail -n 1
    DRIVER="$D/bc$k.pt"
done
cp "$D/bc3.pt" "$ROOT/A3/init.pt"
END=$(date -d "${END_AT:-2026-10-02 05:30}" +%s)
HOURS=$(python3 -c "print(round(max(4.0, ($END - $(date +%s)) / 3600), 2))")
echo "[$(date '+%F %T')] launching A3 from round-3 clone for $HOURS h"
rm -rf "$ROOT/A3/run" "$ROOT/A3/status.json"
exec env GPU=0 INIT_CKPT="$ROOT/A3/init.pt" CRITIC_WARMUP=200 tools/v4_arm.sh A3 "$HOURS" \
    --preset hard-dense --obs avoid-v2 --set w_space=0 --seed 0 --squash clip --log-std-max 0.0 --bound-coef 0.01 \
    --set w_ttc=0.1 --ramp w_ttc=0:0.2:0.4 --lr 1e-4
