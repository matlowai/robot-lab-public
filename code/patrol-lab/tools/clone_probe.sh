#!/usr/bin/env bash
# D47 clone probe: why did the planner clone fail to avoid people? The D46 recipe (BC on the planner, then 3 DAgger
# rounds, round 3 fixed in advance) re-run with planner-derived observations (rl/track_features.py), validation only:
#
#   systemd-run --user --unit=patrol-probe-<ver> -p MemoryMax=24G --working-directory=/mnt/work/AI/patrol-lab \
#       tools/clone_probe.sh <avoid-v2t|avoid-v2ts>
#
# Seeds are the D46 ones (teacher 1000-1599, rounds 1600-2199, 2200-2799, 2800-3399, validation 21-60); the sealed
# test seeds are never run. The unit runs a frozen copy of this script, so editing it mid-run is safe (D46 amendment 2).
# Re-running resumes: every stage whose output exists is skipped. GPU= (empty) fits on CPU.
set -euo pipefail
VER=${1:?usage: clone_probe.sh <obs version>}
if [[ -z "${PROBE_FROZEN:-}" ]]; then  # re-exec from a private copy: bash reads scripts lazily
    ROOT=${PROBE_ROOT:-/mnt/weights/ai/patrol-lab-data/rl/clone-probe}/$VER
    mkdir -p "$ROOT"
    cp "$0" "$ROOT/clone_probe.sh"
    exec env PROBE_FROZEN=1 REPO="$(cd "$(dirname "$0")/.." && pwd)" bash "$ROOT/clone_probe.sh" "$@"
fi
PY=${ISAAC_PY:-/mnt/weights/ai/isaac/IsaacLab/.venv/bin/python}
ROOT=${PROBE_ROOT:-/mnt/weights/ai/patrol-lab-data/rl/clone-probe}/$VER
LOCK=${PROBE_ROOT:-/mnt/weights/ai/patrol-lab-data/rl/clone-probe}/eval.lock
WORKERS=${WORKERS:-56}
cd "$REPO"; export PYTHONPATH="$REPO"

val() {  # val <ckpt> <tag>
    [[ -n $(ls "$ROOT/val-$2"/*/summary.json 2>/dev/null) ]] && return
    flock "$LOCK" env CUDA_VISIBLE_DEVICES="" "$PY" -m benchmarks.avoidance.rl.eval --ckpt "$1" --seeds 21-60 \
        --controllers rl --live hard --workers 24 --out "$ROOT/val-$2" | tail -n 1
}
fit() {  # fit <data dirs> <out>
    [[ -f "${2%.pt}.json" ]] && return
    CUDA_VISIBLE_DEVICES=${GPU-0} "$PY" tools/bc_train.py --data "$1" --out "$2" --epochs 12 | tail -n 2
}

echo "[$(date '+%F %T')] $VER: teacher on 1000-1599"
[[ -f "$ROOT/teacher/summary.json" ]] || "$PY" tools/collect_teacher.py --seeds 1000-1599 --live hard --workers "$WORKERS" --obs-version "$VER" \
    --out "$ROOT/teacher" | tail -n 12
DATA="$ROOT/teacher"
fit "$DATA" "$ROOT/bc0.pt"
val "$ROOT/bc0.pt" bc0
DRIVER="$ROOT/bc0.pt"
SEEDS=("1600-2199" "2200-2799" "2800-3399")
for k in 1 2 3; do
    echo "[$(date '+%F %T')] $VER: DAgger round $k, driver $DRIVER, seeds ${SEEDS[$((k-1))]}"
    [[ -f "$ROOT/round$k/summary.json" ]] || "$PY" tools/collect_teacher.py --seeds "${SEEDS[$((k-1))]}" --live hard --workers "$WORKERS" --driver "$DRIVER" \
        --out "$ROOT/round$k" | tail -n 12
    DATA="$DATA,$ROOT/round$k"
    fit "$DATA" "$ROOT/bc$k.pt"
    val "$ROOT/bc$k.pt" "bc$k"
    DRIVER="$ROOT/bc$k.pt"
done
touch "$ROOT/DONE"
echo "[$(date '+%F %T')] $VER: done"
