#!/usr/bin/env bash
# One RL v4 arm (D46), end to end, in one process tree so one systemd unit covers it:
#
#   systemd-run --user --unit=patrol-v4-<arm> -p MemoryMax=16G --working-directory=/mnt/work/AI/patrol-lab \
#       tools/v4_arm.sh <arm> <hours> <train.py args...>
#
#   0. (arm A3 only, BC=1) behaviour-clone the planner from $TEACHER into $OUT/bc.pt, sanity-eval it on validation
#      seeds, and warm-start training from it (--init-ckpt is added by this script)
#   1. train      CUDA_VISIBLE_DEVICES=$GPU, checkpoints every $CKPT_MIN minutes
#   2. scan       every checkpoint on the validation seeds 21-60, live hard tier, rl only           (selection data)
#   3. select     tools/v4_select.py: the pre-registered rule in DECISIONS.md D46, against the planner's validation run
#   4. held out   the selected checkpoint once on the sealed seeds 121-200, live hard tier, rl only
# Evals run on CPU under one shared lock ($LOCK) so arms finishing together don't stack 24-worker pools.
# Every stage writes $OUT/status.json ({"stage", "state": running|ok|failed, "detail"}); a failed stage stops this arm
# only. The planner / control references and the hybrid are separate units (tools/v4_refs.sh, tools/v4_hybrid.sh).
set -euo pipefail

ARM=${1:?usage: v4_arm.sh <arm> <hours> <train args...>}
HOURS=${2:?}
shift 2
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}  # set REPO when running a frozen copy (D46 amendment 2)
PY=${ISAAC_PY:-/mnt/weights/ai/isaac/IsaacLab/.venv/bin/python}
ROOT=${V4_ROOT:-/mnt/weights/ai/patrol-lab-data/rl/v4}
OUT="$ROOT/$ARM"
BENCH="$ROOT/bench"
LOCK="$ROOT/eval.lock"
GPU=${GPU:-0}
CKPT_MIN=${CKPT_MIN:-30}
VAL_SEEDS=${VAL_SEEDS:-21-60}
TEST_SEEDS=${TEST_SEEDS:-121-200}
WORKERS=${WORKERS:-24}
cd "$REPO"
export PYTHONPATH="$REPO"
mkdir -p "$OUT" "$BENCH"

status() { printf '{"arm": "%s", "stage": "%s", "state": "%s", "detail": "%s", "at": "%s"}\n' \
    "$ARM" "$1" "$2" "${3:-}" "$(date '+%F %T')" > "$OUT/status.json"; echo "[$(date '+%F %T')] $ARM $1 $2 ${3:-}"; }
trap 'status "${STAGE:-?}" failed "exit $? at line $LINENO"' ERR

evalrun() {  # evalrun <ckpt> <seeds> <controllers> <outdir>
    flock "$LOCK" env CUDA_VISIBLE_DEVICES="" "$PY" -m benchmarks.avoidance.rl.eval --ckpt "$1" --seeds "$2" \
        --controllers "$3" --live hard --workers "$WORKERS" --out "$4" | tail -n 1
}

EXTRA=()
if [[ -n "${INIT_CKPT:-}" ]]; then  # a warm start built outside this script (D46 amendment: the DAgger clone)
    EXTRA=(--init-ckpt "$INIT_CKPT" --critic-warmup "${CRITIC_WARMUP:-60}")
elif [[ "${BC:-0}" == 1 ]]; then
    STAGE=bc; status bc running "$TEACHER"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" tools/bc_train.py --data "${TEACHER:?}" --out "$OUT/bc.pt" --epochs "${BC_EPOCHS:-12}"
    evalrun "$OUT/bc.pt" "$VAL_SEEDS" rl "$BENCH/$ARM-bc-val"
    status bc ok "$OUT/bc.pt"
    EXTRA=(--init-ckpt "$OUT/bc.pt" --critic-warmup "${CRITIC_WARMUP:-60}")
fi

STAGE=train; status train running "$HOURS h on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU "$PY" -m benchmarks.avoidance.rl.train --out "$OUT/run" --hours "$HOURS" \
    --ckpt-min "$CKPT_MIN" "${EXTRA[@]}" "$@" > "$OUT/train.log" 2>&1
test -f "$OUT/run/DONE"
status train ok "$(cat "$OUT/run/DONE")"

STAGE=scan; status scan running "$VAL_SEEDS"
for ck in "$OUT"/run/ckpt_*.pt; do
    name=$(basename "$ck" .pt)
    [[ -f "$BENCH/$ARM-scan/$name/DONE" ]] && continue
    evalrun "$ck" "$VAL_SEEDS" rl "$BENCH/$ARM-scan/$name"
    touch "$BENCH/$ARM-scan/$name/DONE"
done
status scan ok

STAGE=select; status select running
"$PY" tools/v4_select.py --scan "$BENCH/$ARM-scan" --ref "$BENCH/refs-val" --out "$OUT/selected.json"
SEL=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['ckpt'])" "$OUT/selected.json")
status select ok "$SEL"

STAGE=heldout; status heldout running "$SEL on $TEST_SEEDS"
evalrun "$SEL" "$TEST_SEEDS" rl "$BENCH/$ARM-heldout"
status heldout ok
touch "$OUT/ARM_DONE"
