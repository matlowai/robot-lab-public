#!/usr/bin/env bash
# RL v4 (D46) hybrid arm "rl_shield" (benchmarks/avoidance/rl/hybrid.py), always reported as a hybrid.
# Waits for the pure arms (ARMS, default "A0 A1 A2 A3"), takes the pure arm with the best validation score (each arm's
# selected.json, tools/v4_select.py rule) and evaluates THAT checkpoint wrapped in the shield: validation seeds, then
# the sealed test seeds. No separate checkpoint selection for the hybrid (pre-registered).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${ISAAC_PY:-/mnt/weights/ai/isaac/IsaacLab/.venv/bin/python}
ROOT=${V4_ROOT:-/mnt/weights/ai/patrol-lab-data/rl/v4}
BENCH="$ROOT/bench"; LOCK="$ROOT/eval.lock"; ARMS=${ARMS:-"A0 A1 A2 A3"}
DEADLINE=$(( $(date +%s) + ${WAIT_H:-12} * 3600 ))
cd "$REPO"; export PYTHONPATH="$REPO"; mkdir -p "$ROOT/H"
while :; do
    pending=0; for a in $ARMS; do [[ -f "$ROOT/$a/ARM_DONE" || -f "$ROOT/$a/status.json" && $(grep -c '"failed"' "$ROOT/$a/status.json") -gt 0 ]] || pending=1; done
    [[ $pending == 0 ]] && break
    [[ $(date +%s) -gt $DEADLINE ]] && { echo "hybrid: gave up waiting"; exit 1; }
    sleep 120
done
BEST=$("$PY" - "$ROOT" $ARMS <<'PYEOF'
import json, sys, pathlib
root, arms = pathlib.Path(sys.argv[1]), sys.argv[2:]
c = []
for a in arms:
    p = root / a / "selected.json"
    if (root / a / "ARM_DONE").exists() and p.exists():
        s = json.load(open(p)); c.append((s["score"], a, s["ckpt"]))
best = min(c)  # lowest validation score; ties -> earlier arm in ARMS order (stable)
print(f"{best[1]} {best[2]}")
PYEOF
)
ARM=${BEST%% *}; CK=${BEST#* }
echo "{\"from_arm\": \"$ARM\", \"ckpt\": \"$CK\"}" > "$ROOT/H/selected.json"
for pair in "val:${VAL_SEEDS:-21-60}" "heldout:${TEST_SEEDS:-121-200}"; do
    tag=${pair%%:*}; seeds=${pair#*:}
    flock "$LOCK" env CUDA_VISIBLE_DEVICES="" "$PY" -m benchmarks.avoidance.rl.eval --ckpt "$CK" --seeds "$seeds" \
        --controllers rl_shield --live hard --workers "${WORKERS:-24}" --out "$BENCH/H-$tag" | tail -n 1
done
touch "$ROOT/H/ARM_DONE"; echo "[$(date '+%F %T')] hybrid done ($ARM)"
"$PY" tools/v4_report.py --root "$ROOT" > "$ROOT/report.log" 2>&1 || true
