#!/usr/bin/env bash
# run2 of the verifier QLoRA pipeline on box0 GPU1 (same QLoRA settings as run1). Data = rec_v2: rec_v1 + 772
# release-open positives (RELEASE-phase demo frames with the jaw past the env's 0.5 rad success threshold and the
# object already at its final in-bowl pose) and 45% of in_bowl samples asked with the "visual" wording.
# Why: run1 arm A cut v2 false-success 12/50 -> 3/50 but called set.json-v1 release-instant positives "not released"
# (missed 41/45 vs base 15/45) -- rec_v1 had 805 hover_low hard negatives and only 117 release-instant positives.
# Stages: t_<arm>2 -> e_<arm>2 for ARMS (default "A B"). Base was scored in run1 (+ heldout2 in run1's post-pass).
# Usage: RUN=run2 DATA=rec_v2 nohup setsid bash run_box0_v2.sh <B0 root> > <B0>/runs/run2.out 2>&1 &
set -u
B0=${1:?box0 root}
cd "$B0"
export CUDA_VISIBLE_DEVICES=${GPU:-1} HF_HOME=/media/dylan-matlow/BigU/AI/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 UNSLOTH_COMPILE_LOCATION=$B0/unsloth_compiled_cache TOKENIZERS_PARALLELISM=false
PY=$B0/venv/bin/python
C=$B0/code
D=$B0/data/${DATA:-rec_v2}
T=$B0/data/testsets
R=$B0/runs/${RUN:-run2}
mkdir -p "$R"/{status,logs,eval}
EPOCHS=${EPOCHS:-1}
status() { printf '{"stage":"%s","status":"%s","note":"%s","time":"%s"}\n' "$1" "$2" "${3//\"/\'}" "$(date -Is)" > "$R/status/$1.json"
           echo "[$(date +%H:%M:%S)] STAGE $1 -> $2 $3" | tee -a "$R/logs/orchestrator.log"; }
avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
echo "start $(date -Is) MemAvailable=$(avail_gb)GB GPU=$CUDA_VISIBLE_DEVICES EPOCHS=$EPOCHS DATA=$D ARMS=${ARMS:-A B}" | tee -a "$R/logs/orchestrator.log"
SETS="v2=$T/set_v2.json,v1=$T/set.json,heldout2=$B0/data/rec_v1/test_heldout_v2.json,val=$D/val_final.json,skills72=$T/skills72.json,sugar=$T/test_sugar_300.json"
ev() {
  status "e_$1" running "$2"
  timeout 7200 $PY $C/eval_verifier.py --name "$1" --model "$2" --sets "$3" --out "$R/eval" --bs ${EVAL_BS:-8} \
    > "$R/logs/eval_$1.log" 2>&1
  local rc=$?; [ $rc -eq 0 ] && status "e_$1" ok "$(grep -c '^EVAL' "$R/logs/eval_$1.log") summaries" \
    || status "e_$1" failed "rc=$rc see logs/eval_$1.log"
  return $rc
}
tr_() {
  local tag=$1 arm=$2; shift 2
  status "t_$tag" running "arm $arm $*"
  timeout 14400 $PY $C/train_qlora.py --arm "$arm" --data "$D" --out "$R/arm_$tag" --epochs "$EPOCHS" "$@" \
    > "$R/logs/train_$tag.log" 2>&1
  local rc=$?; [ $rc -eq 0 ] && status "t_$tag" ok "$(grep TRAIN_DONE "$R/logs/train_$tag.log" | cut -c1-300)" \
    || status "t_$tag" failed "rc=$rc see logs/train_$tag.log"
  return $rc
}
for arm in ${ARMS:-A B}; do
  tr_ "${arm}2" "$arm" && ev "${arm}2" "$R/arm_${arm}2/adapter" "$SETS"
done
status DONE ok "all stages attempted; see status/*.json"
