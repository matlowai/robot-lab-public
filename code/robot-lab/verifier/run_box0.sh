#!/usr/bin/env bash
# Verifier QLoRA pipeline on box0 (operator 2026-10-06: "Do gemma on box0"). One GPU (box0 GPU1, freed by stopping
# vlm-qwen38-27b; GPU0 keeps serving vlm-gemma4-12b-64k for skills-v0). Sequential stages, status file per stage,
# chain on exit codes:
#   e_base -> t_A -> e_A -> t_B -> e_B -> t_Ans (arm A without sugar box: leave-one-object-out probe) -> e_Ans -> DONE
# Usage: nohup setsid bash run_box0.sh <B0 root> > <B0>/run.log 2>&1 &
set -u
B0=${1:?box0 root}
cd "$B0"
export CUDA_VISIBLE_DEVICES=${GPU:-1} HF_HOME=/media/dylan-matlow/BigU/AI/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 UNSLOTH_COMPILE_LOCATION=$B0/unsloth_compiled_cache TOKENIZERS_PARALLELISM=false
PY=$B0/venv/bin/python
C=$B0/code
D=$B0/data/rec_v1
T=$B0/data/testsets
R=$B0/runs/${RUN:-run1}
mkdir -p "$R"/{status,logs,eval}
EPOCHS=${EPOCHS:-1}
status() { printf '{"stage":"%s","status":"%s","note":"%s","time":"%s"}\n' "$1" "$2" "${3//\"/\'}" "$(date -Is)" > "$R/status/$1.json"
           echo "[$(date +%H:%M:%S)] STAGE $1 -> $2 $3" | tee -a "$R/logs/orchestrator.log"; }
avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
echo "start $(date -Is) MemAvailable=$(avail_gb)GB GPU=$CUDA_VISIBLE_DEVICES EPOCHS=$EPOCHS" | tee -a "$R/logs/orchestrator.log"
SETS="v2=$T/set_v2.json,v1=$T/set.json,heldout=$D/test_heldout.json,val=$D/val_final.json,skills72=$T/skills72.json,sugar=$T/test_sugar_300.json"

ev() {  # ev <name> <model or adapter> <sets>
  status "e_$1" running "$2"
  timeout 7200 $PY $C/eval_verifier.py --name "$1" --model "$2" --sets "$3" --out "$R/eval" --bs ${EVAL_BS:-8} \
    > "$R/logs/eval_$1.log" 2>&1
  local rc=$?; [ $rc -eq 0 ] && status "e_$1" ok "$(grep -c '^EVAL' "$R/logs/eval_$1.log") summaries" \
    || status "e_$1" failed "rc=$rc see logs/eval_$1.log"
  return $rc
}
tr_() {  # tr_ <tag> <arm> [extra args]
  local tag=$1 arm=$2; shift 2
  status "t_$tag" running "arm $arm $*"
  timeout 14400 $PY $C/train_qlora.py --arm "$arm" --data "$D" --out "$R/arm_$tag" --epochs "$EPOCHS" "$@" \
    > "$R/logs/train_$tag.log" 2>&1
  local rc=$?; [ $rc -eq 0 ] && status "t_$tag" ok "$(grep TRAIN_DONE "$R/logs/train_$tag.log" | cut -c1-300)" \
    || status "t_$tag" failed "rc=$rc see logs/train_$tag.log"
  return $rc
}
[ -f "$R/status/e_base.json" ] && grep -q '"ok"' "$R/status/e_base.json" || ev base google/gemma-4-12B-it "$SETS"
tr_ A A && ev A "$R/arm_A/adapter" "$SETS"
tr_ B B && ev B "$R/arm_B/adapter" "$SETS"
tr_ Ans A --exclude_object "sugar box" && ev Ans "$R/arm_Ans/adapter" "$SETS"
status DONE ok "all stages attempted; see status/*.json"
