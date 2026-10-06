#!/usr/bin/env bash
# Post-pass after run_box0.sh: (1) every model on the refreshed held-out-object set (test_heldout_v2.json, which adds
# the night-2 held-out evals that finished after run1 started: 12 positives); (2) batch-size control -- base and A
# re-scored on set_v2 with bs=1 (no padding) to show the bs=8 left-padded numbers are not a batching artefact.
set -u
B0=${1:?box0 root}
cd "$B0"
export CUDA_VISIBLE_DEVICES=${GPU:-1} HF_HOME=/media/dylan-matlow/BigU/AI/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 UNSLOTH_COMPILE_LOCATION=$B0/unsloth_compiled_cache TOKENIZERS_PARALLELISM=false
PY=$B0/venv/bin/python; C=$B0/code; D=$B0/data/rec_v1; T=$B0/data/testsets; R=$B0/runs/${RUN:-run1}
mkdir -p "$R/eval_post" "$R/logs"
st() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$R/logs/post.log"; }
for m in base:google/gemma-4-12B-it A:$R/arm_A/adapter B:$R/arm_B/adapter Ans:$R/arm_Ans/adapter; do
  n=${m%%:*}; p=${m#*:}
  [ "$n" != base ] && [ ! -d "$p" ] && { st "skip $n (no adapter)"; continue; }
  st "heldout2 $n"
  timeout 3600 $PY $C/eval_verifier.py --name "$n" --model "$p" --sets "heldout2=$D/test_heldout_v2.json" \
    --out "$R/eval_post" --bs 8 > "$R/logs/post_heldout2_$n.log" 2>&1; st "heldout2 $n rc=$?"
done
for m in base:google/gemma-4-12B-it A:$R/arm_A/adapter; do
  n=${m%%:*}; p=${m#*:}
  st "bs1 $n"
  timeout 3600 $PY $C/eval_verifier.py --name "${n}_bs1" --model "$p" --sets "v2=$T/set_v2.json" --out "$R/eval_post" \
    --bs 1 > "$R/logs/post_bs1_$n.log" 2>&1; st "bs1 $n rc=$?"
done
st POST_DONE
