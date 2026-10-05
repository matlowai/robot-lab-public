#!/usr/bin/env bash
# Resume an overnight run from S3 (LoRA) -> S4 evals -> S5 report, reusing its demos + dataset.
#   bash tools/overnight_resume.sh <RUN_DIR> [full|tiny] [min_free_gb]
# Overnight pipeline (GPU 0 only): demos -> LeRobot dataset -> FLUX SO-101 LoRA -> closed-loop evals -> report.
#
#   bash tools/overnight.sh [tiny]        # "tiny" = end-to-end dry run with minimal sizes (gate G6)
#
# Rules: stages chain on exit codes (never on log text); each stage writes <RUN>/status/<stage>.json;
# a failed stage is recorded and later stages that don't depend on it still run.
set -u
RUN="$1"                      # existing run dir (demos + dataset already there)
MODE="${2:-full}"
MIN_FREE_GB="${3:-30}"        # wait for this much MemAvailable before loading FLUX (the load spike is ~25 GB)
[ -d "$RUN/lerobot_ds" ] || { echo "no dataset in $RUN"; exit 1; }

ISAAC=/mnt/weights/ai/isaac/IsaacLab
LEROBOT=/mnt/weights/ai/lerobot/lerobot
LAB=/mnt/work/AI/robot-lab
LRPY=$LEROBOT/.venv/bin/python
FXPY=/mnt/work/AI/flux3-action/flux-action/.venv/bin/python
BASE_POLICY=/mnt/weights/ai/flux3-action/flux-3-action-so101
export CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1

TRAIN_OBJECTS=("mug" "blue block" "soup can" "sugar box")
HELDOUT_OBJECTS=("mustard bottle" "cracker box")
if [ "$MODE" = tiny ]; then
  DEMO_ENVS=4; DEMO_EPS=4; LORA_STEPS=8; LORA_SAVE=4; EVAL_EPS=1; EVAL_SECS=6; MID_CKPT=000004
else
  DEMO_ENVS=16; DEMO_EPS=480; LORA_STEPS=10000; LORA_SAVE=2500; EVAL_EPS=12; EVAL_SECS=20; MID_CKPT=005000
fi

status() {  # status <stage> <ok|failed|skipped> <note>
  printf '{"stage":"%s","status":"%s","note":"%s","time":"%s"}\n' "$1" "$2" "$3" "$(date -Is)" > "$RUN/status/$1.json"
  echo "[$(date +%H:%M:%S)] STAGE $1 -> $2 $3" | tee -a "$RUN/logs/orchestrator.log"
}
tag() { echo "$1" | tr ' ' '_'; }

# ---------------- S3: FLUX SO-101 LoRA ----------------
LORA=$RUN/lora
DS=$RUN/lerobot_ds
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge "$MIN_FREE_GB" ]; do
  echo "[$(date +%H:%M:%S)] waiting for ${MIN_FREE_GB} GB free RAM before loading FLUX" | tee -a "$RUN/logs/orchestrator.log"; sleep 60
done
RESUME_CFG="$LORA/checkpoints/last/pretrained_model/train_config.json"
if [ -f "$RESUME_CFG" ]; then
  # A checkpoint exists: continue it. LeRobot restores optimizer, LR scheduler, EMA, RNG and step count
  # from checkpoints/last/training_state (docs: "Resume from the raw pretrained_model checkpoint").
  echo "[$(date +%H:%M:%S)] resuming LoRA from $(readlink -f "$LORA/checkpoints/last")" | tee -a "$RUN/logs/orchestrator.log"
  (cd "$LEROBOT" && timeout 18000 $LRPY -m lerobot.scripts.lerobot_train --config_path="$RESUME_CFG" --resume=true \
      >> "$RUN/logs/lora.log" 2>&1)
  rc=$?; [ $rc -eq 0 ] && status s3_lora ok "resumed; $(grep -o 'eval_loss=[0-9.]*' "$RUN/logs/lora.log" | tail -1)" || status s3_lora failed "resume rc=$rc"
elif true; then
  rm -rf "$LORA"  # died before the first checkpoint: nothing to keep
  (cd "$LEROBOT" && timeout 18000 $LRPY -m lerobot.scripts.lerobot_train --config_path=examples/flux3/lora.json \
      --policy.path=$BASE_POLICY --policy.device=cuda \
      --policy.video_vae_id=/mnt/weights/ai/flux3-action/flux-3-action-base/video_vae.safetensors \
      --policy.text_encoder_id=/mnt/weights/ai/flux3-action/flux-3-action-base/text_encoder \
      --dataset.repo_id=local/so101_pickplace --dataset.root="$DS" --output_dir="$LORA" \
      --steps=$LORA_STEPS --save_freq=$LORA_SAVE --eval_steps=$LORA_SAVE --max_eval_samples=400 \
      --log_freq=20 --num_workers=2 > "$RUN/logs/lora.log" 2>&1)
  rc=$?; [ $rc -eq 0 ] && status s3_lora ok "$(grep -o 'eval_loss=[0-9.]*' "$RUN/logs/lora.log" | tail -1)" || status s3_lora failed "rc=$rc"
else
  status s3_lora skipped "no dataset"; rc=1
fi
S3_RC=$rc

# ---------------- S4: closed-loop evals ----------------
UNITS=$DS/units.json
[ -f "$UNITS" ] || UNITS=/mnt/weights/ai/robot-lab-data/lerobot/pilot_multi/units.json
run_eval() {  # run_eval <label> <policy_dir> <objects...>
  local label=$1 pol=$2; shift 2
  local out=$RUN/eval/$label; mkdir -p "$out"
  if [ ! -d "$pol" ]; then status "s4_eval_$label" skipped "no policy dir $pol"; return; fi
  (cd "$LAB" && exec $LRPY tools/policy_server.py --policy "$pol" --port 6100) > "$out/server.log" 2>&1 &
  local spid=$!
  local t0=$(date +%s)
  until grep -qE "SERVER ready|Traceback" "$out/server.log" 2>/dev/null || ! kill -0 $spid 2>/dev/null; do
    sleep 3; [ $(( $(date +%s) - t0 )) -gt 900 ] && break
  done
  if ! grep -q "SERVER ready" "$out/server.log"; then status "s4_eval_$label" failed "server did not start"; kill $spid 2>/dev/null; return; fi
  local okn=0
  for obj in "$@"; do
    (cd "$ISAAC" && timeout 5400 uv run --extra teleop python "$LAB/tools/eval_policy.py" --object "$obj" \
        --episodes $EVAL_EPS --max_seconds $EVAL_SECS --units "$UNITS" --out "$out" --seed 1000 \
        > "$out/eval_$(tag "$obj").log" 2>&1) && okn=$((okn+1))
  done
  kill $spid 2>/dev/null; wait $spid 2>/dev/null
  status "s4_eval_$label" ok "$okn/$# objects evaluated"
}
run_eval base_train "$BASE_POLICY" "${TRAIN_OBJECTS[@]}"
if [ $S3_RC -eq 0 ]; then
  run_eval lora_mid_ema "$LORA/checkpoints/$MID_CKPT/pretrained_model_ema" "${TRAIN_OBJECTS[@]}"
  run_eval lora_final_ema "$LORA/checkpoints/last/pretrained_model_ema" "${TRAIN_OBJECTS[@]}" "${HELDOUT_OBJECTS[@]}"
  run_eval lora_final_raw "$LORA/checkpoints/last/pretrained_model" "${TRAIN_OBJECTS[@]}"
fi
run_eval base_heldout "$BASE_POLICY" "${HELDOUT_OBJECTS[@]}"

# ---------------- S5: report ----------------
(cd "$LAB" && $FXPY tools/morning_report.py --run "$RUN" > "$RUN/logs/report.log" 2>&1)
rc=$?; [ $rc -eq 0 ] && status s5_report ok "$RUN/MORNING_REPORT.md" || status s5_report failed "rc=$rc"
echo "DONE $RUN" | tee -a "$RUN/logs/orchestrator.log"
