#!/usr/bin/env bash
# Overnight pipeline (GPU 0 only): demos -> LeRobot dataset -> FLUX SO-101 LoRA -> closed-loop evals -> report.
#
#   bash tools/overnight.sh [tiny]        # "tiny" = end-to-end dry run with minimal sizes (gate G6)
#
# Rules: stages chain on exit codes (never on log text); each stage writes <RUN>/status/<stage>.json;
# a failed stage is recorded and later stages that don't depend on it still run.
set -u
MODE="${1:-full}"
STAMP=$(date +%Y%m%d-%H%M)
RUN=/mnt/weights/ai/robot-lab-data/overnight/${MODE}-${STAMP}
mkdir -p "$RUN"/{status,logs,raw,eval}
ln -sfn "$RUN" /mnt/weights/ai/robot-lab-data/overnight/latest-${MODE}

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

# ---------------- S1: scripted demos per training object ----------------
S1_OK=0
for obj in "${TRAIN_OBJECTS[@]}"; do
  t=$(tag "$obj")
  (cd "$ISAAC" && timeout 5400 uv run --extra teleop python "$LAB/tools/record_demos.py" --num_envs $DEMO_ENVS \
      --episodes $DEMO_EPS --object "$obj" --seed 7 --out "$RUN/raw/$t" > "$RUN/logs/demos_$t.log" 2>&1)
  rc=$?
  n=$(grep -c 'success=True' "$RUN/logs/demos_$t.log" 2>/dev/null || echo 0)
  if [ $rc -eq 0 ] && [ "$n" -gt 0 ]; then S1_OK=$((S1_OK+1)); status "s1_demos_$t" ok "$n successes"; else status "s1_demos_$t" failed "rc=$rc successes=$n"; fi
done

# ---------------- S2: convert to LeRobot v3 ----------------
DS=$RUN/lerobot_ds
if [ $S1_OK -gt 0 ]; then
  RAWS=(); for obj in "${TRAIN_OBJECTS[@]}"; do [ -d "$RUN/raw/$(tag "$obj")" ] && RAWS+=("$RUN/raw/$(tag "$obj")"); done
  (cd "$LAB" && timeout 7200 $LRPY tools/to_lerobot.py --raw "${RAWS[@]}" --out "$DS" --repo_id local/so101_pickplace \
      > "$RUN/logs/convert.log" 2>&1)
  rc=$?; [ $rc -eq 0 ] && status s2_convert ok "$(grep -o 'TIMING.*' "$RUN/logs/convert.log")" || status s2_convert failed "rc=$rc"
else
  status s2_convert skipped "no demos"; rc=1
fi
S2_RC=$rc

# ---------------- S3: FLUX SO-101 LoRA ----------------
LORA=$RUN/lora
if [ $S2_RC -eq 0 ]; then
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
