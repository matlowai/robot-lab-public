#!/usr/bin/env bash
# Overnight GR00T N1.7 pipeline (GPU 0 only), mirroring tools/overnight.sh (FLUX):
#   S0 harness replay check -> S1 convert (FLUX's 909-episode dataset -> GR00T v2.1) -> S2 GR00T N1.7 post-training
#   (official SO-100/101 recipe, mid + final checkpoints) -> S3 STRICT closed-loop evals (mid: training objects;
#   final: training + held-out objects) -> S4 morning report (strict real successes are the headline).
#
#   bash tools/overnight_gr00t.sh [tiny|full]
#
# Sizes are env-overridable (STEPS, SAVE, BATCH, EVAL_EPS, EVAL_SECS, REPLAY_EPS, BASE_MODEL, SRC_DS, EXEC_H).
# Rules (lessons from the FLUX night, course Modules 6-7): stages chain on exit codes, never on log text; each stage
# writes <RUN>/status/<stage>.json; a failed stage is recorded and stages that don't depend on it still run; the
# scripts are copied into <RUN>/code at launch and run from there (never edit a running script); every server this
# script starts is killed by the PID it recorded, never by pattern.
set -u
if [ -z "${GR00T_FROZEN:-}" ]; then  # bash reads scripts lazily: run from a frozen copy so edits can't hit a live run
  mkdir -p /mnt/weights/ai/nvidia-action/tmp
  F=$(mktemp /mnt/weights/ai/nvidia-action/tmp/overnight_gr00t.XXXXXX.sh) && cp "$0" "$F" && GR00T_FROZEN=1 exec bash "$F" "$@"
fi
MODE="${1:-full}"
STAMP=$(date +%Y%m%d-%H%M)
RUN=/mnt/weights/ai/robot-lab-data/overnight-gr00t/${MODE}-${STAMP}
mkdir -p "$RUN"/{status,logs,eval,code}
ln -sfn "$RUN" /mnt/weights/ai/robot-lab-data/overnight-gr00t/latest-${MODE}

NA=/mnt/weights/ai/nvidia-action
GR=$NA/Isaac-GR00T
GRPY=$GR/.venv/bin/python
ISAAC=/mnt/weights/ai/isaac/IsaacLab
LAB=/mnt/work/AI/robot-lab
FLUX_RUN=/mnt/weights/ai/robot-lab-data/overnight/full-20260923-2207
SRC_DS=${SRC_DS:-$FLUX_RUN/lerobot_ds}           # the 909-episode / 130,980-frame FLUX training set
RAW=$FLUX_RUN/raw                                # its recorded npz (replay check + pixel alignment check)
BASE_MODEL=${BASE_MODEL:-$NA/models/GR00T-N1.7-3B}
FF7=$NA/ffmpeg7/ffmpeg-n7.1.5-12-g1fdbca85aa-linux64-gpl-shared-7.1/lib
PORT=${PORT:-6120}
EXEC_H=${EXEC_H:-8}

export CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export HF_HOME=$NA/hf HF_HUB_CACHE=$NA/hf/hub HF_XET_CACHE=$NA/hf/xet TMPDIR=$NA/tmp UV_CACHE_DIR=$NA/uv-cache
GRENV=(env LD_LIBRARY_PATH="$FF7" PATH="$GR/.venv/bin:$PATH")   # torchcodec 0.8 is built for FFmpeg <= 7

TRAIN_OBJECTS=("mug" "blue block" "soup can" "sugar box")
HELDOUT_OBJECTS=("mustard bottle" "cracker box")
if [ "$MODE" = tiny ]; then
  PER_TASK=${PER_TASK:-3}; STEPS=${STEPS:-40}; SAVE=${SAVE:-20}; BATCH=${BATCH:-16}
  EVAL_EPS=${EVAL_EPS:-1}; EVAL_SECS=${EVAL_SECS:-4}; REPLAY_EPS=${REPLAY_EPS:-2}
  REPLAY_OBJS=("mug" "sugar box"); MID_TRAIN_OBJS=("mug"); FINAL_TRAIN_OBJS=("mug"); FINAL_HELDOUT_OBJS=("cracker box")
else
  # STEPS unset -> S2a calibrates steps/s on this GPU and sizes the run to TRAIN_HOURS (default 6.5 h of training,
  # ~8.5 h end to end with ~1.5 h of evals). Set STEPS (and SAVE) to skip calibration.
  PER_TASK=${PER_TASK:-0}; STEPS=${STEPS:-}; SAVE=${SAVE:-}; BATCH=${BATCH:-32}; TRAIN_HOURS=${TRAIN_HOURS:-6.5}
  EVAL_EPS=${EVAL_EPS:-12}; EVAL_SECS=${EVAL_SECS:-20}; REPLAY_EPS=${REPLAY_EPS:-8}
  REPLAY_OBJS=("${TRAIN_OBJECTS[@]}"); MID_TRAIN_OBJS=("${TRAIN_OBJECTS[@]}")
  FINAL_TRAIN_OBJS=("${TRAIN_OBJECTS[@]}"); FINAL_HELDOUT_OBJS=("${HELDOUT_OBJECTS[@]}")
fi

status() {  # status <stage> <ok|failed|skipped> <note>
  printf '{"stage":"%s","status":"%s","note":"%s","time":"%s"}\n' "$1" "$2" "${3//\"/\'}" "$(date -Is)" > "$RUN/status/$1.json"
  echo "[$(date +%H:%M:%S)] STAGE $1 -> $2 $3" | tee -a "$RUN/logs/orchestrator.log"
}
tag() { echo "$1" | tr ' ' '_'; }
stamp() { date +%s > "$RUN/status/$1.t0"; }
took() { echo $(( $(date +%s) - $(cat "$RUN/status/$1.t0") ))s; }

# ---- frozen copies of everything this run executes ----
cp "$0" "$LAB"/tools/{to_gr00t.py,gr00t_so101_config.py,gr00t_policy_server.py,gr00t_eval.py,gr00t_morning_report.py} "$RUN/code/"
C=$RUN/code
printf 'MODE=%s STEPS=%s SAVE=%s MID=%s BATCH=%s EVAL_EPS=%s EVAL_SECS=%s EXEC_H=%s BASE_MODEL=%s SRC_DS=%s GR00T_COMMIT=%s\n' \
  "$MODE" "${STEPS:-auto}" "${SAVE:-auto}" "-" "$BATCH" "$EVAL_EPS" "$EVAL_SECS" "$EXEC_H" "$BASE_MODEL" "$SRC_DS" \
  "$(git -C "$GR" rev-parse HEAD)" | tee "$RUN/config.txt" | tee -a "$RUN/logs/orchestrator.log"
nvidia-smi --query-gpu=index,name,memory.used,power.limit --format=csv > "$RUN/logs/gpu_at_start.csv"

# ---------------- preflight: gated backbone access + GPU 0 free (loud, never fatal by itself) ----------------
PF=$("$GRPY" - <<'PYEOF' 2>&1
from huggingface_hub import get_hf_file_metadata, hf_hub_url
try:
    get_hf_file_metadata(hf_hub_url("nvidia/Cosmos-Reason2-2B", "config.json")); print("backbone access OK")
except Exception as e:  # noqa: BLE001 -- reported verbatim in the status file
    print("BLOCKED:", type(e).__name__, str(e).splitlines()[-1][:160])
PYEOF
)
GPU0=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)
case "$PF" in *"access OK"*) status s_preflight ok "$PF; GPU0 used $GPU0" ;;
  *) status s_preflight failed "$PF -- accept the terms at https://huggingface.co/nvidia/Cosmos-Reason2-2B; GPU0 used $GPU0" ;; esac

# ---------------- S0: harness check: replay recorded expert demos through the strict eval path ----------------
stamp s0_replay
okn=0
for obj in "${REPLAY_OBJS[@]}"; do
  t=$(tag "$obj")
  (cd "$ISAAC" && timeout 3600 uv run --extra teleop python "$C/gr00t_eval.py" --object "$obj" --episodes $REPLAY_EPS \
      --units "$SRC_DS/units.json" --out "$RUN/eval/replay_expert" --replay "$RAW/$t" \
      > "$RUN/logs/replay_$t.log" 2>&1) && okn=$((okn+1))
done
[ $okn -eq ${#REPLAY_OBJS[@]} ] && status s0_replay ok "$okn/${#REPLAY_OBJS[@]} objects replayed in $(took s0_replay)" \
  || status s0_replay failed "$okn/${#REPLAY_OBJS[@]} objects replayed"

# ---------------- S1: convert + GR00T stats ----------------
stamp s1_convert
DS=$RUN/gr00t_ds
(cd "$LAB" && timeout 3600 "${GRENV[@]}" "$GRPY" "$C/to_gr00t.py" --src "$SRC_DS" --out "$DS" --per_task $PER_TASK \
    --raw_root "$RAW" --check 12 > "$RUN/logs/convert.log" 2>&1) \
  && (cd "$GR" && timeout 3600 "${GRENV[@]}" "$GRPY" gr00t/data/stats.py --dataset-path "$DS" \
      --embodiment-tag NEW_EMBODIMENT --modality-config-path "$C/gr00t_so101_config.py" > "$RUN/logs/stats.log" 2>&1)
S1_RC=$?
[ $S1_RC -eq 0 ] && status s1_convert ok "$(grep -o 'TIMING.*' "$RUN/logs/convert.log"); $(grep -o 'RAWCHECK.*' "$RUN/logs/convert.log"); $(took s1_convert)" \
  || status s1_convert failed "rc=$S1_RC"

# ---------------- S2a: calibrate training speed and size the run (full mode, STEPS unset) ----------------
finetune() {  # finetune <output dir> <max steps> <save steps> <log>
  (cd "$GR" && timeout $(( 14 * 3600 )) "${GRENV[@]}" NUM_GPUS=1 USE_WANDB=0 MAX_STEPS=$2 SAVE_STEPS=$3 \
      GLOBAL_BATCH_SIZE=$BATCH DATALOADER_NUM_WORKERS=4 bash examples/finetune.sh \
      --base-model-path "$BASE_MODEL" --dataset-path "$DS" --modality-config-path "$C/gr00t_so101_config.py" \
      --embodiment-tag NEW_EMBODIMENT --output-dir "$1" > "$4" 2>&1)
}
if { [ -z "$STEPS" ] || [ "$MODE" = tiny ]; } && [ $S1_RC -eq 0 ]; then  # tiny exercises this path at 10 steps
  stamp s2a_calibrate
  CAL_STEPS=150; [ "$MODE" = tiny ] && CAL_STEPS=10
  finetune "$RUN/calibrate" $CAL_STEPS $CAL_STEPS "$RUN/logs/calibrate.log"
  rc=$?
  # steps/s over the whole 150-step run (includes warm-up, so the estimate is conservative: the run ends early, not late)
  SPS=$(grep -o "'train_steps_per_second': [0-9.]*" "$RUN/logs/calibrate.log" | tail -1 | awk '{print $2}')
  if [ $rc -eq 0 ] && [ -n "$SPS" ]; then
    SIZED=$("$GRPY" -c "print(max(1000, int(float('$SPS') * ${TRAIN_HOURS:-6.5} * 3600) // 1000 * 1000))")
    if [ -z "$STEPS" ]; then STEPS=$SIZED; SAVE=$(( STEPS / 4 )); fi
    status s2a_calibrate ok "$SPS steps/s at batch $BATCH over $CAL_STEPS steps -> ${TRAIN_HOURS:-6.5} h = $SIZED steps; using STEPS=$STEPS; $(took s2a_calibrate)"
    rm -rf "$RUN/calibrate"
  else
    status s2a_calibrate failed "rc=$rc sps=${SPS:-none}"
  fi
fi
STEPS=${STEPS:-0}; SAVE=${SAVE:-1}
MID=$(( STEPS / 2 / SAVE * SAVE )); [ "$MID" -lt "$SAVE" ] && MID=$SAVE
echo "SIZES STEPS=$STEPS SAVE=$SAVE MID=$MID" | tee -a "$RUN/config.txt" "$RUN/logs/orchestrator.log"

# ---------------- S2: GR00T N1.7 post-training (Isaac-GR00T examples/finetune.sh, official SO-100/101 recipe) ----------------
stamp s2_finetune
FT=$RUN/finetune
if [ $S1_RC -eq 0 ] && [ "$STEPS" -gt 0 ]; then
  finetune "$FT" $STEPS $SAVE "$RUN/logs/finetune.log"
  S2_RC=$?
  if [ $S2_RC -eq 0 ] && [ ! -d "$FT/checkpoint-$STEPS" ]; then S2_RC=97; fi  # exit 0 but no final checkpoint
  [ $S2_RC -eq 0 ] && status s2_finetune ok "final checkpoint-$STEPS; $(took s2_finetune)" || status s2_finetune failed "rc=$S2_RC"
else
  S2_RC=1; status s2_finetune skipped "no dataset or no step count (S1 / S2a failed)"
fi

# ---------------- S3: strict closed-loop evals ----------------
run_eval() {  # run_eval <label> <checkpoint dir> <objects...>
  local label=$1 pol=$2; shift 2
  local out=$RUN/eval/$label; mkdir -p "$out"
  stamp "s3_eval_$label"
  if [ ! -d "$pol" ]; then status "s3_eval_$label" skipped "no checkpoint $pol"; return; fi
  (cd "$GR" && exec "${GRENV[@]}" "$GRPY" "$C/gr00t_policy_server.py" --policy "$pol" --port $PORT \
      --exec_horizon $EXEC_H) > "$out/server.log" 2>&1 &
  local spid=$! t0=$(date +%s)
  until grep -qE "SERVER ready|Traceback" "$out/server.log" 2>/dev/null || ! kill -0 $spid 2>/dev/null; do
    sleep 3; [ $(( $(date +%s) - t0 )) -gt 900 ] && break
  done
  if ! grep -q "SERVER ready" "$out/server.log"; then
    status "s3_eval_$label" failed "server did not start"; kill $spid 2>/dev/null; wait $spid 2>/dev/null; return
  fi
  local okn=0
  for obj in "$@"; do
    (cd "$ISAAC" && timeout 5400 uv run --extra teleop python "$C/gr00t_eval.py" --object "$obj" --episodes $EVAL_EPS \
        --max_seconds $EVAL_SECS --units "$SRC_DS/units.json" --out "$out" --seed 1000 --port $PORT \
        > "$out/eval_$(tag "$obj").log" 2>&1) && okn=$((okn+1))
  done
  kill $spid 2>/dev/null; wait $spid 2>/dev/null
  [ $okn -eq $# ] && status "s3_eval_$label" ok "$okn/$# objects evaluated in $(took "s3_eval_$label")" \
    || status "s3_eval_$label" failed "$okn/$# objects evaluated"
}
# Base zero-shot is not run: the base N1.7 checkpoint has no SO-101 embodiment head (NEW_EMBODIMENT is
# finetune-only; Gr00tPolicy refuses it), so there is nothing meaningful to roll out before post-training.
status s3_eval_base_zeroshot skipped "GR00T-N1.7-3B has no SO-101 (NEW_EMBODIMENT) head; zero-shot not defined"
if [ $S2_RC -eq 0 ]; then
  run_eval gr00t_mid "$FT/checkpoint-$MID" "${MID_TRAIN_OBJS[@]}"
  run_eval gr00t_final "$FT/checkpoint-$STEPS" "${FINAL_TRAIN_OBJS[@]}" "${FINAL_HELDOUT_OBJS[@]}"
fi

# ---------------- S4: report ----------------
(cd "$LAB" && "$GRPY" "$C/gr00t_morning_report.py" --run "$RUN" > "$RUN/logs/report.log" 2>&1)
rc=$?; [ $rc -eq 0 ] && status s4_report ok "$RUN/MORNING_REPORT.md" || status s4_report failed "rc=$rc"
echo "DONE $RUN" | tee -a "$RUN/logs/orchestrator.log"
