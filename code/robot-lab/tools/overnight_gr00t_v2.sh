#!/usr/bin/env bash
# GR00T night 2 (GPU 0 only). Plan A + release weighting (overnight-gr00t-v2/PLAN_2026-10-05.md, nodes N1a-N1d):
#   S1 re-record demos for the 4 training objects with the fixed recorder (pre-reset labels + ~1 s release/retreat
#   tail, strict keep) -> S1b raw -> GR00T v2.1 (night-1 units, checks) + stats -> S0 replay of the NEW demos
#   through the strict eval -> S2 fine-tune from BASE GR00T-N1.7-3B, night-1 recipe (batch 32) + release/grasp
#   transition weighting (tools/gr00t_finetune_weighted.py), steps sized to the deadline (cap 25000 = night 1)
#   -> S3 strict evals, 24 episodes/object (mid: 4 train; final: 4 train + 2 held-out), 2 parallel lanes
#   -> S4 morning report + night-2 vs night-1 comparison.
#
#   bash tools/overnight_gr00t_v2.sh [tiny|full]
#
# Env overrides: DEADLINE ("tomorrow 07:00" style, full only), DEMO_EPS, DEMO_ENVS, STEPS, STEP_RATE, EVAL_EPS,
# EVAL_SECS, LANES, RELEASE_WEIGHT, BATCH.
# Rules as night 1: chain on exit codes; status JSON per stage; frozen code copies; servers killed by recorded PID.
set -u
if [ -z "${GR00T_FROZEN:-}" ]; then  # bash reads scripts lazily: run from a frozen copy so edits can't hit a live run
  mkdir -p /mnt/weights/ai/nvidia-action/tmp
  F=$(mktemp /mnt/weights/ai/nvidia-action/tmp/overnight_gr00t_v2.XXXXXX.sh) && cp "$0" "$F" && GR00T_FROZEN=1 GR00T_SRC="$0" exec bash "$F" "$@"
fi
MODE="${1:-full}"
T_START=$(date +%s)
STAMP=$(date +%Y%m%d-%H%M)
ROOT=/mnt/weights/ai/robot-lab-data/overnight-gr00t-v2
RUN=$ROOT/${MODE}-${STAMP}
mkdir -p "$RUN"/{status,logs,eval,code,raw}
ln -sfn "$RUN" $ROOT/latest-${MODE}

NA=/mnt/weights/ai/nvidia-action
GR=$NA/Isaac-GR00T
GRPY=$GR/.venv/bin/python
ISAAC=/mnt/weights/ai/isaac/IsaacLab
LAB=/mnt/work/AI/robot-lab
NIGHT1=/mnt/weights/ai/robot-lab-data/overnight-gr00t/full-20261004-2334
UNITS=$NIGHT1/gr00t_ds/units.json                   # night-1 unit mapping, reused verbatim
BASE_MODEL=${BASE_MODEL:-$NA/models/GR00T-N1.7-3B}
FF7=$NA/ffmpeg7/ffmpeg-n7.1.5-12-g1fdbca85aa-linux64-gpl-shared-7.1/lib
PORT=${PORT:-6230}
EXEC_H=${EXEC_H:-8}
RELEASE_WEIGHT=${RELEASE_WEIGHT:-5}

export CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export HF_HOME=$NA/hf HF_HUB_CACHE=$NA/hf/hub HF_XET_CACHE=$NA/hf/xet TMPDIR=$NA/tmp UV_CACHE_DIR=$NA/uv-cache
GRENV=(env LD_LIBRARY_PATH="$FF7" PATH="$GR/.venv/bin:$PATH")

TRAIN_OBJECTS=("mug" "blue block" "soup can" "sugar box")
HELDOUT_OBJECTS=("mustard bottle" "cracker box")
if [ "$MODE" = tiny ]; then
  REC_OBJS=("mug" "sugar box"); DEMO_ENVS=${DEMO_ENVS:-4}; DEMO_EPS=${DEMO_EPS:-6}
  STEPS=${STEPS:-40}; SAVE=${SAVE:-20}; BATCH=${BATCH:-16}; EVAL_EPS=${EVAL_EPS:-2}; EVAL_SECS=${EVAL_SECS:-4}
  REPLAY_EPS=${REPLAY_EPS:-2}; LANES=${LANES:-2}
  MID_OBJS=("mug"); FINAL_OBJS=("mug" "cracker box")
else
  REC_OBJS=("${TRAIN_OBJECTS[@]}"); DEMO_ENVS=${DEMO_ENVS:-16}; DEMO_EPS=${DEMO_EPS:-480}
  STEPS=${STEPS:-}; STEP_CAP=${STEP_CAP:-25000}; BATCH=${BATCH:-32}; EVAL_EPS=${EVAL_EPS:-24}; EVAL_SECS=${EVAL_SECS:-20}
  REPLAY_EPS=${REPLAY_EPS:-8}; LANES=${LANES:-2}
  MID_OBJS=("${TRAIN_OBJECTS[@]}"); FINAL_OBJS=("${TRAIN_OBJECTS[@]}" "${HELDOUT_OBJECTS[@]}")
  # Night-1 measured training rate: 25000 steps in 14166 s = 1.765 steps/s incl. checkpoint saves (batch 32).
  STEP_RATE=${STEP_RATE:-1.65}
  DEADLINE_EPOCH=$(date -d "${DEADLINE:-tomorrow 07:00}" +%s)
  # Eval budget: (96 mid + 144 final) episodes. Measured 2026-10-05 with the night-1 checkpoint: 2 lanes run a 20 s
  # episode in ~43 s wall each (1 lane: ~40 s) -> ~23 s effective per episode on 2 lanes, + ~10 min launches/loads.
  EVAL_EST_S=${EVAL_EST_S:-$(( (96 + 144) * EVAL_EPS / 24 * 23 + 600 ))}
fi

status() {
  printf '{"stage":"%s","status":"%s","note":"%s","time":"%s"}\n' "$1" "$2" "${3//\"/\'}" "$(date -Is)" > "$RUN/status/$1.json"
  echo "[$(date +%H:%M:%S)] STAGE $1 -> $2 $3" | tee -a "$RUN/logs/orchestrator.log"
}
tag() { echo "$1" | tr ' ' '_'; }
stamp() { date +%s > "$RUN/status/$1.t0"; }
took() { echo $(( $(date +%s) - $(cat "$RUN/status/$1.t0") ))s; }

cp "${GR00T_SRC:-$0}" "$LAB"/tools/{record_demos.py,gr00t_raw_to_ds.py,gr00t_so101_config.py,gr00t_finetune_weighted.py,gr00t_policy_server.py,gr00t_eval.py,gr00t_morning_report.py,gr00t_compare_v2.py} "$RUN/code/"
C=$RUN/code
# record_demos.py imports robot_lab.tasks from the repo; the frozen copy only freezes the recorder itself
{
  echo "MODE=$MODE DEMO_ENVS=$DEMO_ENVS DEMO_EPS=$DEMO_EPS REC_LANES=${REC_LANES:-2} STEPS=${STEPS:-auto} BATCH=$BATCH EVAL_EPS=$EVAL_EPS"
  echo "EVAL_SECS=$EVAL_SECS LANES=$LANES EXEC_H=$EXEC_H RELEASE_WEIGHT=$RELEASE_WEIGHT BASE_MODEL=$BASE_MODEL"
  echo "UNITS=$UNITS GR00T_COMMIT=$(git -C "$GR" rev-parse HEAD) DEADLINE=${DEADLINE_EPOCH:+$(date -d @$DEADLINE_EPOCH -Is)}"
  echo "RECORDER_SHA256=$(sha256sum "$C/record_demos.py" | cut -c1-16) EVAL_EST_S=${EVAL_EST_S:-n/a} STEP_RATE=${STEP_RATE:-n/a}"
} | tee "$RUN/config.txt" | tee -a "$RUN/logs/orchestrator.log"
nvidia-smi --query-gpu=index,name,memory.used,power.limit --format=csv > "$RUN/logs/gpu_at_start.csv"

# ---------------- preflight ----------------
PF=$("$GRPY" - <<'PYEOF' 2>&1
from huggingface_hub import get_hf_file_metadata, hf_hub_url
try:
    get_hf_file_metadata(hf_hub_url("nvidia/Cosmos-Reason2-2B", "config.json")); print("backbone access OK")
except Exception as e:  # noqa: BLE001
    print("BLOCKED:", type(e).__name__, str(e).splitlines()[-1][:160])
PYEOF
)
case "$PF" in *"access OK"*) status s_preflight ok "$PF; GPU0 used $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)" ;;
  *) status s_preflight failed "$PF" ;; esac

# ---------------- S1: record demos (fixed recorder) ----------------
stamp s1_record
# The recorder is CPU-bound (one Python process ~1.4 cores, GPU ~1% busy; 16 envs: 0.31 s/step, 32 envs: 0.53 s/step,
# measured 2026-10-05), so objects are recorded in REC_LANES parallel processes on GPU 0 (~10 GB RAM each).
REC_LANES=${REC_LANES:-2}
rlp=()
for (( l=0; l<REC_LANES; l++ )); do
  (
    i=0
    for obj in "${REC_OBJS[@]}"; do
      if [ $(( i % REC_LANES )) -eq $l ]; then
        t=$(tag "$obj")
        (cd "$ISAAC" && timeout 7200 uv run --extra teleop python "$C/record_demos.py" --num_envs $DEMO_ENVS \
            --episodes $DEMO_EPS --object "$obj" --seed 7 --out "$RUN/raw/$t" > "$RUN/logs/demos_$t.log" 2>&1)
        echo $? > "$RUN/logs/demos_$t.rc"
      fi
      i=$((i+1))
    done
  ) &
  rlp+=($!)
done
wait "${rlp[@]}"
RAWS=(); okn=0
for obj in "${REC_OBJS[@]}"; do
  t=$(tag "$obj"); rc=$(cat "$RUN/logs/demos_$t.rc" 2>/dev/null || echo 99)
  n=$(grep -c 'success=True' "$RUN/logs/demos_$t.log" 2>/dev/null); n=${n:-0}
  if [ "$rc" -eq 0 ] && [ "$n" -gt 0 ]; then okn=$((okn+1)); RAWS+=("$RUN/raw/$t"); status "s1_demos_$t" ok "$n strict successes of $DEMO_EPS"
  else status "s1_demos_$t" failed "rc=$rc successes=$n"; fi
done
[ $okn -eq ${#REC_OBJS[@]} ] && status s1_record ok "$okn objects in $(took s1_record) on $REC_LANES lanes" || status s1_record failed "$okn/${#REC_OBJS[@]} objects"

# ---------------- S1b: convert + stats ----------------
stamp s1b_convert
DS=$RUN/gr00t_ds
S1_RC=1
if [ ${#RAWS[@]} -gt 0 ]; then
  (cd "$LAB" && timeout 3600 "${GRENV[@]}" "$GRPY" "$C/gr00t_raw_to_ds.py" --raw "${RAWS[@]}" --out "$DS" \
      --units "$UNITS" --check 8 > "$RUN/logs/convert.log" 2>&1) \
    && (cd "$GR" && timeout 3600 "${GRENV[@]}" "$GRPY" gr00t/data/stats.py --dataset-path "$DS" \
        --embodiment-tag NEW_EMBODIMENT --modality-config-path "$C/gr00t_so101_config.py" > "$RUN/logs/stats.log" 2>&1)
  S1_RC=$?
fi
[ $S1_RC -eq 0 ] && status s1b_convert ok "$(grep -o 'TIMING.*' "$RUN/logs/convert.log"); $(grep -o 'LABELCHECK OK.*units over [0-9]* episodes' "$RUN/logs/convert.log"); $(grep -o 'RELEASE {[^}]*}' "$RUN/logs/convert.log"); $(grep -o 'RAWCHECK.*' "$RUN/logs/convert.log")" \
  || status s1b_convert failed "rc=$S1_RC"

# ---------------- S0: replay the NEW demos through the strict eval path ----------------
stamp s0_replay
okn=0
for obj in "${REC_OBJS[@]}"; do
  t=$(tag "$obj")
  [ -d "$RUN/raw/$t" ] || continue
  (cd "$ISAAC" && timeout 3600 uv run --extra teleop python "$C/gr00t_eval.py" --object "$obj" --episodes $REPLAY_EPS \
      --units "$UNITS" --out "$RUN/eval/replay_expert" --replay "$RUN/raw/$t" > "$RUN/logs/replay_$t.log" 2>&1) && okn=$((okn+1))
done
RS=$(cat "$RUN"/eval/replay_expert/results_*.json 2>/dev/null | grep -o '"strict_successes": [0-9]*' | awk '{s+=$2} END {print s+0}')
[ $okn -eq ${#REC_OBJS[@]} ] && status s0_replay ok "$okn objects, $RS strict of $(( okn * REPLAY_EPS )) replayed; $(took s0_replay)" \
  || status s0_replay failed "$okn/${#REC_OBJS[@]} objects replayed"

# ---------------- S2: size + fine-tune (release-weighted) ----------------
finetune() {  # finetune <output dir> <max steps> <save steps> <log>  -- examples/finetune.sh's exact arguments
  (cd "$GR" && timeout $(( 14 * 3600 )) "${GRENV[@]}" GR00T_RELEASE_WEIGHT=$RELEASE_WEIGHT \
      "$GRPY" "$C/gr00t_finetune_weighted.py" \
      --base_model_path "$BASE_MODEL" --dataset_path "$DS" --embodiment_tag NEW_EMBODIMENT --num_gpus 1 \
      --output_dir "$1" --save_steps "$3" --save_total_limit 5 --max_steps "$2" --warmup_ratio 0.05 \
      --weight_decay 1e-5 --learning_rate 1e-4 --global_batch_size $BATCH --dataloader_num_workers 4 \
      --shard_size 1024 --num_shards_per_epoch 100000 --episode_sampling_rate 0.1 \
      --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
      --modality_config_path "$C/gr00t_so101_config.py" > "$4" 2>&1)
}
weights_live() { grep -E "RELEASE_WEIGHT pid=.* weighted=[1-9]" "$1" | tail -1; }
if [ "$MODE" = tiny ] && [ $S1_RC -eq 0 ]; then  # exercise the calibration path at 10 steps
  stamp s2a_calibrate
  finetune "$RUN/calibrate" 10 10 "$RUN/logs/calibrate.log"; rc=$?
  SPS=$(grep -o "'train_steps_per_second': [0-9.]*" "$RUN/logs/calibrate.log" | tail -1 | awk '{print $2}')
  [ $rc -eq 0 ] && [ -n "$SPS" ] && status s2a_calibrate ok "$SPS steps/s over 10 steps; $(weights_live "$RUN/logs/calibrate.log")" \
    || status s2a_calibrate failed "rc=$rc"
  rm -rf "$RUN/calibrate"
fi
if [ -z "${STEPS:-}" ]; then
  NOW=$(date +%s)
  AVAIL=$(( DEADLINE_EPOCH - NOW - EVAL_EST_S - 900 ))   # 15 min margin for report + checkpoint loads
  STEPS=$("$GRPY" -c "print(max(0, min($STEP_CAP, int($AVAIL * $STEP_RATE) // 500 * 500)))")
  SAVE=$(( STEPS / 4 ))
fi
SAVE=${SAVE:-$(( STEPS / 4 ))}; [ "$SAVE" -lt 1 ] && SAVE=1
MID=$(( STEPS / 2 / SAVE * SAVE )); [ "$MID" -lt "$SAVE" ] && MID=$SAVE
echo "SIZES STEPS=$STEPS SAVE=$SAVE MID=$MID (deadline sizing: ${AVAIL:-n/a} s train budget at $STEP_RATE steps/s, eval est ${EVAL_EST_S:-n/a} s)" \
  | tee -a "$RUN/config.txt" "$RUN/logs/orchestrator.log"
stamp s2_finetune
FT=$RUN/finetune
if [ $S1_RC -eq 0 ] && [ "$STEPS" -gt 0 ]; then
  finetune "$FT" $STEPS $SAVE "$RUN/logs/finetune.log"
  S2_RC=$?
  if [ $S2_RC -eq 0 ] && [ ! -d "$FT/checkpoint-$STEPS" ]; then S2_RC=97; fi
  WL=$(weights_live "$RUN/logs/finetune.log")
  if [ $S2_RC -eq 0 ] && [ -z "$WL" ]; then S2_RC=98; fi       # weighting not proven live -> fail loudly
  [ $S2_RC -eq 0 ] && status s2_finetune ok "final checkpoint-$STEPS; $WL; $(took s2_finetune)" \
    || status s2_finetune failed "rc=$S2_RC (97 = no final checkpoint, 98 = no RELEASE_WEIGHT evidence) $WL"
else
  S2_RC=1; status s2_finetune skipped "no dataset or STEPS=0"
fi

# ---------------- S3: strict evals, LANES parallel lanes (one server per lane) ----------------
run_eval() {  # run_eval <label> <checkpoint dir> <objects...>
  local label=$1 pol=$2; shift 2
  local out=$RUN/eval/$label; mkdir -p "$out"
  stamp "s3_eval_$label"
  if [ ! -d "$pol" ]; then status "s3_eval_$label" skipped "no checkpoint $pol"; return; fi
  local objs=("$@") spids=() lpids=() l
  # free ports picked at runtime (another track may hold fixed ones; 2026-10-05 tiny gate collided on 6130)
  local ports=($("$GRPY" - $PORT $LANES <<'PYEOF'
import socket, sys
base, n, out = int(sys.argv[1]), int(sys.argv[2]), []
p = base
while len(out) < n:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p)); out.append(p)
    except OSError:
        pass
    finally:
        s.close()
    p += 1
print(*out)
PYEOF
))
  echo "EVAL $label ports ${ports[*]}" >> "$RUN/logs/orchestrator.log"
  for (( l=0; l<LANES; l++ )); do
    (cd "$GR" && exec "${GRENV[@]}" "$GRPY" "$C/gr00t_policy_server.py" --policy "$pol" --port ${ports[$l]} \
        --exec_horizon $EXEC_H) > "$out/server_$l.log" 2>&1 &
    spids+=($!)
  done
  local t0=$(date +%s) ready=0
  while [ $(( $(date +%s) - t0 )) -lt 900 ]; do
    ready=0; for (( l=0; l<LANES; l++ )); do grep -q "SERVER ready" "$out/server_$l.log" 2>/dev/null && ready=$((ready+1)); done
    [ $ready -eq $LANES ] && break
    grep -q Traceback "$out"/server_*.log 2>/dev/null && break
    sleep 3
  done
  if [ $ready -ne $LANES ]; then
    status "s3_eval_$label" failed "$ready/$LANES servers started"; kill "${spids[@]}" 2>/dev/null; wait "${spids[@]}" 2>/dev/null; return
  fi
  for (( l=0; l<LANES; l++ )); do
    (
      okl=0; i=0
      for obj in "${objs[@]}"; do
        if [ $(( i % LANES )) -eq $l ]; then
          (cd "$ISAAC" && timeout 7200 uv run --extra teleop python "$C/gr00t_eval.py" --object "$obj" --episodes $EVAL_EPS \
              --max_seconds $EVAL_SECS --units "$UNITS" --out "$out" --seed 1000 --port ${ports[$l]} \
              > "$out/eval_$(tag "$obj").log" 2>&1) && okl=$((okl+1))
        fi
        i=$((i+1))
      done
      echo $okl > "$out/lane_$l.ok"
    ) &
    lpids+=($!)
  done
  wait "${lpids[@]}"
  kill "${spids[@]}" 2>/dev/null; wait "${spids[@]}" 2>/dev/null
  local okn=0; for (( l=0; l<LANES; l++ )); do okn=$(( okn + $(cat "$out/lane_$l.ok" 2>/dev/null || echo 0) )); done
  [ $okn -eq ${#objs[@]} ] && status "s3_eval_$label" ok "$okn/${#objs[@]} objects x $EVAL_EPS eps in $(took "s3_eval_$label") on $LANES lanes" \
    || status "s3_eval_$label" failed "$okn/${#objs[@]} objects evaluated"
}
status s3_eval_base_zeroshot skipped "GR00T-N1.7-3B has no SO-101 (NEW_EMBODIMENT) head; zero-shot not defined"
if [ $S2_RC -eq 0 ]; then
  run_eval gr00t_mid "$FT/checkpoint-$MID" "${MID_OBJS[@]}"
  run_eval gr00t_final "$FT/checkpoint-$STEPS" "${FINAL_OBJS[@]}"
fi

# ---------------- S4: report + night-1 comparison ----------------
(cd "$LAB" && "$GRPY" "$C/gr00t_morning_report.py" --run "$RUN" > "$RUN/logs/report.log" 2>&1 \
  && "$GRPY" "$C/gr00t_compare_v2.py" --run "$RUN" --night1 "$NIGHT1" >> "$RUN/logs/report.log" 2>&1)
rc=$?; [ $rc -eq 0 ] && status s4_report ok "$RUN/MORNING_REPORT.md; total $(( $(date +%s) - T_START ))s" || status s4_report failed "rc=$rc"
echo "DONE $RUN" | tee -a "$RUN/logs/orchestrator.log"
