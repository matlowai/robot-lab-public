#!/usr/bin/env bash
# N2c queue: wait for the night-2 imitation run's FINAL checkpoint, choose the GPU, then run scripts/run_rl.sh on it,
# sized to finish (train + paired eval + report) by END_BY.
#
#   systemd-run --user --collect --unit=gr00t-rl-queue -p MemoryMax=30G -E HF_TOKEN_PATH -E PATH -E HOME \
#       bash /mnt/work/AI/robot-lab/rl/scripts/queue_rl.sh
#
# Decision rules (all logged to $QDIR/queue.log; the decision is also written to $QDIR/decision.json):
#  1. Wait for $V2/latest-full/status/s2_finetune.json. If its status is not "ok": do NOT substitute another
#     checkpoint. Log and exit 2; the coordinator decides.
#  2. The checkpoint = <resolved latest-full>/finetune/checkpoint-<N>, with N parsed from the status note
#     ("final checkpoint-N"). It must contain config.json.
#  3. GPU choice (operator rule 2026-10-05, "if ram ooms drop to 1"):
#       - if $DATA/OOM_EVENT exists (any RAM event tonight), or MemAvailable < NEED_GB, or GPU 1 has
#         < NEED_VRAM_MIB free: wait for the unit gr00t-overnight-v2 to be inactive, then use GPU 0 alone;
#       - otherwise use GPU 1 now, alongside the night-2 eval on GPU 0.
#     If a GPU-1 run trips the RAM guard (exit 3) while gr00t-overnight-v2 is still active, wait for it to exit and
#     resume the SAME run (ckpt/latest.pt) on GPU 0 if at least MIN_TRAIN_MIN of training time remain.
#  4. TRAIN_END = END_BY - EVAL_BUDGET_MIN. If fewer than MIN_TRAIN_MIN remain, TRAIN_END = now + MIN_TRAIN_MIN (the
#     run then ends late; logged as such).
# Env: END_BY ("tomorrow 09:00" if it is after 12:00 now, else "today 09:00"), EVAL_BUDGET_MIN=65, MIN_TRAIN_MIN=40,
#      NEED_GB=26 (12 GB floor + measured ~12 GB server + trainer + margin, RUN_PLAN.md), NEED_VRAM_MIB=30000,
#      DRY_RUN=1 (log the decision, do not start), OOM_FLAG (test override),
#      V2=/mnt/weights/ai/robot-lab-data/overnight-gr00t-v2, V2_UNIT=gr00t-overnight-v2, plus run_rl.sh's env.
set -u
RLDIR=/mnt/work/AI/robot-lab/rl
source "$RLDIR/scripts/env.sh"
V2=${V2:-/mnt/weights/ai/robot-lab-data/overnight-gr00t-v2}
V2_UNIT=${V2_UNIT:-gr00t-overnight-v2}
EVAL_BUDGET_MIN=${EVAL_BUDGET_MIN:-65}; MIN_TRAIN_MIN=${MIN_TRAIN_MIN:-40}
NEED_GB=${NEED_GB:-26}; NEED_VRAM_MIB=${NEED_VRAM_MIB:-30000}
OOM_FLAG=${OOM_FLAG:-$DATA/OOM_EVENT}; DRY_RUN=${DRY_RUN:-0}
if [ -z "${END_BY:-}" ]; then
  if [ "$(date +%H)" -ge 12 ]; then END_BY="tomorrow 09:00"; else END_BY="today 09:00"; fi
fi
END_EPOCH=$(date -d "$END_BY" +%s)
QDIR=$DATA/queue
mkdir -p "$QDIR"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$QDIR/queue.log"; }
avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
gpu1_free() { nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '$1==1 {print $2-$3}'; }
v2_active() { systemctl --user is-active --quiet "$V2_UNIT"; }
wait_v2_exit() {
  log "waiting for $V2_UNIT to exit (one GPU job at a time)"
  while v2_active; do sleep 60; done
  log "$V2_UNIT is no longer active ($(systemctl --user is-active "$V2_UNIT" 2>&1))"
}

log "queue armed: waiting for $V2/latest-full/status/s2_finetune.json; END_BY=$(date -d @$END_EPOCH -Is)"
ST=$V2/latest-full/status/s2_finetune.json
seen_v2=0
while [ ! -f "$ST" ]; do
  v2_active && seen_v2=1
  if [ $seen_v2 = 1 ] && ! v2_active && [ ! -f "$ST" ]; then
    sleep 120; [ -f "$ST" ] && break
    log "$V2_UNIT exited without writing $ST -> NOT starting RL. exit 2"
    printf '{"decision":"skipped","reason":"night-2 unit exited without s2_finetune status"}\n' > "$QDIR/decision.json"; exit 2
  fi
  if [ $(( END_EPOCH - $(date +%s) )) -lt $(( (MIN_TRAIN_MIN + EVAL_BUDGET_MIN) * 60 )) ]; then
    log "too late: less than $((MIN_TRAIN_MIN + EVAL_BUDGET_MIN)) min before END_BY and no checkpoint yet -> exit 2"
    printf '{"decision":"skipped","reason":"no night-2 checkpoint before the latest useful start"}\n' > "$QDIR/decision.json"; exit 2
  fi
  sleep 60
done
sleep 5  # let the writer finish
log "found $ST: $(cat "$ST")"
if ! grep -q '"status":"ok"' "$ST"; then
  log "night-2 fine-tune did not finish ok -> NOT starting RL (no checkpoint substitution). exit 2"
  printf '{"decision":"skipped","reason":"s2_finetune not ok","status_file":"%s"}\n' "$ST" > "$QDIR/decision.json"
  exit 2
fi
V2RUN=$(readlink -f "$V2/latest-full")
N=$(grep -o 'checkpoint-[0-9]*' "$ST" | head -1)
CKPT=$V2RUN/finetune/$N
if [ -z "$N" ] || [ ! -f "$CKPT/config.json" ]; then
  log "cannot resolve the final checkpoint (note: $(cat "$ST"); tried $CKPT) -> exit 2"
  printf '{"decision":"skipped","reason":"checkpoint not found","tried":"%s"}\n' "$CKPT" > "$QDIR/decision.json"
  exit 2
fi
log "checkpoint: $CKPT"

GPU=1; why="GPU 1 alongside night-2 (RAM ${NEED_GB} GB + VRAM ${NEED_VRAM_MIB} MiB available)"
A=$(avail_gb); F=$(gpu1_free)
if [ -f "$OOM_FLAG" ]; then
  GPU=0; why="OOM_EVENT present tonight -> one GPU job at a time: $(tail -1 "$OOM_FLAG" | tr -d '"')"
elif [ "$A" -lt "$NEED_GB" ]; then
  GPU=0; why="MemAvailable ${A} GB < ${NEED_GB} GB"
elif [ "${F:-0}" -lt "$NEED_VRAM_MIB" ]; then
  GPU=0; why="GPU 1 free ${F} MiB < ${NEED_VRAM_MIB} MiB"
fi
log "GPU decision: $GPU ($why); MemAvailable=${A} GB, GPU1 free=${F} MiB, $V2_UNIT active=$(v2_active && echo yes || echo no)"
if [ "$DRY_RUN" = 1 ]; then log "DRY_RUN: would start on GPU $GPU with $CKPT"; exit 0; fi
if [ "$GPU" = 0 ] && v2_active; then wait_v2_exit; fi

RUN=$DATA/runs/rl-$(date +%Y%m%d-%H%M)
start_run() {  # start_run <gpu>
  local now te
  now=$(date +%s)
  te=$(( END_EPOCH - EVAL_BUDGET_MIN * 60 ))
  if [ $(( te - now )) -lt $(( MIN_TRAIN_MIN * 60 )) ]; then
    te=$(( now + MIN_TRAIN_MIN * 60 ))
    log "less than ${MIN_TRAIN_MIN} min of training before END_BY - eval; training ${MIN_TRAIN_MIN} min anyway (ends late)"
  fi
  printf '{"decision":"run","gpu":%s,"why":"%s","checkpoint":"%s","run":"%s","train_end":"%s","end_by":"%s","time":"%s"}\n' \
    "$1" "$why" "$CKPT" "$RUN" "$(date -d @$te -Is)" "$(date -d @$END_EPOCH -Is)" "$(date -Is)" > "$QDIR/decision.json"
  log "starting run_rl.sh: RUN=$RUN GPU=$1 TRAIN_END=$(date -d @$te -Is)"
  RUN=$RUN GR00T_CKPT=$CKPT TRAIN_END=$te GPU=$1 SEEDS=sealed bash "$RLDIR/scripts/run_rl.sh" >> "$QDIR/queue.log" 2>&1
}
start_run $GPU
rc=$?
log "run_rl.sh exit $rc"
if [ $rc -eq 3 ] && [ "$GPU" = 1 ]; then  # RAM guard on GPU 1: yield to the imitation run, resume alone on GPU 0
  why="RAM guard tripped on GPU 1 -> resumed on GPU 0 after $V2_UNIT exit"
  wait_v2_exit
  if [ $(( END_EPOCH - $(date +%s) )) -gt $(( (MIN_TRAIN_MIN + EVAL_BUDGET_MIN) * 60 )) ]; then
    start_run 0; rc=$?; log "run_rl.sh (GPU 0 resume) exit $rc"
  else
    log "not enough time left to resume on GPU 0; stopping (the run dir has the partial training log)"
  fi
fi
exit $rc
