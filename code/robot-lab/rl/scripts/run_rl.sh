#!/usr/bin/env bash
# One residual-RL run, end to end: batched GR00T server -> training rounds (one object per Isaac process, cycling the
# four training objects, resuming ckpt/latest.pt) until TRAIN_END -> paired STRICT eval of the FINAL checkpoint on
# every object (sealed seeds, base vs RL) -> REPORT.md.  Mirrors tools/overnight_gr00t.sh rules: frozen code copy,
# status/<stage>.json per stage, chain on exit codes not log text, the server is killed by its recorded PID only.
#
#   GR00T_CKPT=<checkpoint-N> RUN=<dir> TRAIN_END=<unix s> GPU=1 bash scripts/run_rl.sh
# Env (defaults): NUM_ENVS=64 WAVES_PER_ROUND=12 EVAL_WAVES=2 EVAL_ENVS=64 SEEDS=sealed PORT=6190
#   OBJECTS="mug|blue block|soup can|sugar box"   EVAL_OBJECTS="<4 training>|mustard bottle|cracker box"
#   MAX_ROUNDS=999  ACTOR_PRIV=0
# Exit 3 = RAM guard tripped (MemAvailable < 12 GB): training stops, no eval is started, OOM_EVENT is flagged
# (operator rule 2026-10-05: on RAM trouble drop to one GPU job; the imitation run has priority).
set -u
if [ -z "${RL_FROZEN:-}" ]; then  # bash reads scripts lazily: run from a frozen copy so edits can't hit a live run
  : "${RUN:?set RUN}"
  mkdir -p "$RUN/code"
  cp -r /mnt/work/AI/robot-lab/rl/{gr00t_rl,train.py,evaluate.py,report.py,scripts,DESIGN.md} "$RUN/code/" 2>/dev/null
  rm -rf "$RUN/code/gr00t_rl/__pycache__"
  RL_FROZEN=1 exec bash "$RUN/code/scripts/run_rl.sh" "$@"
fi
C=$RUN/code
source "$C/scripts/env.sh"
: "${GR00T_CKPT:?set GR00T_CKPT}" "${TRAIN_END:?set TRAIN_END (unix seconds)}"
GPU=${GPU:-1}; NUM_ENVS=${NUM_ENVS:-64}; WAVES_PER_ROUND=${WAVES_PER_ROUND:-12}; EVAL_WAVES=${EVAL_WAVES:-2}
EVAL_ENVS=${EVAL_ENVS:-64}; SEEDS=${SEEDS:-sealed}; PORT=${PORT:-6190}; MAX_ROUNDS=${MAX_ROUNDS:-999}
ACTOR_PRIV=${ACTOR_PRIV:-0}
IFS='|' read -r -a OBJ <<< "${OBJECTS:-mug|blue block|soup can|sugar box}"
IFS='|' read -r -a EOBJ <<< "${EVAL_OBJECTS:-mug|blue block|soup can|sugar box|mustard bottle|cracker box}"
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$RUN"/{status,logs,eval,ckpt}

status() {  # status <stage> <ok|failed|skipped> <note>
  printf '{"stage":"%s","status":"%s","note":"%s","time":"%s"}\n' "$1" "$2" "${3//\"/\'}" "$(date -Is)" > "$RUN/status/$1.json"
  echo "[$(date +%H:%M:%S)] STAGE $1 -> $2 $3" | tee -a "$RUN/logs/orchestrator.log"
}
avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
reap_telemetry() {  # Isaac Kit can leave omni.telemetry.transmitter behind (~0.5 GB, seen 2026-10-05). Reap only ones that
  # inherited THIS run's environment (RUN=<this run dir>): never another job's. (In a user unit /proc/self/cgroup reads
  # "0::/", so a cgroup-based filter would have matched the whole machine -- checked 2026-10-05, not used.)
  local p
  for p in $(pgrep -f omni.telemetry.transmitter); do
    [ "$p" = "$$" ] && continue
    case "$(readlink "/proc/$p/exe" 2>/dev/null)" in */omni.telemetry.transmitter) ;; *) continue ;; esac
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qxF "RUN=$RUN" && kill "$p" 2>/dev/null \
      && echo "reaped telemetry pid $p" >> "$RUN/logs/orchestrator.log"
  done
}

printf 'GR00T_CKPT=%s GPU=%s NUM_ENVS=%s WAVES_PER_ROUND=%s TRAIN_END=%s (%s) EVAL_WAVES=%s EVAL_ENVS=%s SEEDS=%s OBJECTS=%s EVAL_OBJECTS=%s ACTOR_PRIV=%s START=%s\n' \
  "$GR00T_CKPT" "$GPU" "$NUM_ENVS" "$WAVES_PER_ROUND" "$TRAIN_END" "$(date -d @"${TRAIN_END%.*}" -Is)" "$EVAL_WAVES" \
  "$EVAL_ENVS" "$SEEDS" "${OBJ[*]}" "${EOBJ[*]}" "$ACTOR_PRIV" "$(date -Is)" | tee -a "$RUN/config.txt"
nvidia-smi --query-gpu=index,name,memory.used --format=csv > "$RUN/logs/gpu_at_start.csv"
echo "MemAvailable at start: $(avail_gb) GB" | tee -a "$RUN/logs/orchestrator.log"

# ---- GR00T server (killed by PID); refuse a port someone else already listens on ----
if ss -ltn "sport = :$PORT" | grep -q ":$PORT"; then
  status s0_server failed "port $PORT already in use: $(ss -ltnp "sport = :$PORT" | tail -1)"; exit 1
fi
(cd "$GR" && exec "${GRENV[@]}" "$GRPY" "$C/gr00t_rl/gr00t_server.py" --policy "$GR00T_CKPT" --port "$PORT") \
  > "$RUN/logs/server.log" 2>&1 &
SPID=$!
t0=$(date +%s)
until grep -qE "SERVER ready|Traceback" "$RUN/logs/server.log" 2>/dev/null || ! kill -0 $SPID 2>/dev/null; do
  sleep 3; [ $(( $(date +%s) - t0 )) -gt 900 ] && break
done
if ! grep -q "SERVER ready" "$RUN/logs/server.log"; then
  status s0_server failed "GR00T server did not start (see logs/server.log)"; kill $SPID 2>/dev/null; exit 1
fi
status s0_server ok "pid $SPID port $PORT $(grep -m1 LOADED "$RUN/logs/server.log")"
cleanup() { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; }
trap cleanup EXIT

# ---- training rounds ----
PRIV=(); [ "$ACTOR_PRIV" = 1 ] && PRIV=(--actor_priv)
r=$(ls "$RUN"/logs/train_round*.log 2>/dev/null | wc -l); r0=$r; fails=0; ram_event=0  # a resumed run continues the round count (fresh seeds)
while [ "$(date +%s)" -lt "${TRAIN_END%.*}" ] && [ $(( r - r0 )) -lt "$MAX_ROUNDS" ]; do
  obj=${OBJ[$(( r % ${#OBJ[@]} ))]}
  (cd "$ISAAC" && timeout 7200 uv run --extra teleop python "$C/train.py" --object "$obj" --num_envs "$NUM_ENVS" \
      --waves "$WAVES_PER_ROUND" --round $r --run_dir "$RUN" --ckpt_in "$RUN/ckpt/latest.pt" --port "$PORT" \
      --deadline "$TRAIN_END" --video_every 10 "${PRIV[@]}" > "$RUN/logs/train_round$(printf %03d $r).log" 2>&1)
  rc=$?
  reap_telemetry
  echo "[$(date +%H:%M:%S)] round $r ($obj) rc=$rc avail=$(avail_gb)GB" | tee -a "$RUN/logs/orchestrator.log"
  if [ $rc -eq 137 ]; then  # SIGKILL: on this box that is the OOM killer / MemoryMax -> same rule as the RAM guard
    printf '{"time":"%s","reason":"round %s (%s) killed by SIGKILL (rc 137): treated as OOM"}\n' "$(date -Is)" "$r" "$obj" \
      >> "$DATA/OOM_EVENT"
    rc=3
  fi
  if [ $rc -eq 3 ]; then ram_event=1; break; fi
  if [ $rc -ne 0 ]; then fails=$((fails+1)); [ $fails -ge 2 ] && break; else fails=0; fi
  r=$((r+1))
done
NW=$(wc -l < "$RUN/train_log.jsonl" 2>/dev/null || echo 0)
if [ $ram_event -eq 1 ]; then
  status s1_train failed "RAM guard tripped in round $r (MemAvailable < 12 GB); OOM_EVENT flagged; eval NOT started"
  "$ISAACPY" "$C/report.py" "$RUN" >/dev/null 2>&1
  exit 3
fi
[ -f "$RUN/ckpt/latest.pt" ] && status s1_train ok "$r rounds, $NW waves" || { status s1_train failed "no checkpoint (rounds $r, fails $fails)"; exit 1; }

# ---- paired eval of the FINAL checkpoint (no selection) ----
cp "$RUN/ckpt/latest.pt" "$RUN/ckpt/final.pt"
okn=0
for obj in "${EOBJ[@]}"; do
  (cd "$ISAAC" && timeout 3600 uv run --extra teleop python "$C/evaluate.py" --object "$obj" --num_envs "$EVAL_ENVS" \
      --waves "$EVAL_WAVES" --ckpt "$RUN/ckpt/final.pt" --out "$RUN/eval" --seeds "$SEEDS" --port "$PORT" \
      > "$RUN/logs/eval_$(echo "$obj" | tr ' ' '_').log" 2>&1)
  rc=$?
  reap_telemetry
  [ $rc -eq 0 ] && okn=$((okn+1))
  [ $rc -eq 3 ] && { status s2_eval failed "RAM guard tripped during $obj"; ram_event=1; break; }
done
if [ $ram_event -eq 0 ]; then
  [ $okn -eq ${#EOBJ[@]} ] && status s2_eval ok "$okn/${#EOBJ[@]} objects evaluated" \
    || status s2_eval failed "$okn/${#EOBJ[@]} objects evaluated (see logs/eval_*.log)"
fi
"$ISAACPY" "$C/report.py" "$RUN" > "$RUN/logs/report.log" 2>&1 && status s3_report ok "$RUN/REPORT.md" || status s3_report failed "see logs/report.log"
[ $ram_event -eq 1 ] && exit 3
exit 0
