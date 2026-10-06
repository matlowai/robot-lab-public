# robot-lab/rl: residual RL on a frozen GR00T N1.7 SO-101 policy

Design and the exact reward: [DESIGN.md](DESIGN.md). Tonight's run and the queue: [RUN_PLAN.md](RUN_PLAN.md).
Data, runs and logs: `/mnt/weights/ai/robot-lab-data/gr00t-rl/` (`bench/`, `smoke/`, `runs/`, `queue/`).

## Layout

| path | venv | what |
|---|---|---|
| `gr00t_rl/gr00t_server.py` | Isaac-GR00T | batched GR00T planning server. Seeded / injected flow-matching noise, pooled image embedding. `--selftest` checks parity, `--bench` measures latency |
| `gr00t_rl/isaac_env.py` | Isaac Lab | `ChunkEnv` (N envs, one object type) and `run_wave` (synchronous wave: settle, then <= 75 chunks of 8 ticks, GR00T + residual) |
| `gr00t_rl/reward.py` | any (torch) | `RewardTracker`: milestones, potential shaping, STRICT bookkeeping identical to `tools/gr00t_eval.py` |
| `gr00t_rl/ppo.py` | any (torch) | asymmetric actor-critic: Gaussian arm residual x Bernoulli release gate, PPO update, GAE |
| `gr00t_rl/seeds.py` | any | training / dev / SEALED eval seed ranges, `mix_seed` |
| `gr00t_rl/units.py` | any (numpy) | dataset units <-> radians (`units.json`) |
| `train.py` | Isaac Lab | one training round (one object, one Isaac process), resumes `ckpt/latest.pt` |
| `evaluate.py` | Isaac Lab | paired base-vs-RL STRICT eval on identical layouts and noise, with videos and strips |
| `report.py` | Isaac Lab | `REPORT.md` and `curves.png` for a run dir |
| `scripts/run_rl.sh` | bash | a whole run: server, training rounds until TRAIN_END, eval of the final checkpoint, report |
| `scripts/queue_rl.sh` | bash | N2c waiter: night-2's final checkpoint -> GPU choice under the RAM rule -> `run_rl.sh` |
| `tests/test_core.py` | Isaac Lab python | CPU tests (reward, PPO, seeds, units). Self-running; no pytest in either venv |

## Commands

```bash
# CPU tests
/mnt/weights/ai/isaac/IsaacLab/.venv/bin/python /mnt/work/AI/robot-lab/rl/tests/test_core.py

# GR00T parity self-test / batch benchmark (GPU 1)
source /mnt/work/AI/robot-lab/rl/scripts/env.sh && cd $GR && CUDA_VISIBLE_DEVICES=1 "${GRENV[@]}" $GRPY \
  $RL/gr00t_rl/gr00t_server.py --policy <checkpoint-N> --selftest      # or --bench 1,8,32,64

# A whole run as a user unit (HF_TOKEN_PATH is passed by NAME: the unit reads the token file itself)
systemd-run --user --collect --unit=gr00t-rl-run -p MemoryMax=30G -E HF_TOKEN_PATH -E PATH -E HOME \
  -E RUN=/mnt/weights/ai/robot-lab-data/gr00t-rl/runs/<name> -E GR00T_CKPT=<checkpoint-N> \
  -E TRAIN_END=$(date -d '+3 hours' +%s) -E GPU=1 bash /mnt/work/AI/robot-lab/rl/scripts/run_rl.sh
```

Rules carried over from tools/overnight_gr00t.sh:
- `run_rl.sh` runs from a frozen copy in `<run>/code`.
- Every stage writes `<run>/status/<stage>.json`.
- The server is killed by its recorded PID only.
- RAM guard: any process that sees MemAvailable < 12 GB exits 3 and appends to
  `/mnt/weights/ai/robot-lab-data/gr00t-rl/OOM_EVENT`. After that the queue runs one GPU job at a time (operator,
  2026-10-05).
