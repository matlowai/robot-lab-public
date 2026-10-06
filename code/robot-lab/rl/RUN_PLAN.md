# RUN_PLAN: tonight's residual-RL run (N2c), 2026-10-05 / 06

## 1. What is queued

User unit `gr00t-rl-queue` runs `scripts/queue_rl.sh`. Its log is `/mnt/weights/ai/robot-lab-data/gr00t-rl/queue/queue.log`,
and its decision is written to `queue/decision.json`.

1. **Waits for** `/mnt/weights/ai/robot-lab-data/overnight-gr00t-v2/latest-full/status/s2_finetune.json`.
   - status `ok`: the checkpoint is `<latest-full>/finetune/checkpoint-<N>` (N parsed from the note) and must contain
     config.json.
   - any other status, the night-2 unit exiting without the file, or no checkpoint by ~07:15: **no RL run** (exit 2).
     No other checkpoint is substituted; the coordinator decides.
2. **GPU choice**, applying the operator's rule (2026-10-05): "Thanks if ram ooms drop to 1".
   - **GPU 1 now**, alongside night 2's evals on GPU 0, if all of these hold: no `gr00t-rl/OOM_EVENT` file, MemAvailable
     >= 26 GB, and GPU 1 free >= 30,000 MiB. The 26 GB is the 12 GB floor plus the measured ~12 GB server and trainer
     RSS plus a margin.
   - **otherwise GPU 0, after `gr00t-overnight-v2` exits.**
   - Any RL process that sees MemAvailable < 12 GB exits 3 and appends to `OOM_EVENT`. If that happens on GPU 1, the
     waiter waits for `gr00t-overnight-v2` to exit, then resumes the same run (`ckpt/latest.pt`) on GPU 0, if
     >= 105 min remain before 09:00.
3. **Run**: `scripts/run_rl.sh`, from a frozen copy in `<run>/code`, in `/mnt/weights/ai/robot-lab-data/gr00t-rl/runs/rl-<stamp>/`.
   - GR00T server on port 6190.
   - Training rounds of 12 waves x 64 envs (~23 min each), cycling mug -> blue block -> soup can -> sugar box, until
     TRAIN_END = 09:00 - 65 min = **07:55**.
   - Then the pre-registered paired eval of `ckpt/final.pt`: 6 objects x 2 waves x 64 envs x 2 arms, sealed seeds,
     30-tick rest check (~50 min).
   - Then `REPORT.md`. **Expected end: ~08:45-09:00.**
4. **Expected start**: when night 2's fine-tune reports ok. The coordinator's estimate is ~03:15, which gives ~4 h 40 min
   of training (about 150 waves, about 9,600 episodes at N = 64).
   - If the start waits for GPU 0 (RAM rule), it begins after night 2's evals end (~05:00): about 2 h 55 min of
     training.
   - If fewer than 40 min of training remain, it still trains 40 min and ends late. The log says so.

Pre-registration: DESIGN.md section 5. Headline = STRICT successes / valid layouts per object and arm, with the paired
split, and the rest-check column beside it. The final checkpoint only, no selection. Sealed seeds 900,000,000+ were
never used in training. Held-out objects are never trained on.

## 2. Smoke-test evidence (night-1 checkpoint-25000, GPU 1)

| check | result | evidence |
|---|---|---|
| CPU tests | 9/9 pass | `gr00t-rl/logs/test_core_3.log` |
| GR00T noise-injection parity vs GR00T's own `get_action` | max diff 0.0. Seeded noise is batch-order invariant; changing the seed moves the chunk by up to 17.8 units | `gr00t-rl/bench/selftest.log` |
| GR00T batched plan, ms/env | 84 (B=1), 16.5 (B=8), 12.1 (B=32), 12.3 (B=64); VRAM 6.4 -> 10.6 GB | `gr00t-rl/bench/gr00t_bench.log` |
| Training throughput | N=32: ~70 s/wave, 258-292 env ticks/s. N=64: 108-114 s/wave, 337-354 env ticks/s, 32-34 RL steps/s | `smoke/run-20261005-2136/train_log.jsonl`, `smoke/train_n64/train_log.jsonl` |
| RAM / VRAM | trainer RSS 7.8-10.0 GB, eval process 9.1-9.5 GB, server ~2.3 GB. Cgroup peaks including page cache: 20.8-26 GB (cap 30G). GPU 1 in use: 18-19 GB (N=32), 22.9 GB (N=64) | same, plus `journalctl --user -u gr00t-rl-smoke*` |
| 25-min training run | 18 waves, 576 episodes, mug + blue block. Stochastic per-wave strict 13-23 of ~30 | `smoke/run-20261005-2136/REPORT.md`, `curves.png` |
| Paired eval, dev seeds 50,500,000, final smoke checkpoint | blue block: RL 15/31 vs base 5/32 (paired 4 both / 1 base-only / 11 RL-only). Mustard bottle (HELD-OUT): RL 9/31 vs base 0/31. With the rest check: blue block RL 14/31 (14 rested) vs base 2/32 (2 rested); a second run, mustard 3/15 vs 0/15 at N=16 | `smoke/run-20261005-2136/eval*/results_*.json`, `smoke/run-evalpath-2223/` |
| Rollouts looked at | Base: mug carried into the bowl, then held and carried away (night 1's failure). RL blue block env 9: grasp, lift, carry, one release-gate chunk at ~4 s, block drops into the bowl and rests for 1 s, bowl upright | `smoke/eval_base_n8/mug_base_w0_env0_fail_strip.png`, `smoke/run-20261005-2136/eval_rest/blue_block_rl_w0_env9_strict.mp4` |
| Queue logic, dry runs | ok checkpoint -> GPU 1. OOM flag -> GPU 0 after the v2 unit. s2 failed -> exit 2, no substitution | `gr00t-rl/queue/queue-dryrun-tests-20261005.log` |

The smoke numbers come from DEV seeds, one wave, and a 25-minute policy on the NIGHT-1 checkpoint. They show the
loop works and the effect is large. They are not the result. The result is the sealed paired eval on the night-2
checkpoint.

## 3. What to watch overnight

- `queue/queue.log` and `queue/decision.json`: which GPU was chosen, and why.
- `<run>/logs/orchestrator.log` (one line per round); `<run>/train_log.jsonl` (per wave: strict, p_release_over_bowl,
  rss, avail, wave_s).
- `gr00t-rl/OOM_EVENT`: if present, a RAM event happened and the queue is in one-job mode.
- If night 2 itself fails, no RL run starts. That is deliberate.
- Stop the RL run with `systemctl --user stop gr00t-rl-queue`. It kills its own children only. The RL server PID is
  in `orchestrator.log`.
