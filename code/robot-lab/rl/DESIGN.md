# Residual RL on a frozen GR00T N1.7 SO-101 policy: design (2026-10-05)

Goal (operator, 2026-10-05): the policy should reliably RELEASE the object into the bowl and complete STRICT successes
more often. Night 1 (`overnight-gr00t/full-20261004-2334`): 8/72 strict, and in 48 of 59 lifted-but-not-placed episodes
the policy never commands the jaw open after the lift. The demos hold ~1 release frame per episode.

Scope: this is RL on top of the imitation policy. GR00T stays frozen. Task, objects, strict criteria and the GR00T
model do not change. A PPO fine-tune of the full 3B model is out of scope.

## 1. Method: residual RL with a release gate (primary); noise steering (fallback, hook built)

| | Residual + release gate (chosen) | Noise steering (DSRL) |
|---|---|---|
| What the small actor controls | an additive arm correction (5 joints, at most 4 deg) and a Bernoulli "release now" gate per executed chunk | the initial noise of GR00T's flow-matching head |
| Can it produce a release GR00T never makes? | **yes**: the gate commands the jaw open (50 units, 45 deg) whatever GR00T plans | only if GR00T's output distribution contains opening in that state. With ~1 release frame per demo, it may not |
| Action dimension | 6 | 40 x 132 padded noise (5,280); 16 x 6 = 96 if only the used block is steered |
| Starts exactly at the base policy | yes: zero-initialised output layer, release bias -3, so the deterministic policy is GR00T | yes (zero noise mean) |
| Feasible with GR00T's code? | yes, it acts on the decoded joint targets | yes. `gr00t_server.py` re-implements the 4-step Euler loop of `Gr00tN1d7ActionHead.get_action_with_features` (`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:349-428`) with the noise passed in. The `--selftest` parity is **exact** (max diff 0.0 on identical noise) |

Measured, `bench/selftest.log`: changing only the noise seed moves the planned chunk by up to 17.8 dataset units. So
noise steering has some authority. But the failure we must fix is a missing behaviour, not a badly sampled one.
The residual's gate reaches it directly. Noise steering stays the fallback: the server already accepts a `noise`
override for the leading (h, d) block, so a DSRL actor needs only a new actor/action mapping. The seeded noise is used
now for common random numbers in the paired eval.

Release primitive: gate = 1 commands the jaw at least 50 units for the current chunk and the next one
(`RELEASE_HOLD_CHUNKS = 2`, 16 ticks = 0.53 s). The jaw velocity limit is 2 rad/s, so opening past the 0.5 rad
success threshold takes ~8 ticks. A single-chunk opening could be re-closed by GR00T before the success term reads it.

Asymmetric actor-critic. The ACTOR sees only deployable inputs (72-d + GR00T embedding):
- joint positions (units / 100) and velocities
- GR00T's executed plan (8 x [arm relative to the current joints, jaw / 100])
- the object one-hot (= the language instruction)
- the episode time fraction
- the previous residual and gate, and the hold flag
- GR00T's pooled image embedding: the mean over image tokens of the action head's processed backbone features
  (2048-d), passed through a learned 2048->64 projection

The CRITIC also sees privileged simulator state (17-d): object minus TCP, object minus bowl, rise, xy distance, TCP
distance, jaw angle, bowl tilt and shift, and the latched milestone flags. `--actor_priv` (ABLATION, off by default)
gives the actor the privileged state too. Its results would not be comparable with vision-only GR00T and must be
labelled as such.

## 2. Reward (exact; `gr00t_rl/reward.py`, unit tests `tests/test_core.py`)

Per control tick, summed over the 8 ticks of an RL step. Readings are never taken from a tick on which the env
terminated (Isaac auto-resets inside `env.step`). The last valid reading decides, as in
`tools/gr00t_eval.py:231-243`.

| term | value | when |
|---|---|---|
| reach | +0.25 once | TCP within 2 cm of the object centre |
| lift | +1.0 once | object centre >= 3 cm above its settled start height (the strict lift criterion) |
| over_bowl | +1.0 once | after lift: object centre inside the bowl radius (5.5 cm, xy) and above the bowl root |
| release | +2.0 once | after lift: jaw > 0.5 rad while the object centre is inside the bowl radius (xy) |
| strict | +10.0 terminal | the env's success term fires AND lifted AND bowl ok on the last valid reading AND no blow-up (= gr00t_eval STRICT) |
| blowup | -2.0, ends the episode | joint speed > 30 rad/s, object or bowl z < -3 cm, or object speed > 3 m/s |
| dropped | -1.0 terminal | the env's object_dropping termination |
| bowl_bad | -2.0 once | first tick with bowl tilt > 20 deg or shift > 3 cm |
| shaping | gamma_t * Phi(s') - Phi(s) | Phi = 2.0 * (0.3 - min(d_xy(object, bowl), 0.3)) while "carried", else 0. carried = lifted (latched) AND (held [rise >= 2 cm and TCP-object <= 5 cm] OR object inside the bowl radius). gamma_t = 0.99^(1/8) |

Hacking guards, each tested:
- **Hovering pays nothing.** Milestones are latched and paid once. Phi >= 0, so hovering while carrying costs
  (1 - gamma) * Phi per tick. A constant negative potential would have paid a survival bonus; the first version did,
  the test caught it, and it was fixed.
- **Pushing without lifting pays exactly 0.** Lift is required for over_bowl, release, the carried branch and strict.
- **Opening early pays nothing.** The release milestone needs the object over the bowl. Dropping outside the bowl
  returns the potential.
- **Lift / drop / re-lift cycles net <= 0** (telescoping potential, `test_hover_and_regrasp_cannot_farm`: -0.039).
- **The bowl criterion uses the LAST valid reading**, as the eval does (`test_strict_uses_last_valid_bowl_reading`).

The reward is never the headline. The headline is the STRICT success count on the sealed paired eval (section 5).

## 3. Throughput (measured on GPU 1, RTX PRO 6000 Max-Q)

- **GR00T batched plan** (`bench/gr00t_bench.log`, random 256 x 256 images, median of 5):

  | batch | ms per plan | ms per env | peak VRAM |
  |---|---|---|---|
  | 1 | 84 | 84 | 6.4 GB |
  | 8 | 132 | 16.5 | |
  | 32 | 388 | 12.1 | 8.5 GB |
  | 64 | 788 | 12.3 | 10.6 GB |

  About 3 ms per env of this is CPU preprocessing.
- **Isaac sim with two 256 x 256 cameras per env** (training waves, `smoke/*/train_log.jsonl`):

  | N | wave (600 ticks) | sim | GR00T plan | env ticks/s | RL steps/s | trainer RSS |
  |---|---|---|---|---|---|---|
  | 16 | 57-61 s | 44-47 s | 11-13 s | 157-168 | 11-14 | 9.4-10.0 GB |
  | 32 | 65-74 s | 50-53 s | 14-21 s | 258-292 | 14-21 | 7.8-9.3 GB |
  | 64 | 108-114 s | 60-61 s | 46-51 s | 337-354 | 32-34 | 8.5-8.6 GB |

  Sim is render-overhead bound, so N = 64 gives 1.34x the episodes/hour of N = 32 (0.58 vs 0.43 episodes/s).
  **The queued run uses N = 64.** GPU 1 total in use during N = 64 training: 22.9 GB (server + trainer + desktop).
  The GR00T server process takes ~2.3 GB RSS.
- Why a separate process: Isaac Lab's venv has torch 2.11; GR00T's has torch 2.9 with a flash-attn build for it. One
  socket round trip per RL step carries all envs (unlike the one-env `tools/gr00t_policy_server.py`).
- Port 6190 and authkey `robot-lab-rl`. The imitation pipelines use 6120-6139 and now 6230+. On 2026-10-05 our first
  smoke server sat on 6130, where night 2's tiny gate expected its own server. A foreign client's failed handshake
  then killed our server. Fix: we moved to 6190, `run_rl.sh` refuses a busy port, and the server now rejects bad
  handshakes and keeps listening (tested).

## 4. Algorithm and budget

- PPO (`gr00t_rl/ppo.py`, patrol-lab style):
  - Gaussian arm (pre-tanh u, env gets 4 deg * tanh(u); initial std 0.3) times Bernoulli gate
  - gamma 0.99 per RL step, lambda 0.95, clip 0.2, lr 3e-4, 5 epochs x 4 minibatches, entropy 0.003
  - target KL 0.02 (early stop at 1.5x), grad-norm clip 1.0
  - observation normalisers updated AFTER each PPO update, for the first 8 global waves only, then frozen
- RL step = one executed chunk (exec horizon 8 = 0.27 s); an episode is at most 75 steps (20 s, the eval budget).
- Synchronous waves: all N envs reset with one seed, so every episode is complete within a wave. The time fraction
  is in the observation, so the 20 s end is treated as terminal.
- One object type per Isaac process (Isaac Lab 3.0 EA heterogeneous spawning made too few instances; course NOTES).
  A round = one Isaac process, one training object, 12 waves, resuming `ckpt/latest.pt`. Rounds cycle mug ->
  blue block -> soup can -> sugar box.
- Budget: training until TRAIN_END, then the eval (section 5). The queued run is sized to end by ~09:00 local (see
  RUN_PLAN.md).

## 5. Evaluation protocol (pre-registered; fixed before the queued run starts)

1. **Checkpoint**: the FINAL checkpoint of the run (`ckpt/final.pt`). No selection among checkpoints (patrol-lab
   D49 correction: selecting on small validation sets selects noise).
2. **Seeds**: the SEALED range 900,000,000 + 10,000 * object_index + wave (`gr00t_rl/seeds.py`). Training seeds are
   10,000,000 + 1,000 * round + wave. `train_seed` asserts it never lands in the sealed range, and the tests check the
   two ranges are disjoint. Development and smoke runs use 50,000,000+.
3. **Paired arms**: per wave, BASE (GR00T alone) and RL (GR00T + deterministic residual: arm = 4 deg * tanh(mean),
   release = p > 0.5) run on the same wave seed, so the N layouts are identical. GR00T's initial noise is seeded per
   (eval key, wave seed, env, chunk) and is identical across arms (common random numbers).
4. **Objects**: the 4 training objects, plus the 2 HELD-OUT objects (mustard bottle, cracker box). The held-out
   objects were never used in RL training: `train.py` refuses them. The residual has no one-hot for them (zeros).
5. **Episodes**: EVAL_WAVES x EVAL_ENVS per object per arm (default 2 x 32 = 64). Layouts with the object < 10 cm
   (xy) from the bowl, or the bowl tilted > 20 deg, after settling are excluded and counted (`invalid_layouts`), as
   in gr00t_eval's re-seeding.
6. **Headline**: STRICT successes / valid episodes per arm and object, with Wilson 95% CIs, the paired split (both /
   base-only / RL-only / neither), and train and held-out totals. Everything else (task term, released-over,
   reward) is diagnostic.
7. **Look before quoting**: videos and 8-frame strips of envs 0-3 of wave 0, for both arms, for every object.
8. **Rest check** (secondary column, required next to the headline): in the eval only, the success term does not
   auto-reset. After it fires, the arm is held still with the jaw open for 30 ticks (1 s). `strict_rested` = strict
   AND, at the end of that second, the object centre is inside the bowl footprint and < 7 cm above the bowl root,
   and the bowl is upright (<= 20 deg) and within 3 cm of its start. This answers "was the release real?" without
   changing the strict definition. A layout counts only if it is valid in BOTH arms: GPU physics is not bitwise
   reproducible, so the post-settle check can differ by an env between arms.

Harness deviation from `tools/gr00t_eval.py`, stated rather than hidden:
- the settle is one full chunk (2 pre-shape ticks plus 6 hold ticks), not 2 ticks;
- z0 is read after it;
- GR00T's noise is seeded rather than drawn from the global RNG;
- episodes run synchronously in batches.

The base arm is therefore "GR00T in this harness". The night-1 numbers from gr00t_eval are a separate reference,
not a paired comparison.

## 6. Smoke-test findings (2026-10-05, GPU 1, night-1 checkpoint-25000)

Full numbers are in RUN_PLAN.md section 2. In short:
- The loop works end to end: server, waves, PPO, checkpoints and resume, paired eval, report. 9/9 CPU tests pass.
  Server noise-injection parity is exact.
- **The base harness reproduces night 1's failure on video**: the mug is carried over and into the bowl, then held
  and carried away (`smoke/eval_base_n8/mug_base_w0_env0_fail_strip.png`).
- After 18 training waves (576 episodes, ~25 min), the deterministic residual already releases over the bowl.
  Dev seeds, 1 wave each:
  - blue block: 15/31 vs base 5/32
  - mustard bottle (held-out, never trained): 9/31 vs 0/31
  - with the rest check, blue block 14/31 vs 2/32; every strict success also rested
  - each RL success used exactly one release-gate chunk
  - videos inspected: the block drops into the bowl and stays there
- The stochastic exploration policy (5% release per chunk) already gets 13-23/30 strict per training wave. Release
  is the bottleneck, as the failure analysis said.
