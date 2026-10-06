"""Seed streams. Training and the sealed evaluation use disjoint ranges; training code asserts it never touches the
eval range (rl/DESIGN.md section 5). Each wave seed drives env.reset(seed) for all N envs, so a layout is a function of
(wave seed, env index); GR00T's initial noise is seeded per (noise key, wave seed, env, chunk) via mix_seed.

  training waves : 10_000_000 + 1_000 * round + wave        (round < 9_000, wave < 1_000)
  smoke/eval-dev : 50_000_000 + ...                          (anything we look at while developing)
  SEALED eval    : 900_000_000 + 10_000 * object_index + wave (object_index over TRAIN + HELDOUT, fixed order)
Noise keys: training 0x7A1, sealed eval 0xE7A1 (same key for every arm: common random numbers).
"""

from __future__ import annotations

TRAIN_BASE, DEV_BASE, EVAL_BASE = 10_000_000, 50_000_000, 900_000_000
TRAIN_NOISE_KEY, EVAL_NOISE_KEY = 0x7A1, 0xE7A1
EVAL_OBJECT_ORDER = ["mug", "blue block", "soup can", "sugar box", "mustard bottle", "cracker box"]


def train_seed(round_idx: int, wave: int) -> int:
    assert 0 <= round_idx < 9_000 and 0 <= wave < 1_000, (round_idx, wave)
    s = TRAIN_BASE + 1_000 * round_idx + wave
    assert not is_eval_seed(s)
    return s


def dev_seed(i: int) -> int:
    assert 0 <= i < 100_000_000
    return DEV_BASE + i


def eval_seed(object_name: str, wave: int) -> int:
    assert 0 <= wave < 10_000
    return EVAL_BASE + 10_000 * EVAL_OBJECT_ORDER.index(object_name) + wave


def is_eval_seed(s: int) -> bool:
    return EVAL_BASE <= s < EVAL_BASE + 10_000 * len(EVAL_OBJECT_ORDER)


def mix_seed(*xs: int) -> int:
    """Deterministic 63-bit hash of integers (splitmix64), for per-(wave, env, chunk) GR00T noise seeds."""
    h = 0x9E3779B97F4A7C15
    for x in xs:
        h = (h ^ (int(x) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        h = (h + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = h
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        h = z ^ (z >> 31)
    return h & 0x7FFFFFFFFFFFFFFF
