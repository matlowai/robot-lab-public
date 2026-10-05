"""sim2d's opt-in per-step trace (run_episode(trace_path=...), rl/eval.py --trace-dir): off by default, and turning
it on changes nothing about the episode row or the summary (it only reads state)."""

import json

import numpy as np

from benchmarks.avoidance import sim2d
from benchmarks.avoidance.crowd import generate_live, to_scenario_file
from benchmarks.avoidance.pedestrians import KIND_NAMES

SEED, TIER, MAX_S = 121, "hard", 25.0


def _untimed(row):
    """The row without its wall-clock field (ms_per_decision differs run to run, traced or not)."""
    return {k: v for k, v in row.items() if k != "ms_per_decision"}


def _untimed_summary(rows):
    return {c: {k: v for k, v in s.items() if k != "ms_per_decision_median"} for c, s in sim2d.summarize(rows).items()}


def _ensure_scenario():
    to_scenario_file(generate_live(SEED, TIER), sim2d.REPO / "data/scenarios/crowd")


def test_trace_off_by_default_and_identical_rows(tmp_path):
    _ensure_scenario()
    for ctrl in ("control", "heuristic"):
        plain = sim2d.run_episode(SEED, ctrl, max_s=MAX_S, live=TIER)
        path = tmp_path / f"{ctrl}.npz"
        traced = sim2d.run_episode(SEED, ctrl, max_s=MAX_S, live=TIER, trace_path=str(path))
        assert json.dumps(_untimed(plain), sort_keys=True) == json.dumps(_untimed(traced), sort_keys=True)
        assert _untimed_summary([plain]) == _untimed_summary([traced])
        assert path.exists()
    assert sorted(p.name for p in tmp_path.glob("*.npz")) == ["control.npz", "heuristic.npz"]


def test_trace_contents_match_the_episode(tmp_path):
    _ensure_scenario()
    path = tmp_path / "h.npz"
    row = sim2d.run_episode(SEED, "heuristic", max_s=MAX_S, live=TIER, trace_path=str(path))
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    cols = {c: i for i, c in enumerate(meta["columns"])}
    rob, pos = z["robot"], z["pos"]
    assert rob.shape[0] == pos.shape[0] == z["state"].shape[0] == sum(row["statuses"].values())
    assert pos.shape[1] == len(meta["ids"]) == len(row["min_clearance_m"])
    assert set(KIND_NAMES[k] for k in z["kind"]) <= set(KIND_NAMES)
    # the per-person minimum clearance recomputed from the trace equals the episode's (rounded to 3 dp)
    d = np.hypot(pos[..., 0] - rob[:, None, cols["x"]], pos[..., 1] - rob[:, None, cols["y"]]) - sim2d.PERSON_R - sim2d.SPOT_R
    for pid, c in zip(meta["ids"], d.min(0)):
        assert abs(round(float(c), 3) - row["min_clearance_m"][pid]) <= 1e-3
    # position integrates the recorded simulator velocity (one unicycle step between consecutive distinct times)
    k = np.nonzero(np.diff(rob[:, cols["t"]]) > 1e-9)[0][5]
    x, y, yaw = rob[k + 1, cols["x"]], rob[k + 1, cols["y"]], rob[k + 1, cols["yaw"]]
    vx, vy = rob[k + 1, cols["vx"]], rob[k + 1, cols["vy"]]
    assert abs(yaw - (rob[k, cols["yaw"]] + rob[k + 1, cols["wz"]] * sim2d.DT)) < 1e-9
    assert abs(x - (rob[k, cols["x"]] + (vx * np.cos(yaw) - vy * np.sin(yaw)) * sim2d.DT)) < 1e-9
    assert np.isnan(rob[:, cols["ctrl_vx"]]).all()  # the heuristic keeps no velocity estimate
