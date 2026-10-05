# Patrol Lab

A simulated robot patrol: Boston Dynamics **Spot** on the perimeter and a **humanoid** (a Unitree G1 stand-in for Atlas) for stairs. It runs on Isaac Sim 6.1 / Isaac Lab 3.0, and ROS 2 + Nav2 run in Docker. The robots report **events** (zone entries, falls, open gates, obstacles) to an explainable risk engine; they never judge people.

This folder is a code snapshot (2026-10-04, private repo commit `d05e441`). The plan, decision log and
running notes are not included; the course (Part II) tells the story and cites the results.

## Quickstart (M0, no GPU)

```bash
uv sync                      # Python 3.12 venv with pyyaml, jsonschema, pytest
uv run pytest                # rules, scorer, schemas, robots, missions, invariants, end-to-end scenarios
uv run python tools/run_scenario.py scenarios/library/SCN-M0-001.yaml --validate [--out data/runs]
```

Large outputs (runs, datasets, checkpoints, maps) go in `data/` → `/mnt/weights/ai/patrol-lab-data`. They are never committed.

Sister projects: `/mnt/work/AI/robot-lab` (SO-101 manipulation) and `/mnt/work/AI/flux3-action` (FLUX 3 Action). Isaac lives at `/mnt/weights/ai/isaac/IsaacLab`.
