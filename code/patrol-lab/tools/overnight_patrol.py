"""Overnight orchestrator for docs/overnight/2026-09-25.md. Stages run in order; within a stage, Isaac runs go
in parallel (GPU 0, <= --parallel processes). Each run's verdict comes from its own report.json, never from log
text. Each stage writes status/<stage>.json; a failed stage never blocks independent ones.

    systemd-run --user --unit=patrol-night -p MemoryHigh=44G -p MemoryMax=50G --working-directory=$PWD \
        uv run python tools/overnight_patrol.py --stages s1,s2,s6,s3eval,s4 --rl-dir <run> [--parallel 4]

Run it as its own systemd unit, never as a child of the editor: on 2026-09-24 six Isaac processes launched in the
same second, systemd-oomd killed the whole VS Code scope (59.5 GB), and the orchestrator, every run and the
session all died with it. Launches are now serialised: one at a time, each RAMP_S after the last (Isaac grows
to ~7-8 GB after startup, so a free-RAM check at launch time alone doesn't see the others coming), and only with
MIN_FREE_RAM_GB + ISAAC_GB available.

Stages
  s2d  2-D held-out benchmark: seeds 21-220 x {control, heuristic, heuristic_v1}         (minutes, CPU)
  s1   M1 gate: 10 randomised incident variants in Isaac with the heuristic stack          (~45 min)
  s2   Isaac crowd baseline: held-out seeds 21-28 x {heuristic, control}                   (~45 min)
  s6   FLUX training data: heuristic "bot" episodes recorded from Spot's head camera       (~70 min)
  s3eval  waits for <rl-dir>/DONE; every checkpoint on seeds 21-120 (rl only), then the best one vs
          control/heuristic on 21-220                                                     (minutes, CPU)
  s4   the best RL checkpoint drives Spot in Isaac on the s2 seeds                         (~45 min)
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ISAAC_PY = "/mnt/weights/ai/isaac/IsaacLab/.venv/bin/python"
DATA = REPO / "data"
MIN_FREE_RAM_GB = 20
ISAAC_GB = 9  # measured peak 7.2 GB for one crowd run, plus margin
RAMP_S = 90  # an Isaac process reaches its working set within ~60 s of launch
_launch_lock = threading.Lock()
_last_launch = [0.0]


def free_ram_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 999.0


def wait_for_launch_slot() -> None:
    """One launch at a time, RAMP_S apart, and only with room for another Isaac on top of the reserve."""
    with _launch_lock:
        while time.time() - _last_launch[0] < RAMP_S or free_ram_gb() < MIN_FREE_RAM_GB + ISAAC_GB:
            time.sleep(10)
        _last_launch[0] = time.time()


def isaac(args: list[str], log: Path, timeout_s: int = 5400) -> int:
    wait_for_launch_slot()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMNI_KIT_ACCEPT_EULA": "YES", "PYTHONUNBUFFERED": "1"}
    with open(log, "w") as f:
        try:
            return subprocess.run([ISAAC_PY, *args], cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            return 124


def report_of(log: Path) -> dict | None:
    for line in reversed(log.read_text(errors="replace").splitlines()):
        if line.startswith("REPORT "):
            return json.loads(line[7:])
    return None


class Night:
    def __init__(self, root: Path, parallel: int, rl_dir: Path | None = None):
        self.root, self.parallel, self.rl_dir = root, parallel, rl_dir
        self.best_ckpt: Path | None = None
        (root / "status").mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)

    def status(self, stage: str, ok: bool, note: str, extra: dict | None = None) -> None:
        doc = {"stage": stage, "status": "ok" if ok else "failed", "note": note, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
               **(extra or {})}
        (self.root / "status" / f"{stage}.json").write_text(json.dumps(doc, indent=2))
        print(f"[{time.strftime('%H:%M:%S')}] STAGE {stage} -> {doc['status']}  {note}", flush=True)

    def isaac_batch(self, stage: str, jobs: list[tuple[str, list[str]]]) -> list[dict]:
        def one(job):
            name, args = job
            log = self.root / "logs" / f"{stage}_{name}.log"
            rc = isaac(args, log)
            rep = report_of(log)
            return {"name": name, "rc": rc, "report": rep}
        with ThreadPoolExecutor(max_workers=self.parallel) as ex:
            results = list(ex.map(one, jobs))
        (self.root / f"{stage}_results.json").write_text(json.dumps(results, indent=2))
        return results

    # --- stages ---------------------------------------------------------------------------------------------------
    def s2d(self):
        log = self.root / "logs" / "s2d.log"
        with open(log, "w") as f:
            rc = subprocess.run(["uv", "run", "python", "-m", "benchmarks.avoidance.sim2d", "--seeds", "21-220",
                                 "--controllers", "control,heuristic,heuristic_v1", "--workers", "48",
                                 "--out", str(self.root / "sim2d")], cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
        summary = next((self.root / "sim2d").glob("*/summary.json"), None)
        ok = rc == 0 and summary is not None
        by = json.loads(summary.read_text())["by_controller"] if summary else {}
        self.status("s2d", ok, " | ".join(f"{k}: contact {v['episodes_with_contact']}/{v['episodes']}" for k, v in by.items()),
                    {"summary": str(summary) if summary else None})

    def s1(self):
        subprocess.run(["uv", "run", "python", "-m", "scenarios.variants", "scenarios/library/SCN-M0-001.yaml", "--seeds",
                        "1-5", "--out", str(DATA / "scenarios/variants")], cwd=REPO, check=True, capture_output=True)
        subprocess.run(["uv", "run", "python", "-m", "scenarios.variants", "scenarios/library/SCN-M0-002.yaml", "--seeds",
                        "1-5", "--out", str(DATA / "scenarios/variants")], cwd=REPO, check=True, capture_output=True)
        files = sorted((DATA / "scenarios/variants").glob("SCN-V00[12]-00[1-5].yaml"))
        jobs = [(f.stem, ["tools/run_scenario_isaac.py", str(f), "--controller", "heuristic"]
                 + (["--video"] if f.stem in ("SCN-V001-001", "SCN-V002-001") else [])) for f in files]
        res = self.isaac_batch("s1", jobs)
        passed = [r["name"] for r in res if r["report"] and r["report"].get("passed")]
        failed = {r["name"]: (r["report"] or {}).get("misses", []) + (r["report"] or {}).get("false_alarms", [])
                  + (["fallen"] if (r["report"] or {}).get("fallen") else []) + ([] if r["report"] else [f"no report (rc={r['rc']})"])
                  for r in res if r["name"] not in passed}
        self.status("s1", len(passed) == len(files), f"{len(passed)}/{len(files)} variants perfect", {"failed": failed})

    def s2(self):
        subprocess.run(["uv", "run", "python", "-m", "benchmarks.avoidance.crowd", "--seeds", "21-28"], cwd=REPO,
                       check=True, capture_output=True)
        jobs = []
        for seed in range(21, 29):
            f = DATA / "scenarios/crowd" / f"SCN-AV-{seed:03d}.yaml"
            for arm in ("heuristic", "control"):
                jobs.append((f"{f.stem}_{arm}", ["tools/run_scenario_isaac.py", str(f), "--controller", arm]
                             + (["--video"] if seed == 21 else [])))
        res = self.isaac_batch("s2", jobs)
        summary = {}
        for arm in ("heuristic", "control"):
            rs = [r["report"] for r in res if r["name"].endswith(arm) and r["report"]]
            summary[arm] = {"runs": len(rs), "near_miss_runs": sum(1 for r in rs if r["avoidance"]["near_misses"]),
                            "contact_runs": sum(1 for r in rs if r["avoidance"]["contacts"]),
                            "fallen": sum(1 for r in rs if r["fallen"]), "complete": sum(1 for r in rs if r["patrol_complete"]),
                            "closest_m": sorted(min(r["avoidance"]["min_clearance_m"].values(), default=99) for r in rs)}
        ok = summary["heuristic"]["runs"] == 8 and summary["control"]["runs"] == 8
        self.status("s2", ok, " | ".join(f"{a}: contact {v['contact_runs']}/{v['runs']}, fallen {v['fallen']}"
                                         for a, v in summary.items()), {"summary": summary})

    def s6(self):
        subprocess.run(["uv", "run", "python", "-m", "benchmarks.avoidance.crowd", "--seeds", "101-130"], cwd=REPO,
                       check=True, capture_output=True)
        out = Path("/mnt/weights/ai/patrol-lab-data/flux_data/heuristic_bot")
        out.mkdir(parents=True, exist_ok=True)
        jobs = [(f"SCN-AV-{s:03d}", ["tools/run_scenario_isaac.py", str(DATA / "scenarios/crowd" / f"SCN-AV-{s:03d}.yaml"),
                                     "--controller", "heuristic", "--record-head", str(out)]) for s in range(101, 131)]
        res = self.isaac_batch("s6", jobs)
        episodes = sorted(out.glob("*.npz"))
        self.status("s6", len(episodes) >= 20, f"{len(episodes)} episodes recorded", {"dir": str(out)})

    def s3eval(self):
        if self.rl_dir is None:
            raise ValueError("s3eval needs --rl-dir")
        while not (self.rl_dir / "DONE").exists():  # the training unit writes DONE last
            time.sleep(60)
        py = [ISAAC_PY, "-m", "benchmarks.avoidance.rl.eval"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO)}
        out = self.root / "rl_eval"

        def evaluate(ckpt: Path, seeds: str, ctrls: str, tag: str) -> dict:
            log = self.root / "logs" / f"s3eval_{tag}.log"
            with open(log, "w") as f:
                subprocess.run(py + ["--ckpt", str(ckpt), "--seeds", seeds, "--controllers", ctrls, "--workers", "24",
                                     "--out", str(out / tag)], cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
                               check=True)
            return json.loads(next((out / tag).glob("*/summary.json")).read_text())["by_controller"]

        scan = {}
        for ck in sorted(self.rl_dir.glob("ckpt_*.pt")):
            r = evaluate(ck, "21-120", "rl", ck.stem)["rl"]
            scan[ck.name] = {"contact": r["episodes_with_contact"], "near_miss": r["episodes_with_near_miss"],
                             "completed": r["completed"]}
        best = min(scan, key=lambda k: (scan[k]["contact"], scan[k]["near_miss"], -scan[k]["completed"]))
        self.best_ckpt = self.rl_dir / best
        by = evaluate(self.best_ckpt, "21-220", "rl,control,heuristic", "best")
        self.status("s3eval", True, f"best {best} | " + " | ".join(
            f"{k}: contact {v['episodes_with_contact']}/{v['episodes']}" for k, v in by.items()),
            {"best_ckpt": str(self.best_ckpt), "checkpoint_scan_seeds_21_120": scan})

    def s4(self):
        if self.best_ckpt is None:
            done = self.root / "status" / "s3eval.json"
            self.best_ckpt = Path(json.loads(done.read_text())["best_ckpt"])
        jobs = [(f"SCN-AV-{seed:03d}_rl", ["tools/run_scenario_isaac.py", str(DATA / "scenarios/crowd" / f"SCN-AV-{seed:03d}.yaml"),
                                           "--controller", f"rl:{self.best_ckpt}"] + (["--video"] if seed == 21 else []))
                for seed in range(21, 29)]
        res = self.isaac_batch("s4", jobs)
        rs = [r["report"] for r in res if r["report"]]
        summary = {"runs": len(rs), "near_miss_runs": sum(1 for r in rs if r["avoidance"]["near_misses"]),
                   "contact_runs": sum(1 for r in rs if r["avoidance"]["contacts"]),
                   "fallen": sum(1 for r in rs if r["fallen"]), "complete": sum(1 for r in rs if r["patrol_complete"]),
                   "closest_m": sorted(min(r["avoidance"]["min_clearance_m"].values(), default=99) for r in rs)}
        self.status("s4", len(rs) == 8, f"rl: contact {summary['contact_runs']}/{summary['runs']}, "
                                        f"fallen {summary['fallen']}", {"summary": summary, "ckpt": str(self.best_ckpt)})

    def lc(self):
        """Live crowds in Isaac (D40): seeds 1-8 x {heuristic, control} on the tier given by --lc-tier."""
        tier = self.lc_tier
        subprocess.run(["uv", "run", "python", "-m", "benchmarks.avoidance.crowd", "--seeds", "1-8", "--live", tier],
                       cwd=REPO, check=True, capture_output=True)
        tag = "" if tier == "base" else f"{tier.upper()}-"
        jobs = []
        for seed in range(1, 9):
            f = DATA / "scenarios/crowd" / f"SCN-LC-{tag}{seed:03d}.yaml"
            for arm in ("heuristic", "control"):
                jobs.append((f"{f.stem}_{arm}", ["tools/run_scenario_isaac.py", str(f), "--controller", arm]
                             + (["--video"] if seed == 2 else [])))
        res = self.isaac_batch("lc", jobs)
        summary = {}
        for arm in ("heuristic", "control"):
            rs = [r["report"] for r in res if r["name"].endswith(arm) and r["report"]]
            lk = [r["live_crowd"] for r in rs]
            com, hit = sum(x["committed"] for x in lk), sum(x["hit"] for x in lk)
            summary[arm] = {"runs": len(rs), "locks": sum(x["locks"] for x in lk), "committed": com, "hit": hit,
                            "hit_per_commit": round(hit / com, 3) if com else None,
                            "fallen": sum(1 for r in rs if r["fallen"]), "complete": sum(1 for r in rs if r["patrol_complete"]),
                            "ambient_contact_runs": sum(1 for x in lk if x["ambient_contacts"])}
        ok = all(v["runs"] == 8 for v in summary.values())
        self.status("lc", ok, " | ".join(f"{a}: hit {v['hit']}/{v['committed']} commits, fallen {v['fallen']}, "
                                        f"complete {v['complete']}" for a, v in summary.items()), {"summary": summary, "tier": tier})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="s2d,s1,s2,s6")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--rl-dir", default=None, help="PPO run directory (for s3eval/s4)")
    ap.add_argument("--lc-tier", default="base", help="live-crowd tier for the lc stage")
    ap.add_argument("--root", default=str(DATA / "overnight" / time.strftime("night-%Y%m%d-%H%M")))
    a = ap.parse_args()
    night = Night(Path(a.root), a.parallel, Path(a.rl_dir) if a.rl_dir else None)
    night.lc_tier = a.lc_tier
    link = DATA / "overnight" / "latest"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(a.root))
    print(f"ROOT {a.root}", flush=True)
    for stage in a.stages.split(","):
        try:
            getattr(night, stage)()
        except Exception as exc:  # a crashed stage is recorded, and the night goes on
            night.status(stage, False, f"crashed: {type(exc).__name__}: {exc}")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
