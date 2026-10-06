"""Small shared helpers: RAM guard (operator rule 2026-10-05), checkpoints, stats."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

DATA = Path("/mnt/weights/ai/robot-lab-data/gr00t-rl")
OOM_FLAG = DATA / "OOM_EVENT"     # presence = a RAM event happened tonight -> queue falls back to one GPU job at a time
MIN_AVAIL_GB = 12.0


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def rss_gb(pid: int | None = None) -> float:
    pid = pid or os.getpid()
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024 / 1024
    return float("nan")


def flag_oom(reason: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with open(OOM_FLAG, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid(), "reason": reason}) + "\n")


def ram_guard(where: str) -> None:
    """Raise SystemExit(3) and flag the OOM event if MemAvailable < 12 GB."""
    avail = mem_available_gb()
    if avail < MIN_AVAIL_GB:
        flag_oom(f"{where}: MemAvailable {avail:.1f} GB < {MIN_AVAIL_GB} GB")
        print(f"RAM GUARD: MemAvailable {avail:.1f} GB < {MIN_AVAIL_GB} GB at {where}; stopping (exit 3)", flush=True)
        raise SystemExit(3)


def gpu_used_mib(index_visible: str | None = None) -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip().replace("\n", "; ")
    except Exception as e:  # noqa: BLE001 -- diagnostic only
        return f"nvidia-smi failed: {e}"


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def atomic_torch_save(obj, path: Path) -> None:
    import torch

    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
