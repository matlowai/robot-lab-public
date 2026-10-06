"""mp4 / PNG-strip writers (system ffmpeg CLI, as tools/gr00t_eval.py does)."""

from __future__ import annotations

import subprocess

import numpy as np


def write_mp4(path, frames, fps: int = 30) -> None:
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "23",
                          "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(np.stack(frames)).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def write_strip(path, frames, n: int = 8) -> None:
    from PIL import Image

    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    Image.fromarray(np.concatenate([frames[i] for i in idx], 0 if frames[0].shape[1] > 300 else 1)).save(path)
