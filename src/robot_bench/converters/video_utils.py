from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np


def write_video(path: Path, frames: np.ndarray, codec: str) -> tuple[str, bool]:
    """Write uint8 frames [T,H,W,C] as mp4; return codec and fallback flag."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.ndim == 4 and frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3] frames, got {frames.shape}")
    try:
        if codec == "av1":
            iio.imwrite(
                path,
                frames,
                fps=10,
                codec="libaom-av1",
                macro_block_size=1,
                output_params=["-crf", "30", "-b:v", "0"],
            )
            return "av1", False
        iio.imwrite(
            path,
            frames,
            fps=10,
            codec="libx264",
            macro_block_size=1,
            output_params=["-crf", "23", "-pix_fmt", "yuv420p"],
        )
        return "h264", codec != "h264"
    except Exception:
        np.save(path.with_suffix(".npy"), frames)
        return "npy-video-fallback", True


def read_video(path: Path) -> np.ndarray:
    if path.exists():
        return np.asarray(iio.imread(path), dtype=np.uint8)
    npy = path.with_suffix(".npy")
    if npy.exists():
        return np.asarray(np.load(npy), dtype=np.uint8)
    raise FileNotFoundError(path)
