from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

from robot_bench.utils.fs import file_sizes, recursive_size_bytes


@dataclass(slots=True)
class DiskMetrics:
    disk_bytes: int
    mean_episode_bytes: float
    median_episode_bytes: float
    p95_episode_bytes: float


def measure_disk(path: Path, episode_count: int) -> DiskMetrics:
    sizes = file_sizes(path)
    total = recursive_size_bytes(path)
    if not sizes:
        return DiskMetrics(total, 0.0, 0.0, 0.0)
    approx_episode_sizes = sizes if episode_count <= 0 else [total / episode_count] * episode_count
    sorted_sizes = sorted(approx_episode_sizes)
    p95_idx = min(len(sorted_sizes) - 1, int(round(0.95 * (len(sorted_sizes) - 1))))
    return DiskMetrics(
        disk_bytes=total,
        mean_episode_bytes=float(statistics.mean(approx_episode_sizes)),
        median_episode_bytes=float(statistics.median(approx_episode_sizes)),
        p95_episode_bytes=float(sorted_sizes[p95_idx]),
    )
