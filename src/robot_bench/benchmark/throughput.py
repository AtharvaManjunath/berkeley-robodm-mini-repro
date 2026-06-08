from __future__ import annotations

import statistics
import time

from robot_bench.benchmark.dummy_training import TinyModel, dummy_forward
from robot_bench.loaders.base import FormatLoader


def run_loop(
    loader: FormatLoader,
    path,
    episode_ids: list[str],
    batch_size: int,
    warmup_batches: int,
    num_batches: int,
    include_backward: bool = False,
) -> dict[str, float]:
    if not episode_ids:
        raise RuntimeError("no episodes available for benchmark")
    model = TinyModel()
    latencies: list[float] = []
    pure_load: list[float] = []
    total_steps = 0
    total_frames = 0
    measured_episodes = 0
    total_iters = warmup_batches + num_batches
    for batch_idx in range(total_iters):
        ids = [episode_ids[(batch_idx * batch_size + j) % len(episode_ids)] for j in range(batch_size)]
        load_start = time.perf_counter()
        episodes = [loader.load_episode(path, ep_id) for ep_id in ids]
        load_end = time.perf_counter()
        steps, frames = dummy_forward(episodes, model, include_backward=include_backward)
        end = time.perf_counter()
        if batch_idx >= warmup_batches:
            latencies.append(end - load_start)
            pure_load.append(load_end - load_start)
            total_steps += steps
            total_frames += frames
            measured_episodes += len(episodes)
    total_time = sum(latencies)
    sorted_lat = sorted(latencies)
    p95_ms = 0.0
    if sorted_lat:
        p95_idx = min(len(sorted_lat) - 1, int(round(0.95 * (len(sorted_lat) - 1))))
        p95_ms = sorted_lat[p95_idx] * 1000
    return {
        "benchmark_seconds": total_time,
        "pure_load_time_seconds": sum(pure_load),
        "throughput_episodes_per_sec": measured_episodes / total_time if total_time else 0.0,
        "throughput_frames_per_sec": total_frames / total_time if total_time else 0.0,
        "mean_batch_latency_ms": statistics.mean(latencies) * 1000 if latencies else 0.0,
        "p50_batch_latency_ms": statistics.median(latencies) * 1000 if latencies else 0.0,
        "p95_batch_latency_ms": p95_ms,
        "steps_loaded": float(total_steps),
        "frames_loaded": float(total_frames),
    }
