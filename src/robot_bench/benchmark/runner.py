from __future__ import annotations

import importlib.metadata as md
import json
import platform
import socket
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from robot_bench.benchmark.disk import measure_disk
from robot_bench.benchmark.memory import MemoryMonitor
from robot_bench.benchmark.throughput import run_loop
from robot_bench.config import BenchmarkConfig
from robot_bench.converters.base import ConversionResult
from robot_bench.converters.hdf5_converter import HDF5Converter
from robot_bench.converters.lerobot_style_converter import LeRobotStyleConverter
from robot_bench.converters.rlds_converter import RLDSStyleConverter
from robot_bench.converters.robodm_style_converter import RoboDMStyleConverter
from robot_bench.dataset_registry import load_dataset, save_selection_manifest
from robot_bench.episode_schema import Episode
from robot_bench.loaders.hdf5_loader import HDF5Loader
from robot_bench.loaders.lerobot_style_loader import LeRobotStyleLoader
from robot_bench.loaders.rlds_loader import RLDSStyleLoader
from robot_bench.loaders.robodm_style_loader import RoboDMStyleLoader
from robot_bench.utils.fs import ensure_dir

RESULT_COLUMNS = [
    "dataset_name",
    "format",
    "adapter_kind",
    "codec",
    "compression",
    "profile",
    "episodes_requested",
    "episodes_loaded",
    "steps_loaded",
    "frames_loaded",
    "batch_size",
    "num_batches",
    "workers",
    "disk_bytes",
    "mean_episode_bytes",
    "median_episode_bytes",
    "p95_episode_bytes",
    "throughput_episodes_per_sec",
    "throughput_frames_per_sec",
    "mean_batch_latency_ms",
    "p50_batch_latency_ms",
    "p95_batch_latency_ms",
    "peak_rss_mb",
    "conversion_seconds",
    "benchmark_seconds",
    "pure_load_time_seconds",
    "seed",
    "hostname",
    "platform",
    "python",
    "package_versions",
    "status",
    "error_type",
    "error_message",
    "notes",
]


@dataclass(slots=True)
class BenchmarkRow:
    dataset_name: str
    format: str
    adapter_kind: str = ""
    codec: str = ""
    compression: str = "{}"
    profile: str = "small"
    episodes_requested: int = 0
    episodes_loaded: int = 0
    steps_loaded: int = 0
    frames_loaded: int = 0
    batch_size: int = 0
    num_batches: int = 0
    workers: int = 0
    disk_bytes: int = 0
    mean_episode_bytes: float = 0.0
    median_episode_bytes: float = 0.0
    p95_episode_bytes: float = 0.0
    throughput_episodes_per_sec: float = 0.0
    throughput_frames_per_sec: float = 0.0
    mean_batch_latency_ms: float = 0.0
    p50_batch_latency_ms: float = 0.0
    p95_batch_latency_ms: float = 0.0
    peak_rss_mb: float = 0.0
    conversion_seconds: float = 0.0
    benchmark_seconds: float = 0.0
    pure_load_time_seconds: float = 0.0
    seed: int = 0
    hostname: str = field(default_factory=socket.gethostname)
    platform: str = field(default_factory=platform.platform)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    package_versions: str = "{}"
    status: str = "ok"
    error_type: str = ""
    error_message: str = ""
    notes: str = ""


def package_versions() -> str:
    names = ["numpy", "pandas", "pyarrow", "h5py", "psutil", "torch", "matplotlib"]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = "not-installed"
    return json.dumps(versions, sort_keys=True)


def converter_for(name: str):
    return {
        "rlds": RLDSStyleConverter(),
        "hdf5": HDF5Converter(),
        "lerobot": LeRobotStyleConverter(),
        "robodm": RoboDMStyleConverter(),
    }[name]


def loader_for(name: str):
    return {
        "rlds": RLDSStyleLoader(),
        "hdf5": HDF5Loader(),
        "lerobot": LeRobotStyleLoader(),
        "robodm": RoboDMStyleLoader(),
    }[name]


def convert_datasets(config: BenchmarkConfig, profile_name: str, formats: list[str]) -> list[ConversionResult]:
    profile = config.profile(profile_name)
    selections: dict[str, list[Episode]] = {}
    results: list[ConversionResult] = []
    for ds in config.datasets:
        try:
            episodes = load_dataset(ds, config, profile, profile_name)
            selections[ds.name] = episodes
        except Exception as exc:
            selections[ds.name] = []
            for fmt in formats:
                results.append(
                    ConversionResult(ds.name, fmt, config.artifact_dir / "converted" / profile_name / ds.name / fmt, "", 0.0, status="failed", error_type=type(exc).__name__, error_message=str(exc))
                )
            continue
        for fmt in formats:
            out = config.artifact_dir / "converted" / profile_name / ds.name / fmt
            try:
                results.append(converter_for(fmt).convert(ds.name, episodes, out))
            except Exception as exc:
                results.append(ConversionResult(ds.name, fmt, out, "", 0.0, status="failed", error_type=type(exc).__name__, error_message=str(exc)))
    save_selection_manifest(config, profile_name, selections)
    return results


def benchmark_converted(
    config: BenchmarkConfig,
    profile_name: str,
    formats: list[str],
    conversions: list[ConversionResult] | None = None,
) -> list[BenchmarkRow]:
    profile = config.profile(profile_name)
    conv_by_key = {(c.dataset_name, c.format_name): c for c in conversions or []}
    rows: list[BenchmarkRow] = []
    versions = package_versions()
    for ds in config.datasets:
        for fmt in formats:
            conv = conv_by_key.get((ds.name, fmt))
            path = config.artifact_dir / "converted" / profile_name / ds.name / fmt
            base = BenchmarkRow(
                dataset_name=ds.name,
                format=fmt,
                profile=profile_name,
                episodes_requested=profile.episodes_per_dataset or 0,
                batch_size=profile.batch_size,
                num_batches=profile.num_batches,
                workers=profile.workers,
                seed=config.seed,
                package_versions=versions,
            )
            if conv and conv.status != "ok":
                base.status = "failed"
                base.error_type = conv.error_type
                base.error_message = conv.error_message
                rows.append(base)
                continue
            try:
                loader = loader_for(fmt)
                episode_ids = loader.list_episode_ids(path)
                disk = measure_disk(path, len(episode_ids))
                with MemoryMonitor(config.memory_interval_sec) as mm:
                    metrics = run_loop(
                        loader,
                        path,
                        episode_ids,
                        profile.batch_size,
                        profile.warmup_batches,
                        profile.num_batches,
                        include_backward=config.include_backward,
                    )
                row = base
                row.adapter_kind = conv.adapter_kind if conv else loader.adapter_kind
                row.codec = conv.codec if conv else ""
                row.compression = json.dumps(conv.compression if conv else {}, sort_keys=True)
                row.episodes_loaded = len(episode_ids)
                row.steps_loaded = int(metrics["steps_loaded"])
                row.frames_loaded = int(metrics["frames_loaded"])
                row.disk_bytes = disk.disk_bytes
                row.mean_episode_bytes = disk.mean_episode_bytes
                row.median_episode_bytes = disk.median_episode_bytes
                row.p95_episode_bytes = disk.p95_episode_bytes
                row.throughput_episodes_per_sec = metrics["throughput_episodes_per_sec"]
                row.throughput_frames_per_sec = metrics["throughput_frames_per_sec"]
                row.mean_batch_latency_ms = metrics["mean_batch_latency_ms"]
                row.p50_batch_latency_ms = metrics["p50_batch_latency_ms"]
                row.p95_batch_latency_ms = metrics["p95_batch_latency_ms"]
                row.peak_rss_mb = mm.peak_rss_mb
                row.conversion_seconds = conv.conversion_seconds if conv else 0.0
                row.benchmark_seconds = metrics["benchmark_seconds"]
                row.pure_load_time_seconds = metrics["pure_load_time_seconds"]
                row.notes = conv.notes if conv else ""
                rows.append(row)
            except Exception as exc:
                base.status = "failed"
                base.error_type = type(exc).__name__
                base.error_message = str(exc)
                rows.append(base)
    return rows


def write_results(rows: list[BenchmarkRow], artifact_dir: Path) -> Path:
    out_dir = ensure_dir(artifact_dir / "results")
    csv_path = out_dir / "results.csv"
    jsonl_path = out_dir / "results.jsonl"
    records = [asdict(r) for r in rows]
    pd.DataFrame(records, columns=RESULT_COLUMNS).to_csv(csv_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    return csv_path
