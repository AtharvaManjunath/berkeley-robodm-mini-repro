from __future__ import annotations

from pathlib import Path

import pandas as pd

from robot_bench.benchmark.disk import measure_disk
from robot_bench.benchmark.memory import MemoryMonitor
from robot_bench.benchmark.runner import (
    RESULT_COLUMNS,
    benchmark_converted,
    convert_datasets,
    write_results,
)
from robot_bench.config import BenchmarkConfig, DatasetConfig, ProfileConfig, load_config
from robot_bench.converters.rlds_converter import RLDSStyleConverter
from robot_bench.dataset_registry import load_dataset
from robot_bench.reports.markdown import write_markdown_report
from robot_bench.reports.plots import write_plots


def test_disk_and_memory(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"1234")
    assert measure_disk(tmp_path, 1).disk_bytes == 4
    with MemoryMonitor(0.01) as mm:
        _ = bytearray(1024)
    assert mm.peak_rss_mb > 0


def test_runner_emits_complete_schema(tmp_path: Path) -> None:
    cfg = load_config(Path("configs/synthetic.yaml"))
    cfg.artifact_dir = tmp_path / "artifacts"
    conversions = convert_datasets(cfg, "small", ["rlds", "hdf5"])
    rows = benchmark_converted(cfg, "small", ["rlds", "hdf5"], conversions)
    csv = write_results(rows, cfg.artifact_dir)
    df = pd.read_csv(csv)
    assert list(df.columns) == RESULT_COLUMNS
    assert set(df["status"]) == {"ok"}


def test_report_generation_from_synthetic_results(tmp_path: Path) -> None:
    cfg = load_config(Path("configs/synthetic.yaml"))
    cfg.artifact_dir = tmp_path / "artifacts"
    rows = benchmark_converted(cfg, "small", ["rlds"])
    # No converted files yet: failures should still be schema-complete.
    csv = write_results(rows, cfg.artifact_dir)
    df = pd.read_csv(csv)
    report = write_markdown_report(df, cfg.artifact_dir / "reports" / "benchmark_report.md")
    plots = write_plots(df, cfg.artifact_dir / "plots")
    assert report.exists()
    assert all(p.exists() for p in plots)


def test_local_rlds_source_loads_deterministic_subset(tmp_path: Path, tiny_episodes) -> None:
    source = tmp_path / "source_rlds"
    RLDSStyleConverter().convert("tiny_test", tiny_episodes, source)
    cfg = BenchmarkConfig(
        seed=42,
        artifact_dir=tmp_path / "artifacts",
        datasets=[DatasetConfig(name="tiny_test", source="local_rlds", local_path=source)],
    )
    profile = ProfileConfig(episodes_per_dataset=2, max_steps_per_episode=4)
    loaded_a = load_dataset(cfg.datasets[0], cfg, profile, "small")
    loaded_b = load_dataset(cfg.datasets[0], cfg, profile, "small")
    assert [ep.episode_id for ep in loaded_a] == [ep.episode_id for ep in loaded_b]
    assert len(loaded_a) == 2
    assert all(len(ep.steps) == 4 for ep in loaded_a)
