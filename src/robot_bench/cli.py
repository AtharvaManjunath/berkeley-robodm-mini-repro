from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from robot_bench.benchmark.runner import benchmark_converted, convert_datasets, write_results
from robot_bench.config import load_config
from robot_bench.dataset_registry import load_dataset, save_selection_manifest
from robot_bench.logging_utils import configure_logging
from robot_bench.reports.markdown import write_markdown_report, write_tradeoffs_report
from robot_bench.reports.plots import write_plots
from robot_bench.reports.replication import write_replication_matrix
from robot_bench.seeds import set_global_seed
from robot_bench.utils.fs import ensure_dir

app = typer.Typer(no_args_is_help=True)


def _formats(value: str) -> list[str]:
    allowed = {"rlds", "hdf5", "lerobot", "robodm"}
    formats = [v.strip() for v in value.split(",") if v.strip()]
    bad = sorted(set(formats) - allowed)
    if bad:
        raise typer.BadParameter(f"unsupported format(s): {bad}")
    return formats


def _load(path: Path, allow_downloads: bool, allow_fallbacks: bool):
    cfg = load_config(path)
    cfg.allow_downloads = allow_downloads or cfg.allow_downloads
    cfg.allow_fallbacks = allow_fallbacks or cfg.allow_fallbacks
    set_global_seed(cfg.seed)
    return cfg


@app.command()
def prepare(
    config: Path = typer.Option(Path("configs/datasets.yaml")),
    profile: str = "small",
    allow_downloads: bool = False,
    allow_fallbacks: bool = False,
    verbose: bool = False,
) -> None:
    configure_logging(verbose)
    cfg = _load(config, allow_downloads, allow_fallbacks)
    prof = cfg.profile(profile)
    selections = {}
    for ds in cfg.datasets:
        try:
            selections[ds.name] = load_dataset(ds, cfg, prof, profile)
        except Exception as exc:
            typer.echo(f"prepare failed for {ds.name}: {type(exc).__name__}: {exc}")
            selections[ds.name] = []
    path = save_selection_manifest(cfg, profile, selections)
    typer.echo(f"wrote {path}")


@app.command()
def convert(
    config: Path = typer.Option(Path("configs/datasets.yaml")),
    formats: str = "rlds,hdf5,lerobot,robodm",
    profile: str = "small",
    allow_downloads: bool = False,
    allow_fallbacks: bool = False,
    verbose: bool = False,
) -> None:
    configure_logging(verbose)
    cfg = _load(config, allow_downloads, allow_fallbacks)
    results = convert_datasets(cfg, profile, _formats(formats))
    for res in results:
        typer.echo(f"{res.dataset_name}/{res.format_name}: {res.status} {res.error_message}")


@app.command()
def benchmark(
    config: Path = typer.Option(Path("configs/datasets.yaml")),
    formats: str = "rlds,hdf5,lerobot,robodm",
    profile: str = "small",
    allow_downloads: bool = False,
    allow_fallbacks: bool = False,
    verbose: bool = False,
) -> None:
    configure_logging(verbose)
    cfg = _load(config, allow_downloads, allow_fallbacks)
    rows = benchmark_converted(cfg, profile, _formats(formats))
    path = write_results(rows, cfg.artifact_dir)
    typer.echo(f"wrote {path}")


@app.command("run-all")
def run_all(
    config: Path = typer.Option(Path("configs/datasets.yaml")),
    formats: str = "rlds,hdf5,lerobot,robodm",
    profile: str = "small",
    allow_downloads: bool = False,
    allow_fallbacks: bool = False,
    verbose: bool = False,
) -> None:
    configure_logging(verbose)
    cfg = _load(config, allow_downloads, allow_fallbacks)
    fmts = _formats(formats)
    conversions = convert_datasets(cfg, profile, fmts)
    rows = benchmark_converted(cfg, profile, fmts, conversions)
    csv_path = write_results(rows, cfg.artifact_dir)
    _report_from_csv(csv_path, cfg.artifact_dir)
    typer.echo(f"wrote {csv_path}")


@app.command()
def report(
    results: Path = typer.Option(Path("artifacts/results/results.csv")),
    artifact_dir: Path = typer.Option(Path("artifacts")),
) -> None:
    _report_from_csv(results, artifact_dir)
    typer.echo(f"wrote reports under {artifact_dir}")


def _report_from_csv(results: Path, artifact_dir: Path) -> None:
    df = pd.read_csv(results)
    ensure_dir(artifact_dir / "reports")
    write_markdown_report(df, artifact_dir / "reports" / "benchmark_report.md")
    write_replication_matrix(df, artifact_dir / "reports" / "replication_matrix.md")
    write_tradeoffs_report(artifact_dir / "reports" / "tradeoffs.md")
    write_plots(df, artifact_dir / "plots")


if __name__ == "__main__":
    app()
