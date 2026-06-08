from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _bar(df: pd.DataFrame, y: str, title: str, ylabel: str, out: Path, log: bool = False) -> None:
    ok = df[df["status"] == "ok"]
    plt.figure(figsize=(8, 5))
    if ok.empty:
        plt.text(0.5, 0.5, "No successful rows", ha="center")
    else:
        agg = ok.groupby("format")[y].mean().sort_values()
        plt.bar(agg.index, agg.values)
        if log:
            plt.yscale("log")
    plt.title(title)
    plt.xlabel("Format")
    plt.ylabel(ylabel)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def write_plots(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / "disk_size_by_format.png",
        out_dir / "throughput_by_format.png",
        out_dir / "peak_ram_by_format.png",
        out_dir / "size_vs_throughput_scatter.png",
    ]
    _bar(df, "disk_bytes", "Disk size by format", "Bytes (log scale)", paths[0], log=True)
    _bar(df, "throughput_episodes_per_sec", "Throughput by format", "Episodes / second", paths[1])
    _bar(df, "peak_rss_mb", "Peak RAM by format", "Peak RSS MB", paths[2])
    ok = df[df["status"] == "ok"]
    plt.figure(figsize=(7, 5))
    if ok.empty:
        plt.text(0.5, 0.5, "No successful rows", ha="center")
    else:
        for fmt, sub in ok.groupby("format"):
            plt.scatter(sub["disk_bytes"], sub["throughput_episodes_per_sec"], label=fmt)
        plt.xscale("log")
        plt.legend()
    plt.title("Size vs throughput")
    plt.xlabel("Disk bytes (log scale)")
    plt.ylabel("Episodes / second")
    plt.tight_layout()
    plt.savefig(paths[3])
    plt.close()
    return paths
