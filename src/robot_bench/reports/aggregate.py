from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    return (
        ok.groupby(["format", "adapter_kind"], dropna=False)
        .agg(
            datasets=("dataset_name", "nunique"),
            disk_bytes_mean=("disk_bytes", "mean"),
            throughput_eps_mean=("throughput_episodes_per_sec", "mean"),
            peak_rss_mb_mean=("peak_rss_mb", "mean"),
        )
        .reset_index()
    )
