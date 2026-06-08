from __future__ import annotations

from pathlib import Path

import pandas as pd

from robot_bench.reports.aggregate import headline_table


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = [str(row[c]).replace("\n", " ") for c in df.columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_markdown_report(df: pd.DataFrame, out_path: Path) -> Path:
    ok = df[df["status"] == "ok"]
    failed = df[df["status"] != "ok"]
    lines = [
        "# robot-data-format-benchmark Report",
        "",
        "## Executive Summary",
        "",
        f"This run produced {len(df)} result rows, with {len(ok)} successful rows and {len(failed)} failed/skipped rows. "
        "The harness is a reproducible benchmark implementation and does not claim to be an official Robo-DM reimplementation.",
        "",
        "## Hardware / Software Environment",
        "",
        _md_table(df[["hostname", "platform", "python", "package_versions"]].drop_duplicates().head(5)),
        "",
        "## Dataset Table",
        "",
        _md_table(df.groupby("dataset_name").agg(rows=("format", "count"), ok=("status", lambda s: int((s == "ok").sum()))).reset_index()),
        "",
        "## Format Implementation Table",
        "",
        _md_table(df[["format", "adapter_kind", "codec", "compression", "notes"]].drop_duplicates()),
        "",
        "## Benchmark Methodology",
        "",
        "Episodes are selected deterministically from each configured dataset and saved under `artifacts/manifests/`. "
        "Measured batch latency includes full episode loading, image decode into arrays, and a deterministic Torch CPU dummy forward workload. "
        "Warmup batches are excluded. RSS is sampled with psutil for the main process and child processes.",
        "",
        "## Headline Results",
        "",
        _md_table(headline_table(df)),
        "",
        "## Per-Dataset Results",
        "",
    ]
    for name, sub in df.groupby("dataset_name"):
        cols = ["format", "status", "disk_bytes", "throughput_episodes_per_sec", "throughput_frames_per_sec", "peak_rss_mb", "error_message"]
        lines.extend([f"### {name}", "", _md_table(sub[cols]), ""])
    lines.extend(
        [
            "## Disk-Size Comparison",
            "",
            "See `artifacts/plots/disk_size_by_format.png`.",
            "",
            "## Peak-RAM Comparison",
            "",
            "See `artifacts/plots/peak_ram_by_format.png`.",
            "",
            "## Throughput Comparison",
            "",
            "See `artifacts/plots/throughput_by_format.png` and `artifacts/plots/size_vs_throughput_scatter.png`.",
            "",
            "## Matched / Partial / Failed Replication",
            "",
            "Matched dimensions: disk size, peak RSS, and loading plus dummy-training throughput. "
            "Partial dimensions: official source loaders and official LeRobot/Robo-DM APIs depend on external packages and dataset availability. "
            "Failures remain in the result schema with `status=failed`.",
            "",
            "## Known Limitations",
            "",
            "- Sample-capped results are not full paper-scale results.",
            "- Robo-DM-style fallback is not official Robo-DM.",
            "- Codec availability changes disk size and decode throughput.",
            "- Hardware strongly affects throughput and RAM.",
            "",
            "## Implementation Tradeoffs",
            "",
            "See `artifacts/reports/tradeoffs.md` and `docs/TRADEOFFS.md`.",
            "",
            "## Exact Reproduction Commands",
            "",
            "```bash",
            "make install",
            "make smoke",
            "make benchmark-small",
            "make report",
            "```",
            "",
            "## Citation / Bibliography",
            "",
            "Berkeley Robo-DM paper and dataset documentation should be cited by downstream users when publishing benchmark results. "
            "This repository is a harness for replication attempts and extensions.",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_tradeoffs_report(out_path: Path) -> Path:
    text = """# Implementation Tradeoffs

- RLDS source format may not be regenerated exactly for every dataset; the local RLDS-style fixture is explicitly labeled as style.
- LeRobot-style conversion flattens unsupported low-dimensional modalities into Parquet vectors and stores image streams as video.
- Robo-DM-style fallback is not official Robo-DM. It is a self-contained per-episode container intended to mimic benchmark-relevant storage/loading behavior.
- Video codec availability can affect results. AV1 CRF 30 is attempted when available; otherwise H.264 CRF 23 or an array fallback is recorded.
- Sample-capped results are workstation-friendly and not full-paper-scale results.
- Hardware, filesystem cache state, and CPU video decode strongly affect throughput and RAM.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
