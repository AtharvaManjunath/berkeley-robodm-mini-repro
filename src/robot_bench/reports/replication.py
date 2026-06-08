from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_replication_matrix(df: pd.DataFrame, out_path: Path) -> Path:
    lines = [
        "# Replication Matrix",
        "",
        "| Paper benchmark dimension | Implemented | Exact or approximate | Reason / notes |",
        "|---|---:|---|---|",
        "| On-disk size | yes | approximate | Same metric family; sample-capped profile defaults differ from full paper scale. |",
        "| Loading throughput | yes | approximate | Dummy CPU training-loop boundary is documented and may differ from paper implementation. |",
        "| Peak RAM | yes | approximate | psutil RSS sampler includes main process and children at configurable interval. |",
        "| RLDS source loading | partial | source/style | Source loader is used only when configured and available; CI uses local RLDS-style fixture. |",
        "| HDF5 | yes | style | One uncompressed .h5 file per episode, matching common robot-learning convention. |",
        "| LeRobot | yes | style | Local Parquet/video layout unless official APIs are integrated externally. |",
        "| Robo-DM | yes | fallback | Clearly labeled Robo-DM-style fallback; not official Robo-DM. |",
        "",
        "## Dataset Coverage",
        "",
        "| Dataset | Statuses observed | Notes |",
        "|---|---|---|",
    ]
    for name, sub in df.groupby("dataset_name"):
        statuses = ", ".join(sorted(set(str(s) for s in sub["status"])))
        notes = "; ".join(sorted(set(str(n) for n in sub.get("notes", []) if str(n) != "nan")))[:200]
        lines.append(f"| {name} | {statuses} | {notes} |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
