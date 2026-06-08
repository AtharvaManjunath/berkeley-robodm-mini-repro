from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robot_bench.converters.video_utils import read_video
from robot_bench.episode_schema import Episode, Step
from robot_bench.loaders.base import FormatLoader


class LeRobotStyleLoader(FormatLoader):
    format_name = "lerobot"
    adapter_kind = "style"

    def list_episode_ids(self, path: Path) -> list[str]:
        return sorted(p.stem for p in (path / "data").glob("*.parquet"))

    def load_episode(self, path: Path, episode_id: str) -> Episode:
        meta_all = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        ep_meta = json.loads((path / f"{episode_id}.json").read_text(encoding="utf-8"))
        df = pd.read_parquet(path / "data" / f"{episode_id}.parquet")
        videos = {
            key: read_video(path / rel)
            for key, rel in ep_meta.get("video_files", {}).items()
        }
        steps: list[Step] = []
        for _, row in df.iterrows():
            idx = int(row["step_index"])
            obs: dict[str, Any] = {key: frames[idx] for key, frames in videos.items()}
            for col in df.columns:
                if col.startswith("observation."):
                    obs[col.removeprefix("observation.")] = np.asarray(row[col], dtype=np.float32)
            steps.append(
                Step(
                    observations=obs,
                    action=np.asarray(row["action"], dtype=np.float32),
                    reward=None if pd.isna(row["reward"]) else float(row["reward"]),
                    discount=None if pd.isna(row["discount"]) else float(row["discount"]),
                    is_first=bool(row["is_first"]),
                    is_last=bool(row["is_last"]),
                    is_terminal=bool(row["is_terminal"]),
                    language_instruction=None if pd.isna(row["language_instruction"]) else str(row["language_instruction"]),
                )
            )
        return Episode(episode_id=episode_id, dataset_name=meta_all["dataset_name"], metadata=ep_meta.get("metadata", {}), steps=steps)
