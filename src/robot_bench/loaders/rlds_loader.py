from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_bench.config import BenchmarkConfig, DatasetConfig, ProfileConfig
from robot_bench.episode_schema import Episode, Step
from robot_bench.loaders.base import FormatLoader


class RLDSStyleLoader(FormatLoader):
    format_name = "rlds"
    adapter_kind = "style"

    def list_episode_ids(self, path: Path) -> list[str]:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        return list(manifest["episodes"])

    def load_episode(self, path: Path, episode_id: str) -> Episode:
        ep_dir = path / episode_id
        meta = json.loads((ep_dir / "metadata.json").read_text(encoding="utf-8"))
        data = np.load(ep_dir / "steps.npz", allow_pickle=False)
        obs_keys = [k for k in data.files if k.startswith("observation__")]
        action_keys = [k for k in data.files if k.startswith("action__")]
        n = len(data["is_first"])
        steps: list[Step] = []
        for i in range(n):
            obs = {k.removeprefix("observation__"): np.asarray(data[k][i]) for k in obs_keys}
            if action_keys == ["action__value"]:
                action: Any = np.asarray(data["action__value"][i])
            else:
                action = {k.removeprefix("action__"): np.asarray(data[k][i]) for k in action_keys}
            reward = None if np.isnan(data["reward"][i]) else float(data["reward"][i])
            discount = None if np.isnan(data["discount"][i]) else float(data["discount"][i])
            steps.append(
                Step(
                    observations=obs,
                    action=action,
                    reward=reward,
                    discount=discount,
                    is_first=bool(data["is_first"][i]),
                    is_last=bool(data["is_last"][i]),
                    is_terminal=bool(data["is_terminal"][i]),
                    language_instruction=str(data["language_instruction"][i]) or None,
                )
            )
        return Episode(episode_id=episode_id, dataset_name=meta["dataset_name"], metadata=meta["metadata"], steps=steps, schema_version=meta["schema_version"])


def load_tfds_rlds_subset(
    ds: DatasetConfig,
    config: BenchmarkConfig,
    profile: ProfileConfig,
    profile_name: str,
) -> list[Episode]:
    raise RuntimeError(
        "Official TFDS/RLDS public loading is intentionally guarded in this scaffold. "
        "Use synthetic CI data by default, or extend this function for a local TFDS cache."
    )
