from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_bench.episode_schema import Episode, Step
from robot_bench.loaders.base import FormatLoader


def _read_group(group: h5py.Group) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            out[key] = _read_group(item)
        else:
            value = item[()]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            out[key] = value
    return out


class HDF5Loader(FormatLoader):
    format_name = "hdf5"
    adapter_kind = "style"

    def list_episode_ids(self, path: Path) -> list[str]:
        return sorted(p.stem for p in path.glob("*.h5"))

    def load_episode(self, path: Path, episode_id: str) -> Episode:
        with h5py.File(path / f"{episode_id}.h5", "r") as h5:
            meta_group = h5["metadata"]
            metadata = json.loads(meta_group.attrs.get("json", "{}"))
            dataset_name = str(meta_group.attrs["dataset_name"])
            schema_version = str(meta_group.attrs["schema_version"])
            obs_all = _read_group(h5["steps"]["observation"])
            action_all = _read_group(h5["steps"]["action"])
            rewards = h5["steps"]["reward"][()]
            discounts = h5["steps"]["discount"][()]
            first = h5["steps"]["is_first"][()]
            last = h5["steps"]["is_last"][()]
            terminal = h5["steps"]["is_terminal"][()]
            lang = h5["steps"]["language_instruction"].asstr()[()]
            n = len(first)
            steps: list[Step] = []
            for i in range(n):
                obs = {k: np.asarray(v[i]) for k, v in obs_all.items() if hasattr(v, "__getitem__")}
                if "value" in action_all:
                    action: Any = np.asarray(action_all["value"][i])
                else:
                    action = {k: np.asarray(v[i]) for k, v in action_all.items()}
                reward = None if np.isnan(rewards[i]) else float(rewards[i])
                discount = None if np.isnan(discounts[i]) else float(discounts[i])
                steps.append(
                    Step(
                        observations=obs,
                        action=action,
                        reward=reward,
                        discount=discount,
                        is_first=bool(first[i]),
                        is_last=bool(last[i]),
                        is_terminal=bool(terminal[i]),
                        language_instruction=str(lang[i]) or None,
                    )
                )
        return Episode(episode_id=episode_id, dataset_name=dataset_name, metadata=metadata, steps=steps, schema_version=schema_version)
