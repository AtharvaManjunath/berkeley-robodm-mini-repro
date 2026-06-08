from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robot_bench.converters.base import ConversionResult, Converter
from robot_bench.episode_schema import Episode
from robot_bench.utils.fs import clean_dir


def _write_value(group: h5py.Group, name: str, value: Any) -> None:
    if isinstance(value, dict):
        sub = group.create_group(name)
        for key, child in value.items():
            _write_value(sub, str(key), child)
        return
    if value is None:
        return
    if isinstance(value, str):
        group.create_dataset(name, data=np.asarray(value, dtype=h5py.string_dtype("utf-8")))
        return
    arr = np.asarray(value)
    group.create_dataset(name, data=arr)


class HDF5Converter(Converter):
    format_name = "hdf5"
    adapter_kind = "style"

    def __init__(self, compression: str | None = None) -> None:
        self.compression = compression

    def convert(self, dataset_name: str, episodes: list[Episode], output_path: Path) -> ConversionResult:
        start = time.perf_counter()
        clean_dir(output_path)
        for ep in episodes:
            ep.validate()
            with h5py.File(output_path / f"{ep.episode_id}.h5", "w") as h5:
                meta = h5.create_group("metadata")
                meta.attrs["json"] = json.dumps(ep.metadata)
                meta.attrs["episode_id"] = ep.episode_id
                meta.attrs["dataset_name"] = ep.dataset_name
                meta.attrs["schema_version"] = ep.schema_version
                steps = h5.create_group("steps")
                obs = steps.create_group("observation")
                action = steps.create_group("action")
                obs_keys = sorted({k for step in ep.steps for k in step.observations})
                for key in obs_keys:
                    vals = [step.observations.get(key) for step in ep.steps]
                    if all(
                        v is not None and isinstance(v, np.ndarray) and v.shape == vals[0].shape
                        for v in vals
                    ):
                        obs.create_dataset(key, data=np.stack(vals), compression=self.compression)
                if all(not isinstance(step.action, dict) for step in ep.steps):
                    action.create_dataset("value", data=np.stack([np.asarray(step.action) for step in ep.steps]), compression=self.compression)
                else:
                    keys = sorted({k for step in ep.steps if isinstance(step.action, dict) for k in step.action})
                    for key in keys:
                        vals = [
                            step.action.get(key) if isinstance(step.action, dict) else None
                            for step in ep.steps
                        ]
                        if not all(v is not None and np.asarray(v).shape == np.asarray(vals[0]).shape for v in vals):
                            continue
                        action.create_dataset(
                            key,
                            data=np.stack([np.asarray(v) for v in vals]),
                            compression=self.compression,
                        )
                rewards = [np.nan if s.reward is None else s.reward for s in ep.steps]
                discounts = [np.nan if s.discount is None else s.discount for s in ep.steps]
                steps.create_dataset("reward", data=np.asarray(rewards, dtype=np.float32))
                steps.create_dataset("discount", data=np.asarray(discounts, dtype=np.float32))
                steps.create_dataset("is_first", data=np.asarray([s.is_first for s in ep.steps], dtype=np.bool_))
                steps.create_dataset("is_last", data=np.asarray([s.is_last for s in ep.steps], dtype=np.bool_))
                steps.create_dataset("is_terminal", data=np.asarray([s.is_terminal for s in ep.steps], dtype=np.bool_))
                lang = [s.language_instruction or "" for s in ep.steps]
                steps.create_dataset("language_instruction", data=np.asarray(lang, dtype=h5py.string_dtype("utf-8")))
        return ConversionResult(
            dataset_name=dataset_name,
            format_name=self.format_name,
            output_path=output_path,
            adapter_kind=self.adapter_kind,
            conversion_seconds=time.perf_counter() - start,
            compression={"hdf5_compression": self.compression or "none"},
        )
