from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from robot_bench.converters.base import ConversionResult, Converter
from robot_bench.episode_schema import Episode
from robot_bench.utils.fs import clean_dir


class RLDSStyleConverter(Converter):
    format_name = "rlds"
    adapter_kind = "style"

    def convert(self, dataset_name: str, episodes: list[Episode], output_path: Path) -> ConversionResult:
        start = time.perf_counter()
        clean_dir(output_path)
        manifest = []
        for ep in episodes:
            ep.validate()
            ep_dir = output_path / ep.episode_id
            ep_dir.mkdir(parents=True)
            np.savez_compressed(
                ep_dir / "steps.npz",
                **_episode_arrays(ep),
            )
            (ep_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "episode_id": ep.episode_id,
                        "dataset_name": ep.dataset_name,
                        "metadata": ep.metadata,
                        "schema_version": ep.schema_version,
                        "adapter_note": "Local RLDS-style export; not a canonical TFRecord TFDS builder.",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            manifest.append(ep.episode_id)
        (output_path / "manifest.json").write_text(json.dumps({"episodes": manifest}, indent=2), encoding="utf-8")
        return ConversionResult(dataset_name, self.format_name, output_path, self.adapter_kind, time.perf_counter() - start, notes="local RLDS-style npz fixture")


def _episode_arrays(ep: Episode) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    keys = sorted({k for step in ep.steps for k in step.observations})
    for key in keys:
        vals = [step.observations.get(key) for step in ep.steps]
        if all(
            v is not None and isinstance(v, np.ndarray) and v.shape == vals[0].shape for v in vals
        ):
            arrays[f"observation__{key}"] = np.stack(vals)
    if all(not isinstance(step.action, dict) for step in ep.steps):
        arrays["action__value"] = np.stack([np.asarray(step.action) for step in ep.steps])
    else:
        action_keys = sorted({k for step in ep.steps if isinstance(step.action, dict) for k in step.action})
        for key in action_keys:
            vals = [step.action.get(key) if isinstance(step.action, dict) else None for step in ep.steps]
            if all(v is not None and np.asarray(v).shape == np.asarray(vals[0]).shape for v in vals):
                arrays[f"action__{key}"] = np.stack([np.asarray(v) for v in vals])
    arrays["reward"] = np.asarray([np.nan if s.reward is None else s.reward for s in ep.steps], dtype=np.float32)
    arrays["discount"] = np.asarray([np.nan if s.discount is None else s.discount for s in ep.steps], dtype=np.float32)
    arrays["is_first"] = np.asarray([s.is_first for s in ep.steps], dtype=np.bool_)
    arrays["is_last"] = np.asarray([s.is_last for s in ep.steps], dtype=np.bool_)
    arrays["is_terminal"] = np.asarray([s.is_terminal for s in ep.steps], dtype=np.bool_)
    arrays["language_instruction"] = np.asarray([s.language_instruction or "" for s in ep.steps])
    return arrays
