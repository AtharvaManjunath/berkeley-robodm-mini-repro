from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"


ArrayLike = np.ndarray | float | int | bool | str | list[Any] | dict[str, Any] | None


@dataclass(slots=True)
class Step:
    observations: dict[str, Any]
    action: Any
    reward: float | None = None
    discount: float | None = None
    is_first: bool = False
    is_last: bool = False
    is_terminal: bool = False
    language_instruction: str | None = None


@dataclass(slots=True)
class Episode:
    episode_id: str
    dataset_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id is required")
        if not self.dataset_name:
            raise ValueError("dataset_name is required")
        if not self.steps:
            raise ValueError(f"episode {self.episode_id} has no steps")
        for idx, step in enumerate(self.steps):
            if not isinstance(step.observations, dict):
                raise ValueError(f"step {idx} observations must be a dict")
            if step.action is None:
                raise ValueError(f"step {idx} action is required")


def is_image_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim in {3, 4}


def flatten_numeric(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        pieces = [flatten_numeric(v).reshape(-1) for _, v in sorted(value.items()) if v is not None]
        return np.concatenate(pieces).astype(np.float32) if pieces else np.zeros((0,), dtype=np.float32)
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) or value.dtype == np.bool_:
            return value.astype(np.float32).reshape(-1)
        return np.zeros((0,), dtype=np.float32)
    if isinstance(value, (int, float, bool, np.number)):
        return np.asarray([value], dtype=np.float32)
    if isinstance(value, list):
        try:
            return np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return np.zeros((0,), dtype=np.float32)
    return np.zeros((0,), dtype=np.float32)


def episode_to_serializable(ep: Episode) -> dict[str, Any]:
    return {
        "episode_id": ep.episode_id,
        "dataset_name": ep.dataset_name,
        "metadata": ep.metadata,
        "schema_version": ep.schema_version,
        "num_steps": len(ep.steps),
    }
