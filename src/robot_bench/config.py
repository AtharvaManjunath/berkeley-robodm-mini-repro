from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    name: str
    source: Literal["synthetic", "tfds", "local_rlds", "huggingface"] = "tfds"
    tfds_name: str | None = None
    local_path: Path | None = None
    hf_repo: str | None = None
    enabled: bool = True
    fallback: str | None = None
    known_size_gb: float | None = None
    notes: str = ""


class ProfileConfig(BaseModel):
    episodes_per_dataset: int | None = 32
    max_steps_per_episode: int | None = 128
    batch_size: int = 8
    num_batches: int = 50
    warmup_batches: int = 5
    workers: int = 4


class BenchmarkConfig(BaseModel):
    seed: int = 12345
    artifact_dir: Path = Path("artifacts")
    cache_dir: Path | None = None
    memory_interval_sec: float = 0.05
    allow_fallbacks: bool = False
    allow_downloads: bool = False
    max_download_gb: float = 25.0
    include_backward: bool = False
    profiles: dict[str, ProfileConfig] = Field(default_factory=dict)
    datasets: list[DatasetConfig]

    def profile(self, name: str) -> ProfileConfig:
        if name == "full":
            base = ProfileConfig(
                episodes_per_dataset=None,
                max_steps_per_episode=None,
                batch_size=8,
                num_batches=200,
                warmup_batches=5,
                workers=4,
            )
            return self.profiles.get(name, base)
        if name in self.profiles:
            return self.profiles[name]
        return ProfileConfig()


def load_config(path: Path) -> BenchmarkConfig:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return BenchmarkConfig.model_validate(data)
