from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from robot_bench.episode_schema import Episode


class FormatLoader(ABC):
    format_name: str
    adapter_kind: str

    @abstractmethod
    def list_episode_ids(self, path: Path) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def load_episode(self, path: Path, episode_id: str) -> Episode:
        raise NotImplementedError
