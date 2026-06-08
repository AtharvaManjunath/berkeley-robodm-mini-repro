from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from robot_bench.episode_schema import Episode


@dataclass(slots=True)
class ConversionResult:
    dataset_name: str
    format_name: str
    output_path: Path
    adapter_kind: str
    conversion_seconds: float
    codec: str = ""
    compression: dict[str, object] = field(default_factory=dict)
    status: str = "ok"
    error_type: str = ""
    error_message: str = ""
    notes: str = ""


class Converter(ABC):
    format_name: str
    adapter_kind: str

    @abstractmethod
    def convert(self, dataset_name: str, episodes: list[Episode], output_path: Path) -> ConversionResult:
        raise NotImplementedError
