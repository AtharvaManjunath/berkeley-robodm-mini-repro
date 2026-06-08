from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robot_bench.converters.base import ConversionResult, Converter
from robot_bench.converters.video_utils import write_video
from robot_bench.episode_schema import Episode, flatten_numeric, is_image_array
from robot_bench.utils.ffmpeg import choose_codec
from robot_bench.utils.fs import clean_dir


class LeRobotStyleConverter(Converter):
    format_name = "lerobot"
    adapter_kind = "style"

    def convert(self, dataset_name: str, episodes: list[Episode], output_path: Path) -> ConversionResult:
        start = time.perf_counter()
        clean_dir(output_path)
        (output_path / "videos").mkdir()
        (output_path / "data").mkdir()
        codec, settings = choose_codec()
        final_codec = codec
        codec_fallback = bool(settings.get("codec_fallback", False))
        manifest: list[dict[str, Any]] = []
        for ep in episodes:
            ep.validate()
            rows: list[dict[str, Any]] = []
            image_keys = sorted(
                {
                    k
                    for step in ep.steps
                    for k, v in step.observations.items()
                    if isinstance(v, np.ndarray) and v.dtype == np.uint8 and v.ndim == 3
                }
            )
            video_files: dict[str, str] = {}
            for key in image_keys:
                frames = np.stack([step.observations[key] for step in ep.steps])
                stream_codec, stream_fallback = write_video(output_path / "videos" / f"{ep.episode_id}_{key}.mp4", frames, codec)
                final_codec = stream_codec
                codec_fallback = codec_fallback or stream_fallback
                video_files[key] = f"videos/{ep.episode_id}_{key}.{ 'mp4' if stream_codec != 'npy-video-fallback' else 'npy'}"
            for t, step in enumerate(ep.steps):
                row: dict[str, Any] = {
                    "episode_id": ep.episode_id,
                    "step_index": t,
                    "action": flatten_numeric(step.action).tolist(),
                    "reward": step.reward,
                    "discount": step.discount,
                    "is_first": step.is_first,
                    "is_last": step.is_last,
                    "is_terminal": step.is_terminal,
                    "language_instruction": step.language_instruction,
                }
                for key, value in step.observations.items():
                    if not is_image_array(value):
                        row[f"observation.{key}"] = flatten_numeric(value).tolist()
                rows.append(row)
            pd.DataFrame(rows).to_parquet(output_path / "data" / f"{ep.episode_id}.parquet", index=False)
            (output_path / f"{ep.episode_id}.json").write_text(
                json.dumps({"metadata": ep.metadata, "video_files": video_files, "num_steps": len(ep.steps)}, indent=2),
                encoding="utf-8",
            )
            manifest.append({"episode_id": ep.episode_id, "video_files": video_files})
        (output_path / "meta.json").write_text(
            json.dumps(
                {
                    "dataset_name": dataset_name,
                    "adapter_kind": "style",
                    "format_note": "Local LeRobot-style Parquet+MP4 layout; official API is not required.",
                    "codec": final_codec,
                    "codec_fallback": codec_fallback,
                    "episodes": manifest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return ConversionResult(
            dataset_name,
            self.format_name,
            output_path,
            self.adapter_kind,
            time.perf_counter() - start,
            codec=final_codec,
            compression={"video": final_codec, "codec_fallback": codec_fallback},
            notes="local LeRobot-style Parquet/video implementation",
        )
