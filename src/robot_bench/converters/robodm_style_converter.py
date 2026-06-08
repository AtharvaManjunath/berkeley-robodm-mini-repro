from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from robot_bench.converters.base import ConversionResult, Converter
from robot_bench.converters.video_utils import write_video
from robot_bench.episode_schema import Episode, flatten_numeric, is_image_array
from robot_bench.utils.ffmpeg import choose_codec
from robot_bench.utils.fs import clean_dir


class RoboDMStyleConverter(Converter):
    format_name = "robodm"
    adapter_kind = "fallback"

    def convert(self, dataset_name: str, episodes: list[Episode], output_path: Path) -> ConversionResult:
        start = time.perf_counter()
        clean_dir(output_path)
        codec, settings = choose_codec()
        final_codec = codec
        codec_fallback = bool(settings.get("codec_fallback", False))
        manifest = []
        for ep in episodes:
            ep.validate()
            tmp_dir = output_path / f".{ep.episode_id}_parts"
            tmp_dir.mkdir()
            image_keys = sorted({k for step in ep.steps for k, v in step.observations.items() if is_image_array(v)})
            streams: dict[str, str] = {}
            numeric: dict[str, np.ndarray] = {
                "action": np.stack([flatten_numeric(s.action) for s in ep.steps]),
                "reward": np.asarray([np.nan if s.reward is None else s.reward for s in ep.steps], dtype=np.float32),
                "discount": np.asarray([np.nan if s.discount is None else s.discount for s in ep.steps], dtype=np.float32),
                "is_first": np.asarray([s.is_first for s in ep.steps], dtype=np.bool_),
                "is_last": np.asarray([s.is_last for s in ep.steps], dtype=np.bool_),
                "is_terminal": np.asarray([s.is_terminal for s in ep.steps], dtype=np.bool_),
            }
            for key in sorted({k for s in ep.steps for k, v in s.observations.items() if not is_image_array(v)}):
                numeric[f"observation__{key}"] = np.stack([flatten_numeric(s.observations.get(key)) for s in ep.steps])
            for key in image_keys:
                frames = np.stack([s.observations[key] for s in ep.steps])
                stream_codec, stream_fallback = write_video(tmp_dir / f"{key}.mp4", frames, codec)
                final_codec = stream_codec
                codec_fallback = codec_fallback or stream_fallback
                streams[key] = f"{key}.{'mp4' if stream_codec != 'npy-video-fallback' else 'npy'}"
            npz_buf = io.BytesIO()
            np.savez(npz_buf, **numeric)
            header = {
                "magic": "RDMSTYLE1",
                "dataset_name": dataset_name,
                "episode_id": ep.episode_id,
                "metadata": ep.metadata,
                "num_steps": len(ep.steps),
                "codec": final_codec,
                "adapter_kind": "fallback",
                "format_note": "Robo-DM-style fallback; not official Robo-DM.",
                "streams": streams,
                "language_instruction": [s.language_instruction or "" for s in ep.steps],
            }
            with zipfile.ZipFile(output_path / f"{ep.episode_id}.rdm", "w", compression=zipfile.ZIP_STORED) as zf:
                zf.writestr("header.json", json.dumps(header, indent=2))
                zf.writestr("numeric.npz", npz_buf.getvalue())
                for rel in streams.values():
                    zf.write(tmp_dir / rel, arcname=f"video/{rel}")
            for p in tmp_dir.glob("*"):
                p.unlink()
            tmp_dir.rmdir()
            manifest.append(ep.episode_id)
        (output_path / "manifest.json").write_text(json.dumps({"episodes": manifest}, indent=2), encoding="utf-8")
        return ConversionResult(
            dataset_name,
            self.format_name,
            output_path,
            self.adapter_kind,
            time.perf_counter() - start,
            codec=final_codec,
            compression={"video": final_codec, "codec_fallback": codec_fallback, "container": "zip-stored-rdm"},
            notes="Robo-DM-style fallback container; not official Robo-DM",
        )
