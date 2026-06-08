from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from robot_bench.converters.video_utils import read_video
from robot_bench.episode_schema import Episode, Step
from robot_bench.loaders.base import FormatLoader


class RoboDMStyleLoader(FormatLoader):
    format_name = "robodm"
    adapter_kind = "fallback"

    def list_episode_ids(self, path: Path) -> list[str]:
        return sorted(p.stem for p in path.glob("*.rdm"))

    def load_episode(self, path: Path, episode_id: str) -> Episode:
        with zipfile.ZipFile(path / f"{episode_id}.rdm", "r") as zf:
            header = json.loads(zf.read("header.json").decode("utf-8"))
            numeric = np.load(io.BytesIO(zf.read("numeric.npz")))
            with tempfile.TemporaryDirectory() as td:
                tdp = Path(td)
                videos = {}
                for key, rel in header.get("streams", {}).items():
                    target = tdp / rel
                    target.write_bytes(zf.read(f"video/{rel}"))
                    videos[key] = read_video(target)
                n = int(header["num_steps"])
                steps: list[Step] = []
                langs = header.get("language_instruction", [""] * n)
                for i in range(n):
                    obs = {key: frames[i] for key, frames in videos.items()}
                    for arr_key in numeric.files:
                        if arr_key.startswith("observation__"):
                            obs[arr_key.removeprefix("observation__")] = np.asarray(numeric[arr_key][i])
                    steps.append(
                        Step(
                            observations=obs,
                            action=np.asarray(numeric["action"][i]),
                            reward=None if np.isnan(numeric["reward"][i]) else float(numeric["reward"][i]),
                            discount=None if np.isnan(numeric["discount"][i]) else float(numeric["discount"][i]),
                            is_first=bool(numeric["is_first"][i]),
                            is_last=bool(numeric["is_last"][i]),
                            is_terminal=bool(numeric["is_terminal"][i]),
                            language_instruction=str(langs[i]) or None,
                        )
                    )
        return Episode(episode_id=episode_id, dataset_name=header["dataset_name"], metadata=header.get("metadata", {}), steps=steps)
