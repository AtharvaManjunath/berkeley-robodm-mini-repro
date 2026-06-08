from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from robot_bench.config import BenchmarkConfig, DatasetConfig, ProfileConfig
from robot_bench.episode_schema import Episode, Step
from robot_bench.utils.fs import ensure_dir

LOGGER = logging.getLogger(__name__)


def generate_synthetic_dataset(
    name: str,
    variant: str = "tiny",
    seed: int = 12345,
    max_steps: int | None = None,
) -> list[Episode]:
    stable_name_hash = int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + stable_name_hash % 10000)
    count = 3 if variant == "tiny" else 4
    episodes: list[Episode] = []
    for ep_idx in range(count):
        length = 8 if variant == "tiny" else int(rng.integers(5, 12))
        if max_steps is not None:
            length = min(length, max_steps)
        steps: list[Step] = []
        for t in range(length):
            obs: dict[str, object] = {
                "image": rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8),
                "state": rng.normal(size=(7,)).astype(np.float32),
            }
            if variant == "multimodal":
                obs["wrist_image"] = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
                obs["depth"] = rng.integers(0, 256, size=(24, 24, 1), dtype=np.uint8)
            language = f"synthetic instruction {ep_idx}" if not (variant == "multimodal" and ep_idx == 1) else None
            action: object = rng.normal(size=(4,)).astype(np.float32)
            if variant == "multimodal":
                action = {
                    "arm": rng.normal(size=(4,)).astype(np.float32),
                    "gripper": rng.normal(size=(1,)).astype(np.float32),
                }
            steps.append(
                Step(
                    observations=obs,
                    action=action,
                    reward=float(rng.normal()) if variant != "missing_reward" else None,
                    discount=1.0 if variant != "missing_reward" else None,
                    is_first=t == 0,
                    is_last=t == length - 1,
                    is_terminal=t == length - 1,
                    language_instruction=language,
                )
            )
        episodes.append(
            Episode(
                episode_id=f"{name}_{ep_idx:04d}",
                dataset_name=name,
                metadata={"synthetic": True, "variant": variant},
                steps=steps,
            )
        )
    if variant == "edge":
        episodes.append(Episode(episode_id=f"{name}_malformed", dataset_name=name, steps=[]))
    return episodes


def deterministic_select(
    episodes: list[Episode],
    dataset_name: str,
    profile: ProfileConfig,
    seed: int,
) -> list[Episode]:
    valid = [ep for ep in episodes if ep.steps]
    order_rng = np.random.default_rng(seed + sum(ord(c) for c in dataset_name))
    indices = np.arange(len(valid))
    order_rng.shuffle(indices)
    if profile.episodes_per_dataset is not None:
        indices = indices[: profile.episodes_per_dataset]
    selected: list[Episode] = []
    for idx in indices:
        ep = valid[int(idx)]
        if profile.max_steps_per_episode is not None and len(ep.steps) > profile.max_steps_per_episode:
            ep = Episode(
                episode_id=ep.episode_id,
                dataset_name=ep.dataset_name,
                metadata=dict(ep.metadata),
                steps=ep.steps[: profile.max_steps_per_episode],
                schema_version=ep.schema_version,
            )
        selected.append(ep)
    return selected


def save_selection_manifest(
    config: BenchmarkConfig,
    profile_name: str,
    selections: dict[str, list[Episode]],
) -> Path:
    out_dir = ensure_dir(config.artifact_dir / "manifests")
    path = out_dir / f"episode_selection_{profile_name}.json"
    payload = {
        "profile": profile_name,
        "seed": config.seed,
        "datasets": {
            name: [{"episode_id": ep.episode_id, "num_steps": len(ep.steps)} for ep in eps]
            for name, eps in selections.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_dataset(
    ds: DatasetConfig,
    config: BenchmarkConfig,
    profile: ProfileConfig,
    profile_name: str,
) -> list[Episode]:
    if not ds.enabled:
        return []
    if ds.source == "synthetic":
        variant = ds.notes or "tiny"
        episodes = generate_synthetic_dataset(ds.name, variant=variant, seed=config.seed, max_steps=profile.max_steps_per_episode)
        return deterministic_select(episodes, ds.name, profile, config.seed)
    if ds.known_size_gb is not None and ds.known_size_gb > config.max_download_gb:
        raise RuntimeError(
            f"{ds.name} known size {ds.known_size_gb} GB exceeds max_download_gb={config.max_download_gb}"
        )
    if ds.source == "local_rlds":
        if ds.local_path is None:
            raise RuntimeError(f"{ds.name} source=local_rlds requires local_path")
        if not ds.local_path.exists():
            raise RuntimeError(f"{ds.name} local_rlds path does not exist: {ds.local_path}")
        from robot_bench.loaders.rlds_loader import RLDSStyleLoader

        loader = RLDSStyleLoader()
        episodes = [loader.load_episode(ds.local_path, ep_id) for ep_id in loader.list_episode_ids(ds.local_path)]
        return deterministic_select(episodes, ds.name, profile, config.seed)
    if ds.source == "tfds":
        if not config.allow_downloads:
            raise RuntimeError(f"{ds.name} requires --allow-downloads for TFDS public-data access")
        try:
            from robot_bench.loaders.rlds_loader import load_tfds_rlds_subset
        except Exception as exc:
            raise RuntimeError(f"TensorFlow/TFDS loader unavailable: {exc}") from exc
        return load_tfds_rlds_subset(ds, config, profile, profile_name)
    if ds.source == "huggingface":
        if not config.allow_downloads:
            raise RuntimeError(f"{ds.name} requires --allow-downloads for Hugging Face access")
        raise RuntimeError("Hugging Face public dataset loading is configured but not implemented in this harness yet")
    raise RuntimeError(f"unsupported dataset source: {ds.source}")
