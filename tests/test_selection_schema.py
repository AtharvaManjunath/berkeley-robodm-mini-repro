from __future__ import annotations

import pytest

from robot_bench.config import ProfileConfig
from robot_bench.dataset_registry import deterministic_select, generate_synthetic_dataset
from robot_bench.episode_schema import Episode


def test_deterministic_episode_selection() -> None:
    episodes = generate_synthetic_dataset("sel", "multimodal", seed=5)
    profile = ProfileConfig(episodes_per_dataset=2, max_steps_per_episode=6)
    a = deterministic_select(episodes, "sel", profile, seed=11)
    b = deterministic_select(episodes, "sel", profile, seed=11)
    assert [ep.episode_id for ep in a] == [ep.episode_id for ep in b]
    assert all(len(ep.steps) <= 6 for ep in a)


def test_schema_validation_rejects_empty() -> None:
    with pytest.raises(ValueError):
        Episode(episode_id="bad", dataset_name="x", steps=[]).validate()
