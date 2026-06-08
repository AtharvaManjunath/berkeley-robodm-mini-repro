from __future__ import annotations

import pytest

from robot_bench.dataset_registry import generate_synthetic_dataset


@pytest.fixture()
def tiny_episodes():
    return generate_synthetic_dataset("tiny_test", "tiny", seed=7)


@pytest.fixture()
def multimodal_episodes():
    return generate_synthetic_dataset("multi_test", "multimodal", seed=9)
