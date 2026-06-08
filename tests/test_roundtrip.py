from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_bench.converters.hdf5_converter import HDF5Converter
from robot_bench.converters.lerobot_style_converter import LeRobotStyleConverter
from robot_bench.converters.rlds_converter import RLDSStyleConverter
from robot_bench.converters.robodm_style_converter import RoboDMStyleConverter
from robot_bench.loaders.hdf5_loader import HDF5Loader
from robot_bench.loaders.lerobot_style_loader import LeRobotStyleLoader
from robot_bench.loaders.rlds_loader import RLDSStyleLoader
from robot_bench.loaders.robodm_style_loader import RoboDMStyleLoader


@pytest.mark.parametrize(
    ("converter", "loader", "lossless"),
    [
        (HDF5Converter(), HDF5Loader(), True),
        (RLDSStyleConverter(), RLDSStyleLoader(), True),
        (LeRobotStyleConverter(), LeRobotStyleLoader(), False),
        (RoboDMStyleConverter(), RoboDMStyleLoader(), False),
    ],
)
def test_roundtrip_formats(tmp_path: Path, tiny_episodes, converter, loader, lossless: bool) -> None:
    out = tmp_path / converter.format_name
    result = converter.convert("tiny_test", tiny_episodes, out)
    assert result.status == "ok"
    ids = loader.list_episode_ids(out)
    assert ids
    loaded = loader.load_episode(out, ids[0])
    assert len(loaded.steps) == len(tiny_episodes[0].steps)
    assert loaded.steps[0].observations["image"].shape == tiny_episodes[0].steps[0].observations["image"].shape
    if lossless:
        np.testing.assert_array_equal(
            loaded.steps[0].observations["state"],
            tiny_episodes[0].steps[0].observations["state"],
        )
        np.testing.assert_array_equal(loaded.steps[0].action, tiny_episodes[0].steps[0].action)
    else:
        assert loaded.steps[0].observations["image"].dtype == np.uint8
