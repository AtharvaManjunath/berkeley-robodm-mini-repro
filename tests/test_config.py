from __future__ import annotations

from pathlib import Path

from robot_bench.config import load_config


def test_config_parsing() -> None:
    cfg = load_config(Path("configs/synthetic.yaml"))
    assert cfg.seed == 12345
    assert cfg.profile("small").batch_size == 2
    assert len(cfg.datasets) == 2
