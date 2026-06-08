from __future__ import annotations

from typer.testing import CliRunner

from robot_bench.cli import app


def test_cli_smoke(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run-all",
            "--config",
            "configs/synthetic.yaml",
            "--profile",
            "small",
            "--formats",
            "rlds,hdf5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "wrote" in result.output
