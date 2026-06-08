from __future__ import annotations

import subprocess


def run_captured(args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False, timeout=timeout)
