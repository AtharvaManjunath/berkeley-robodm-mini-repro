from __future__ import annotations

import shutil

from robot_bench.utils.subprocesses import run_captured


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def supports_av1() -> bool:
    if not ffmpeg_available():
        return False
    proc = run_captured(["ffmpeg", "-hide_banner", "-encoders"], timeout=10)
    return "libaom-av1" in proc.stdout or "libsvtav1" in proc.stdout


def choose_codec() -> tuple[str, dict[str, str | bool]]:
    if supports_av1():
        return "av1", {"crf": "30", "codec_fallback": False}
    return "h264", {"crf": "23", "codec_fallback": True}
