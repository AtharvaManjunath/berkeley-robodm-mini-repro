from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import psutil


@dataclass(slots=True)
class MemoryMonitor:
    interval_sec: float = 0.05
    peak_rss_bytes: int = 0
    _running: bool = False
    _thread: threading.Thread | None = None

    def __enter__(self) -> MemoryMonitor:
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sample()

    def _sample_loop(self) -> None:
        while self._running:
            self._sample()
            time.sleep(self.interval_sec)

    def _sample(self) -> None:
        proc = psutil.Process()
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                continue
        self.peak_rss_bytes = max(self.peak_rss_bytes, total)

    @property
    def peak_rss_mb(self) -> float:
        return self.peak_rss_bytes / (1024 * 1024)
