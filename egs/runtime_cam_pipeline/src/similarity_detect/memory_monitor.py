"""Linux process-tree RSS monitoring for the complete microphone pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Callable


StatusReader = Callable[[int], tuple[int | None, int | None]]
ChildrenReader = Callable[[int], tuple[int, ...]]


def _read_kib_field(path: Path, name: str) -> int | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    prefix = f"{name}:"
    for line in lines:
        if not line.startswith(prefix):
            continue
        fields = line.split()
        if len(fields) >= 2 and fields[1].isdigit():
            return int(fields[1]) * 1024
    return None


def read_linux_process_memory(pid: int) -> tuple[int | None, int | None]:
    """Return `(current RSS, peak RSS)` from `/proc/<pid>/status`."""

    status = Path("/proc") / str(pid) / "status"
    return _read_kib_field(status, "VmRSS"), _read_kib_field(status, "VmHWM")


def read_linux_children(pid: int) -> tuple[int, ...]:
    path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return ()
    if not value:
        return ()
    return tuple(int(item) for item in value.split() if item.isdigit())


def process_tree_pids(
    root_pid: int, children_reader: ChildrenReader = read_linux_children,
) -> tuple[int, ...]:
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        pending.extend(children_reader(pid))
    return tuple(sorted(seen))


@dataclass(frozen=True)
class PipelineMemoryMeasurement:
    pipeline_total_peak_rss_bytes: int | None
    python_host_peak_rss_bytes: int | None
    sampling_interval_ms: float
    sample_count: int
    total_semantics: str = "maximum sampled sum of process-tree RSS"


class ProcessTreeMemoryMonitor:
    """Sample aggregate RSS for one command process and all descendants.

    The aggregate may count shared pages once per process. It is intentionally
    separate from the C runtime's own VmHWM and logical model buffer sizes.
    """

    def __init__(
        self, root_pid: int | None = None, *, interval_seconds: float = 0.01,
        status_reader: StatusReader = read_linux_process_memory,
        children_reader: ChildrenReader = read_linux_children,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("memory sampling interval must be positive")
        self._root_pid = os.getpid() if root_pid is None else root_pid
        self._interval_seconds = interval_seconds
        self._status_reader = status_reader
        self._children_reader = children_reader
        self._peak: int | None = None
        self._sample_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        total = 0
        available = False
        for pid in process_tree_pids(self._root_pid, self._children_reader):
            current, _ = self._status_reader(pid)
            if current is not None:
                total += current
                available = True
        if available:
            self._peak = total if self._peak is None else max(self._peak, total)
        self._sample_count += 1

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("memory monitor is already started")
        self._sample()
        self._thread = threading.Thread(
            target=self._run,
            name="campp-process-tree-memory",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> PipelineMemoryMeasurement:
        if self._thread is None:
            raise RuntimeError("memory monitor was not started")
        self._sample()
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._interval_seconds * 4.0))
        _, python_peak = self._status_reader(self._root_pid)
        return PipelineMemoryMeasurement(
            pipeline_total_peak_rss_bytes=self._peak,
            python_host_peak_rss_bytes=python_peak,
            sampling_interval_ms=self._interval_seconds * 1000.0,
            sample_count=self._sample_count,
        )
