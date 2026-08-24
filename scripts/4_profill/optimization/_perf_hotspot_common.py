"""perf hotspot 스크립트가 공유하는 source-line 파싱과 실행 도우미."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterator, Sequence


PERF_LINE = re.compile(
    r"^\s*(?P<period>\d+)\s+"
    r"(?P<ip>(?:0x)?[0-9a-fA-F]+)\s+"
    r"(?P<symbol>\S+)\s+"
    r"(?P<source>.*):(?P<line>\d+)\s*$"
)
PERF_HEADER_LINE = re.compile(
    r"^\s*(?P<period>\d+)\s+"
    r"(?P<ip>(?:0x)?[0-9a-fA-F]+)\s+"
    r"(?P<symbol>\S+)\s*$"
)
PERF_SRCLINE_LINE = re.compile(r"^\s+(?P<source>\S.*):(?P<line>\d+)\s*$")


def function_span(lines: Sequence[str], signature: str) -> tuple[int, int]:
    start_index = next(
        (index for index, line in enumerate(lines) if signature in line), None
    )
    if start_index is None:
        raise ValueError(f"source signature not found: {signature}")
    depth = 0
    opened = False
    for index in range(start_index, len(lines)):
        depth += lines[index].count("{")
        opened = opened or "{" in lines[index]
        depth -= lines[index].count("}")
        if opened and depth == 0:
            return start_index + 1, index + 1
    raise ValueError(f"unterminated C function: {signature}")


def find_line(lines: Sequence[str], text: str, *, start: int = 1) -> int:
    for index in range(start - 1, len(lines)):
        if text in lines[index]:
            return index + 1
    raise ValueError(f"source marker not found: {text}")


def inside(line: int, span: tuple[int, int]) -> bool:
    return span[0] <= line <= span[1]


def iter_perf_samples(text: str) -> tuple[list[dict[str, Any]], int]:
    samples: list[dict[str, Any]] = []
    malformed = 0
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        if not raw_line.strip():
            index += 1
            continue
        match = PERF_LINE.match(raw_line)
        if match is not None:
            samples.append(
                {
                    "period": int(match.group("period")),
                    "ip": match.group("ip"),
                    "symbol": match.group("symbol"),
                    "source": match.group("source"),
                    "line": int(match.group("line")),
                }
            )
            index += 1
            continue
        header = PERF_HEADER_LINE.match(raw_line)
        if header is None:
            malformed += 1
            index += 1
            continue
        srcline = (
            PERF_SRCLINE_LINE.match(lines[index + 1])
            if index + 1 < len(lines)
            else None
        )
        samples.append(
            {
                "period": int(header.group("period")),
                "ip": header.group("ip"),
                "symbol": header.group("symbol"),
                "source": srcline.group("source") if srcline else "",
                "line": int(srcline.group("line")) if srcline else 0,
            }
        )
        index += 2 if srcline else 1
    return samples, malformed


def pin_cpu_zero() -> None:
    os.sched_setaffinity(0, {0})


def run_command(
    command: Sequence[str], *, root: Path, check: bool = True
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        list(command),
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        preexec_fn=pin_cpu_zero if os.name == "posix" else None,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed
