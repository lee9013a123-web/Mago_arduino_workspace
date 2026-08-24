#!/usr/bin/env python3
"""Print or machine-query a compiler matrix decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]


def _load_decision(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("ready") is not True:
        raise ValueError("compiler matrix decision is not ready")
    if not isinstance(document.get("baseline"), dict) or not isinstance(
        document.get("winner"), dict
    ):
        raise ValueError("compiler matrix decision is incomplete")
    return document


def _print_field(document: dict[str, Any], field: str) -> None:
    if field == "cc":
        print(document.get("cc", "gcc"))
        return
    target_name, value_name = field.split("-", 1)
    target = document[target_name]
    if value_name == "build-dir":
        value = target["build"]["directory"]
    elif value_name == "variant":
        value = target["variant"]
    elif value_name in ("cppflags", "cflags", "ldflags"):
        value = target["flags"][value_name]
    elif value_name == "strip-final":
        value = "1" if target["flags"]["strip_final"] else "0"
    else:
        raise ValueError(f"unsupported field: {field}")
    path = Path(value) if value_name == "build-dir" else None
    print(str(ROOT / path) if path is not None and not path.is_absolute() else value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--decision", type=Path)
    parser.add_argument(
        "--print-field",
        choices=(
            "cc",
            "baseline-build-dir", "baseline-variant",
            "baseline-cppflags", "baseline-cflags", "baseline-ldflags",
            "baseline-strip-final",
            "winner-build-dir", "winner-variant",
            "winner-cppflags", "winner-cflags", "winner-ldflags",
            "winner-strip-final",
        ),
    )
    args = parser.parse_args(argv)
    if args.decision is None and args.run_id is None:
        parser.error("--decision or --run-id is required")
    path = args.decision or (
        ROOT / "results/profiling/e7_98/compiler_matrix" / args.run_id / "decision.json"
    )
    try:
        document = _load_decision(path)
        if args.print_field:
            _print_field(document, args.print_field)
            return 0
        baseline = document["baseline"]
        winner = document["winner"]
        baseline_latency = baseline["aggregate"]["latency"]
        winner_latency = winner["aggregate"]["latency"]
        speedup = float(baseline_latency["p50_ms"]) / float(
            winner_latency["p50_ms"]
        )
        print(f"run: {document['run_id']}")
        print(f"baseline: {baseline['variant']}")
        print(f"winner: {winner['variant']}")
        print(f"p50: {baseline_latency['p50_ms']:.3f} -> "
              f"{winner_latency['p50_ms']:.3f} ms ({speedup:.3f}x)")
        print(f"winner CFLAGS: {winner['flags']['cflags']}")
        print(f"winner LDFLAGS: {winner['flags']['ldflags'] or '(none)'}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"compiler matrix comparison failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
