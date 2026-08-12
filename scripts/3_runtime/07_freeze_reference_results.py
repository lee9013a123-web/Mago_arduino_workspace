#!/usr/bin/env python3
"""검증된 Runtime 결과를 Phase 4가 참조할 고정 보고서로 생성한다."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SOURCE = ROOT / "src" / "python"


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir", type=_path, default=ROOT / "models" / "compiled" / "reference"
    )
    parser.add_argument(
        "--validation-dir", type=_path, default=ROOT / "results" / "runtime"
    )
    parser.add_argument(
        "--config", type=_path, default=ROOT / "configs" / "runtime" / "reference.json"
    )
    parser.add_argument(
        "--registry",
        type=_path,
        default=ROOT / "src" / "c" / "runtime" / "backends" / "cpu_reference"
        / "reference_backend.c",
    )
    parser.add_argument(
        "--output-dir", type=_path, default=ROOT / "results" / "runtime"
    )
    parser.add_argument(
        "--require-phase4-ready",
        action="store_true",
        help="보고서 생성 후 phase4_ready=false이면 오류 코드 3을 반환한다",
    )
    args = parser.parse_args(argv)

    if str(PYTHON_SOURCE) not in sys.path:
        sys.path.insert(0, str(PYTHON_SOURCE))
    from runtime_bundle_exporter.reporting.reference_result_freezer import (
        FreezeResultError,
        freeze_reference_results,
    )

    try:
        result = freeze_reference_results(
            repository_root=ROOT,
            bundle_dir=args.bundle_dir,
            validation_dir=args.validation_dir,
            config_path=args.config,
            registry_path=args.registry,
            output_dir=args.output_dir,
        )
    except FreezeResultError as exc:
        print(f"결과 고정 실패: {exc}", file=sys.stderr)
        return 2

    print("CAM++ Reference Runtime 기준선 생성")
    print(f"  baseline id: {result['baseline_id']}")
    print(f"  baseline frozen: {'YES' if result['baseline_frozen'] else 'NO'}")
    print(f"  failed buckets: {result['failed_buckets'] or 'none'}")
    print(f"  Phase 4 ready: {'YES' if result['phase4_ready'] else 'NO'}")
    print("  outputs:")
    for path in result["output_paths"].values():
        print(f"    {path}")
    # 기준선 생성 자체와 Phase 4 승인 여부는 별개다. 알려진 실패도 보고서로 고정한다.
    if not result["baseline_frozen"]:
        return 1
    if args.require_phase4_ready and not result["phase4_ready"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
