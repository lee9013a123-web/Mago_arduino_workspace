#!/usr/bin/env python3
"""4개 bucket(98/298/498/998)을 `.camppmodel` 하나로 묶는다.

weights는 bucket이 공유하므로 파일 하나에 한 번만 들어간다.  bucket별로 파일을
따로 만들면 7.57 MB weights가 4번 복제된다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from runtime_model_package.format import (  # noqa: E402
    ModelPackageError,
    verify_model_package,
)
from runtime_model_package.multibucket import (  # noqa: E402
    DEFAULT_BUCKETS,
    build_multibucket_package,
)


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def main() -> int:
    bundle = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
    out_dir = ROOT / "models/runtime/campp_sv_multibucket"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=_path, default=bundle)
    parser.add_argument("--buckets", type=int, nargs="+",
                        default=list(DEFAULT_BUCKETS))
    parser.add_argument("--output", type=_path,
                        default=out_dir / "campp_sv_multibucket.camppmodel")
    parser.add_argument("--report", type=_path,
                        default=out_dir / "campp_sv_multibucket.json")
    parser.add_argument("--model-name", default="campp_sv_multibucket")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    try:
        plan_paths = {
            bucket: args.bundle / f"execution_plans/plan_{bucket}.bin"
            for bucket in args.buckets
        }
        weights = args.bundle / "weights.bin"
        manifest = args.bundle / "manifest.json"
        required = [*plan_paths.values(), weights, manifest]
        missing = [p for p in required if not p.is_file()]
        if missing:
            raise ModelPackageError(f"required artifact missing: {missing[0]}")
        if args.preflight_only:
            print(json.dumps({
                "ready": True,
                "buckets": sorted(args.buckets),
                "weights_bytes": weights.stat().st_size,
                "plan_bytes": {str(b): p.stat().st_size
                               for b, p in sorted(plan_paths.items())},
                "output": str(args.output.relative_to(ROOT)),
            }, ensure_ascii=False, indent=2))
            return 0
        if args.output.exists() and not args.force:
            raise ModelPackageError("output exists; use --force")

        package, report = build_multibucket_package(
            plan_paths=plan_paths, weights_path=weights,
            source_manifest_path=manifest,
            model_name=args.model_name, buckets=args.buckets)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(package)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n")

        # 디스크에 쓴 파일을 다시 읽어 확인한다.
        loaded = verify_model_package(args.output)
        if loaded.buckets != tuple(sorted(args.buckets)):
            raise ModelPackageError("on-disk package bucket mismatch")
        for bucket, path in sorted(plan_paths.items()):
            if loaded.plan_for(bucket) != path.read_bytes():
                raise ModelPackageError(
                    f"on-disk plan mismatch for bucket {bucket}")
        print(json.dumps({
            "ready": True,
            "output": str(args.output.relative_to(ROOT)),
            "buckets": report["buckets"],
            "package_size_bytes": report["package_size_bytes"],
            "package_sha256": report["package_sha256"],
            "size_if_separate_files_bytes":
                report["size_if_separate_files_bytes"],
            "size_saved_bytes": report["size_saved_bytes"],
        }, ensure_ascii=False, indent=2))
        return 0
    except ModelPackageError as error:
        print(f"multibucket package failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
