#!/usr/bin/env python3
"""Compare every retained tensor from final V2 and layer-hybrid V3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
FEATURES = (
    ROOT / "benchmarks/campplus/features/multi__speaker_0000__98.f32",
    ROOT / "benchmarks/campplus/features/multi__speaker_0005__98.f32",
    ROOT / "benchmarks/campplus/features/multi__speaker_0006__98.f32",
)


class RetainedValidationError(RuntimeError):
    """Retained tensor validation could not complete."""


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_dump(
    binary: Path, plan: Path, weights: Path, feature: Path, prefix: Path,
) -> dict[str, Any]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(binary), str(plan), str(weights), str(feature), str(prefix)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise RetainedValidationError(
            f"tensor dump failed: {completed.stderr.strip()}"
        )
    payload = prefix.with_suffix(".bin")
    index = prefix.with_suffix(".json")
    if not payload.is_file() or not index.is_file():
        raise RetainedValidationError(f"incomplete tensor dump: {prefix}")
    document = json.loads(index.read_text(encoding="utf-8"))
    tensors = document.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise RetainedValidationError(f"empty tensor index: {index}")
    return {
        "payload": payload,
        "index": index,
        "tensors": tensors,
        "tensor_count": len(tensors),
        "payload_bytes": payload.stat().st_size,
        "payload_sha256": _sha256(payload),
    }


def main(argv: Sequence[str] | None = None) -> int:
    bundle = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v2-binary", type=_path,
        default=ROOT / "build/profill/final_v2_aggressive/campp_reference_dump_final",
    )
    parser.add_argument(
        "--v3-binary", type=_path,
        default=ROOT / "build/profill/final_v3_hybrid/campp_reference_dump_final",
    )
    parser.add_argument(
        "--plan", type=_path,
        default=bundle / "execution_plans/plan_98.bin",
    )
    parser.add_argument("--weights", type=_path, default=bundle / "weights.bin")
    parser.add_argument("--features", type=_path, nargs="+", default=list(FEATURES))
    parser.add_argument(
        "--runs-dir", type=_path,
        default=ROOT / "runs/profiling/e7_98/final_v3_vs_v2/retained",
    )
    parser.add_argument(
        "--output", type=_path,
        default=(
            ROOT / "results/profiling/e7_98/final_v3_vs_v2_retained.json"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        required = [
            args.v2_binary, args.v3_binary, args.plan, args.weights,
            *args.features,
        ]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise RetainedValidationError(
                "missing files:\n  " + "\n  ".join(str(path) for path in missing)
            )
        if len(args.features) != 3:
            raise RetainedValidationError("exactly three fixed inputs are required")
        if args.preflight_only:
            print(json.dumps({
                "ready": True,
                "v2_binary": str(args.v2_binary),
                "v3_binary": str(args.v3_binary),
                "features": [str(path) for path in args.features],
            }, ensure_ascii=False, indent=2))
            return 0
        if args.output.exists() and not args.force:
            raise RetainedValidationError("output exists (use --force)")

        comparisons = []
        all_match = True
        for feature in args.features:
            v2 = _run_dump(
                args.v2_binary, args.plan, args.weights, feature,
                args.runs_dir / "v2" / feature.stem,
            )
            v3 = _run_dump(
                args.v3_binary, args.plan, args.weights, feature,
                args.runs_dir / "v3" / feature.stem,
            )
            index_match = v2["tensors"] == v3["tensors"]
            payload_match = v2["payload_sha256"] == v3["payload_sha256"]
            matched = index_match and payload_match
            all_match = all_match and matched
            comparisons.append({
                "input": feature.name,
                "tensor_count": v2["tensor_count"],
                "payload_bytes": v2["payload_bytes"],
                "v2_sha256": v2["payload_sha256"],
                "v3_sha256": v3["payload_sha256"],
                "index_identical": index_match,
                "all_tensor_bytes_bitwise_identical": payload_match,
                "passed": matched,
            })
        report = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "backend": "cpu_aarch64_o4i4_final",
            "bucket_frames": 98,
            "fixed_input_count": len(args.features),
            "reference": "final_v2_aggressive",
            "candidate": "final_v3_layer_hybrid",
            "tolerance": {"kind": "bitwise", "atol": 0.0, "rtol": 0.0},
            "comparisons": comparisons,
            "all_retained_tensors_bitwise_identical": all_match,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        print(f"retained tensor validation: {'PASS' if all_match else 'FAIL'}")
        return 0 if all_match else 3
    except (
        RetainedValidationError, OSError, ValueError, json.JSONDecodeError
    ) as exc:
        print(f"retained tensor validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
