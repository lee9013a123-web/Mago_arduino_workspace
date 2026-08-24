#!/usr/bin/env python3
"""bucket마다 배포 단위 폴더를 만든다.

한 프로세스는 bucket 하나만 로드하므로 bucket별로 나눈다.  각 폴더는 그 자체로
배포 가능한 단위이며 모델, 짝이 되는 런타임 바이너리, 둘을 묶는 계약을 담는다.

    models/deploy/campp_sv_{bucket}/
        campp_sv_{bucket}.camppmodel   모델 (plan/weights/streaming/dispatch/계약)
        campp_runtime                  런타임 바이너리 (커널 구현체)
        deployment.json                모델·런타임 sha256과 호환 계약
        README.md                      실행 방법

모델과 런타임은 반드시 함께 배포해야 한다.  layer-hybrid 커널 선택은 모델의
dispatch 테이블에 기록되지만 **커널 구현체는 런타임 바이너리에 있다**.  짝이
어긋나면 조용히 다른 커널이 돈다 -- deployment.json의 sha256이 그것을 잡는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from runtime_model_package.format import (  # noqa: E402
    ModelPackageError,
    ModelPackageSectionType as T,
    verify_model_package,
)
from runtime_model_package.per_bucket import build_bucket_package  # noqa: E402

BUCKETS = (98, 298, 498, 998)
DISPATCH_SOURCE = (ROOT / "src/c/profill/optimization/candidates"
                        / "conv_layer_hybrid/conv_layer_hybrid_plan.c")


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


README = """# {model_name}

bucket {bucket} 전용 배포 단위다.  오디오 {seconds}초 ({bucket} frame) 입력을
192차원 화자 임베딩으로 바꾼다.

## 구성

| 파일 | 내용 |
|---|---|
| `{model_name}.camppmodel` | execution plan, packed weights, streaming weights/schedule, kernel dispatch table, frontend/postprocess 계약 |
| `campp_runtime` | 런타임 바이너리.  **v5/v4 NEON 커널 구현체가 여기 있다** |
| `deployment.json` | 모델과 런타임의 sha256, 호환 계약 |

**모델과 런타임은 함께 배포해야 한다.**  모델은 "어느 operator에 어느 커널을
쓸지"를 담고, 런타임은 "그 커널이 무엇인지"를 담는다.  짝이 어긋나면 조용히 다른
커널이 돌기 때문에 `deployment.json`의 sha256으로 대조한다.

## 모델에 들어 있는 최적화

- INT8 양자화
- dense slab (Concat 52개 제거)
- o4i4 weight 재배치 -- 런타임 재패킹 불필요
- E7 operator fusion (1278 -> {operators} op)
- tensor arena 배치
- layer-hybrid kernel dispatch 표: {dispatch}
- weight streaming schedule (RAM 절감용)

## 모델에 없는 것

- v5/v4 NEON 커널 구현체 (런타임 바이너리)
- 화자 검증 임계값 (보정 전)
- 오디오 프론트엔드 구현 (계약만 담김)

## 실행

```bash
./campp_runtime --plan <plan.bin> --weights <weights.bin> \\
    --input <feature.f32> --audio-seconds {seconds} \\
    --warmup 5 --repeat 30 --threads 1
```

현재 런타임은 분리된 plan/weights 파일을 받는다.  `.camppmodel`을 직접 읽는 경로는
C 로더 확장이 끝나야 쓸 수 있다 -- `deployment.json`의 `runtime_reads_model_package`
가 그 상태를 알려준다.

입력은 FBank {bucket}x80 float32다.  파라미터는 모델의 frontend 계약을 따른다
(80 mel, 25 ms/10 ms, dither 0, hamming, time축 CMVN).
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets", type=int, nargs="+", default=list(BUCKETS))
    parser.add_argument("--bundle", type=_path,
                        default=ROOT / "runs/runtime/kernel_optimization/e7/bundle")
    parser.add_argument("--streaming-root", type=_path,
                        default=ROOT / "runs/models/campplus/final_v3/weight_streaming")
    parser.add_argument("--runtime", type=_path,
                        default=ROOT / "build/profill/weight_streaming_98"
                                     / "campp_runtime_benchmark_final")
    parser.add_argument("--out-root", type=_path, default=ROOT / "models/deploy")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    try:
        rows = []
        for bucket in args.buckets:
            plan = args.bundle / f"execution_plans/plan_{bucket}.bin"
            weights = args.bundle / "weights.bin"
            manifest = args.bundle / "manifest.json"
            stream_dir = args.streaming_root / str(bucket)
            sw = stream_dir / f"weights_{bucket}.bin"
            sc = stream_dir / f"weight_schedule_{bucket}.bin"
            missing = [p for p in (plan, weights, manifest) if not p.is_file()]
            if missing:
                raise ModelPackageError(f"missing artifact: {missing[0]}")
            if args.preflight_only:
                rows.append({
                    "bucket": bucket,
                    "streaming_weights": sw.is_file(),
                    "streaming_schedule": sc.is_file(),
                    "dispatch_source": DISPATCH_SOURCE.is_file(),
                    "runtime": args.runtime.is_file(),
                })
                continue

            out_dir = args.out_root / f"campp_sv_{bucket}"
            model_path = out_dir / f"campp_sv_{bucket}.camppmodel"
            if model_path.exists() and not args.force:
                raise ModelPackageError(f"output exists: {model_path}")
            package, report = build_bucket_package(
                bucket_frames=bucket, plan_path=plan, weights_path=weights,
                source_manifest_path=manifest,
                streaming_weights_path=sw if sw.is_file() else None,
                streaming_schedule_path=sc if sc.is_file() else None,
                dispatch_source_path=DISPATCH_SOURCE)
            out_dir.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(package)

            runtime_sha = None
            if args.runtime.is_file():
                shutil.copy2(args.runtime, out_dir / "campp_runtime")
                runtime_sha = _sha256(out_dir / "campp_runtime")

            loaded = verify_model_package(model_path)
            metadata = loaded.json_section(T.MODEL_METADATA_JSON)
            deployment = {
                "schema_version": 1,
                "model_name": report["model_name"],
                "bucket_frames": bucket,
                "audio_seconds": metadata.get("audio_seconds"),
                "model_file": model_path.name,
                "model_sha256": report["package_sha256"],
                "model_size_bytes": report["package_size_bytes"],
                "runtime_file": "campp_runtime" if runtime_sha else None,
                "runtime_sha256": runtime_sha,
                "optimization_suite": report["optimization_suite"],
                "optimization_kernels_in_model": False,
                # C 로더가 .camppmodel을 직접 읽는 경로는 아직 없다.
                "runtime_reads_model_package": False,
                "contract": (
                    "모델과 런타임은 함께 배포한다.  모델은 커널 선택을, 런타임은 "
                    "커널 구현체를 담는다.  짝이 어긋나면 다른 커널이 조용히 돈다."
                ),
                "optional_sections": report["optional_sections"],
                "kernel_dispatch_summary": report["kernel_dispatch_summary"],
                "section_bytes": report["section_bytes"],
                "source": report["source"],
            }
            (out_dir / "deployment.json").write_text(
                json.dumps(deployment, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8", newline="\n")
            (out_dir / "README.md").write_text(README.format(
                model_name=report["model_name"], bucket=bucket,
                seconds=metadata.get("audio_seconds"),
                operators=metadata.get("operator_count"),
                dispatch=json.dumps(report["kernel_dispatch_summary"],
                                    ensure_ascii=False)),
                encoding="utf-8", newline="\n")
            rows.append({
                "bucket": bucket,
                "dir": str(out_dir.relative_to(ROOT)),
                "model_bytes": report["package_size_bytes"],
                "sections": report["section_bytes"],
                "dispatch": report["kernel_dispatch_summary"],
            })

        print(json.dumps({"ready": True, "buckets": rows},
                         ensure_ascii=False, indent=2))
        return 0
    except ModelPackageError as error:
        print(f"deployment bundle failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
