"""bucket 하나를 완전한 배포 단위로 담는 `.camppmodel`.

한 프로세스는 bucket 하나만 로드한다.  그래서 4개를 한 파일에 묶는 대신 bucket마다
파일을 만들고, 필요한 bucket만 기기에 넣는다.  1초 발화만 처리하는 기기라면 98짜리
하나만 배포하면 된다.

담기는 것:

    EXECUTION_PLAN          fusion이 끝난 그래프 (98은 832 op, 나머지는 884)
    PACKED_WEIGHTS          o4i4 재배치된 weight -- 런타임 재패킹 불필요
    STREAMING_WEIGHTS       page 정렬된 windowing용 배치 (선택)
    WEIGHT_STREAM_SCHEDULE  block별 prefetch/evict 표 (선택)
    KERNEL_DISPATCH_TABLE   operator별 커널 선택 (선택)
    *_CONTRACT_JSON         frontend(FBank) / postprocess(cosine) 계약

담기지 않는 것: v5/v4 NEON 커널 구현체.  기계어라 런타임 바이너리에 있다.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from runtime_bundle_exporter.writer.execution_plan_writer import (
    TensorStorageType,
    read_execution_plan,
)

from .dispatch_table import (
    encode_dispatch_table,
    parse_dispatch_source,
    summarise,
)
from .format import (
    ModelPackageError,
    ModelPackageSection,
    ModelPackageSectionType as T,
    build_model_package,
    canonical_json_bytes,
    read_model_package,
)
from .packager import FINAL_98_SUITE, _load_json, _source_manifest_kind, \
    _tensor_contract, sha256_file

AUDIO_SECONDS: Mapping[int, float] = {98: 1.0, 298: 3.0, 498: 5.0, 998: 10.0}


def build_bucket_package(
    *,
    bucket_frames: int,
    plan_path: Path,
    weights_path: Path,
    source_manifest_path: Path,
    streaming_weights_path: Path | None = None,
    streaming_schedule_path: Path | None = None,
    dispatch_source_path: Path | None = None,
    model_name: str | None = None,
    optimization_suite: str = FINAL_98_SUITE,
) -> tuple[bytes, dict[str, Any]]:
    model_name = model_name or f"campp_sv_{bucket_frames}"
    for path in (plan_path, weights_path, source_manifest_path):
        if not Path(path).is_file():
            raise ModelPackageError(f"required artifact missing: {path}")

    plan = read_execution_plan(plan_path)
    if plan.header.bucket_frames != bucket_frames:
        raise ModelPackageError(
            f"{Path(plan_path).name} declares bucket "
            f"{plan.header.bucket_frames}, expected {bucket_frames}")
    inputs = [t for t in plan.tensors
              if int(t.storage_type) == int(TensorStorageType.INPUT)]
    outputs = [t for t in plan.tensors
               if int(t.storage_type) == int(TensorStorageType.OUTPUT)]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ModelPackageError("expected exactly one input and one output")

    plan_bytes = Path(plan_path).read_bytes()
    weights_bytes = Path(weights_path).read_bytes()
    source_manifest = _load_json(source_manifest_path)

    sections = [
        ModelPackageSection(T.PACKED_WEIGHTS, weights_bytes),
    ]
    per_bucket: dict[T, dict[int, bytes]] = {}
    optional: dict[str, Any] = {
        "streaming_weights": False,
        "weight_stream_schedule": False,
        "kernel_dispatch_table": False,
    }

    if streaming_weights_path and Path(streaming_weights_path).is_file():
        sections.append(ModelPackageSection(
            T.STREAMING_WEIGHTS, Path(streaming_weights_path).read_bytes()))
        optional["streaming_weights"] = True
    if streaming_schedule_path and Path(streaming_schedule_path).is_file():
        per_bucket.setdefault(T.WEIGHT_STREAM_SCHEDULE, {})[bucket_frames] = \
            Path(streaming_schedule_path).read_bytes()
        optional["weight_stream_schedule"] = True

    dispatch_summary: dict[str, int] | None = None
    dispatch_source_sha: str | None = None
    if dispatch_source_path and Path(dispatch_source_path).is_file():
        tables = parse_dispatch_source(dispatch_source_path)
        if bucket_frames not in tables:
            raise ModelPackageError(
                f"dispatch source has no table for bucket {bucket_frames}")
        per_bucket.setdefault(T.KERNEL_DISPATCH_TABLE, {})[bucket_frames] = \
            encode_dispatch_table(bucket_frames, tables[bucket_frames])
        dispatch_summary = summarise(tables[bucket_frames])
        dispatch_source_sha = sha256_file(Path(dispatch_source_path))
        optional["kernel_dispatch_table"] = True

    metadata = {
        "schema_version": 2,
        "model_name": model_name,
        "task": "speaker_embedding",
        "deployment_scope": f"qrb2210_bucket_{bucket_frames}",
        "target": {"architecture": "aarch64", "device": "QRB2210",
                   "required_isa": ["neon"],
                   "backend": "cpu_aarch64_o4i4_final"},
        "bucket_frames": bucket_frames,
        "audio_seconds": AUDIO_SECONDS.get(bucket_frames),
        "input": _tensor_contract(inputs[0]),
        "output": _tensor_contract(outputs[0]),
        "operator_count": len(plan.operators),
        "tensor_count": len(plan.tensors),
        "optimization_suite": optimization_suite,
        # 커널 구현체는 런타임 바이너리에 있다.  모델만 바꿔서는 커널이 바뀌지 않는다.
        "optimization_kernels_in_model": False,
        "optional_sections": optional,
        "kernel_dispatch_summary": dispatch_summary,
        "source": {
            "manifest_kind": _source_manifest_kind(source_manifest),
            "manifest_sha256": sha256_file(Path(source_manifest_path)),
            "plan_sha256": sha256_file(Path(plan_path)),
            "weights_sha256": sha256_file(Path(weights_path)),
            "dispatch_source_sha256": dispatch_source_sha,
        },
    }
    frontend = {
        "schema_version": 2,
        "contract_kind": "audio_frontend_requirement",
        "implemented_in_model": False,
        "waveform": {"sample_rate_hz": 16000, "channels": 1,
                     "dtype": "float32", "range": "[-1, 1]"},
        "fbank": {"implementation": "torchaudio.compliance.kaldi.fbank",
                  "num_mel_bins": 80, "frame_length_ms": 25.0,
                  "frame_shift_ms": 10.0, "dither": 0.0, "energy_floor": 0.0,
                  "window_type": "hamming", "use_energy": False,
                  "cmvn": "subtract mean over time axis"},
        "frames": bucket_frames,
        "audio_seconds": AUDIO_SECONDS.get(bucket_frames),
        "short_audio_padding_policy": "undecided",
    }
    postprocess = {
        "schema_version": 2,
        "contract_kind": "speaker_verification_postprocess",
        "implemented_in_model": False,
        "embedding_dim": 192, "scoring": "cosine", "normalisation": "l2",
        "threshold_status": "must_be_calibrated_from_trials",
        "speaker_templates_in_model": False,
    }
    sections += [
        ModelPackageSection(T.MODEL_METADATA_JSON, canonical_json_bytes(metadata)),
        ModelPackageSection(T.FRONTEND_CONTRACT_JSON, canonical_json_bytes(frontend)),
        ModelPackageSection(T.POSTPROCESS_CONTRACT_JSON,
                            canonical_json_bytes(postprocess)),
    ]

    package = build_model_package(
        bucket_frames=bucket_frames, sections=sections,
        plans_by_bucket={bucket_frames: plan_bytes},
        per_bucket_sections=per_bucket)

    loaded = read_model_package(package)
    if loaded.plan_for(bucket_frames) != plan_bytes:
        raise ModelPackageError("execution plan did not round-trip")
    if loaded.sections[T.PACKED_WEIGHTS] != weights_bytes:
        raise ModelPackageError("packed weights did not round-trip")
    for section_type, table in per_bucket.items():
        if loaded.optional_section(section_type, bucket_frames) != \
                table[bucket_frames]:
            raise ModelPackageError(f"{section_type.name} did not round-trip")

    report = {
        "schema_version": 2,
        "model_name": model_name,
        "bucket_frames": bucket_frames,
        "package_size_bytes": len(package),
        "package_sha256": hashlib.sha256(package).hexdigest(),
        "optional_sections": optional,
        "kernel_dispatch_summary": dispatch_summary,
        "section_bytes": {
            "execution_plan": len(plan_bytes),
            "packed_weights": len(weights_bytes),
            "streaming_weights": len(
                loaded.sections.get(T.STREAMING_WEIGHTS, b"")),
            "weight_stream_schedule": len(
                loaded.optional_section(T.WEIGHT_STREAM_SCHEDULE,
                                        bucket_frames) or b""),
            "kernel_dispatch_table": len(
                loaded.optional_section(T.KERNEL_DISPATCH_TABLE,
                                        bucket_frames) or b""),
        },
        "source": metadata["source"],
        "optimization_suite": optimization_suite,
        "optimization_kernels_in_model": False,
    }
    return package, report
