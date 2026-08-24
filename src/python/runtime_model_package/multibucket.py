"""4개 bucket을 한 파일에 담는 multi-bucket 모델 패키지.

ONNX가 파일 하나로 여러 길이를 처리하듯, bucket 98/298/498/998의 execution plan을
한 `.camppmodel`에 넣는다.  **weights는 4개 bucket이 공유한다** -- bucket마다 파일을
따로 만들면 7.57 MB weights가 그만큼 복제되어 31.6 MB가 되지만, 공유하면 8.9 MB로
끝난다.

담기는 것과 담기지 않는 것:

    담긴다   : bucket별 execution plan, 공유 packed weights, 모델 메타데이터,
               frontend(FBank) 계약, postprocess(cosine/L2) 계약
    안 담긴다: 최적화 커널 코드.  Final V3 런타임 바이너리에 있으므로 모델과
               런타임을 함께 배포해야 한다.  화자 임계값도 보정 전이라 뺀다.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from runtime_bundle_exporter.writer.execution_plan_writer import (
    TensorStorageType,
    read_execution_plan,
)

from .format import (
    ModelPackageError,
    ModelPackageSection,
    ModelPackageSectionType,
    build_model_package,
    canonical_json_bytes,
    read_model_package,
)
from .packager import (
    FINAL_98_SUITE,
    _load_json,
    _source_manifest_kind,
    _tensor_contract,
    sha256_file,
)

DEFAULT_BUCKETS: tuple[int, ...] = (98, 298, 498, 998)


def _plan_contract(plan, bucket_frames: int) -> dict[str, Any]:
    inputs = [t for t in plan.tensors
              if int(t.storage_type) == int(TensorStorageType.INPUT)]
    outputs = [t for t in plan.tensors
               if int(t.storage_type) == int(TensorStorageType.OUTPUT)]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ModelPackageError(
            f"bucket {bucket_frames}: expected exactly one input and output")
    return {
        "bucket_frames": bucket_frames,
        "input": _tensor_contract(inputs[0]),
        "output": _tensor_contract(outputs[0]),
        "operator_count": len(plan.operators),
        "tensor_count": len(plan.tensors),
    }


def build_multibucket_package(
    *,
    plan_paths: Mapping[int, Path],
    weights_path: Path,
    source_manifest_path: Path,
    model_name: str = "campp_sv_multibucket",
    optimization_suite: str = FINAL_98_SUITE,
    buckets: Sequence[int] = DEFAULT_BUCKETS,
) -> tuple[bytes, dict[str, Any]]:
    if not weights_path.is_file():
        raise ModelPackageError(f"weights file is missing: {weights_path}")
    missing = [b for b in buckets if b not in plan_paths]
    if missing:
        raise ModelPackageError(f"missing plan for buckets: {missing}")

    source_manifest = _load_json(source_manifest_path)
    source_manifest_kind = _source_manifest_kind(source_manifest)
    source_manifest_sha256 = sha256_file(source_manifest_path)
    weights_bytes = weights_path.read_bytes()
    weights_sha256 = sha256_file(weights_path)

    plans_by_bucket: dict[int, bytes] = {}
    contracts: list[dict[str, Any]] = []
    plan_hashes: dict[str, str] = {}
    for bucket in sorted(buckets):
        path = Path(plan_paths[bucket])
        if not path.is_file():
            raise ModelPackageError(f"plan file is missing: {path}")
        plan = read_execution_plan(path)
        if plan.header.bucket_frames != bucket:
            raise ModelPackageError(
                f"{path.name} declares bucket {plan.header.bucket_frames}, "
                f"expected {bucket}")
        plans_by_bucket[bucket] = path.read_bytes()
        contracts.append(_plan_contract(plan, bucket))
        plan_hashes[str(bucket)] = sha256_file(path)

    # 모든 bucket의 입출력 계약이 frame 축을 빼면 같아야 한다.  다르면 하나의
    # 모델로 묶을 수 없다.
    shapes = {tuple(c["output"]["shape"]) for c in contracts}
    if len(shapes) != 1:
        raise ModelPackageError(f"output shape differs across buckets: {shapes}")

    metadata = {
        "schema_version": 2,
        "model_name": model_name,
        "task": "speaker_embedding",
        "deployment_scope": "qrb2210_multibucket",
        "target": {
            "architecture": "aarch64",
            "device": "QRB2210",
            "required_isa": ["neon"],
            "backend": "cpu_aarch64_o4i4_final",
        },
        "buckets": sorted(buckets),
        "bucket_contracts": contracts,
        "shared_weights": True,
        "optimization_suite": optimization_suite,
        "optimization_kernels_in_model": False,
        "source": {
            "manifest_kind": source_manifest_kind,
            "manifest_sha256": source_manifest_sha256,
            "weights_sha256": weights_sha256,
            "plan_sha256": plan_hashes,
        },
    }
    frontend = {
        "schema_version": 2,
        "contract_kind": "audio_frontend_requirement",
        "implemented_in_model": False,
        "waveform": {"sample_rate_hz": 16000, "channels": 1,
                     "dtype": "float32", "range": "[-1, 1]"},
        "fbank": {
            "implementation": "torchaudio.compliance.kaldi.fbank",
            "num_mel_bins": 80, "frame_length_ms": 25.0,
            "frame_shift_ms": 10.0, "dither": 0.0, "energy_floor": 0.0,
            "window_type": "hamming", "use_energy": False,
            "cmvn": "subtract mean over time axis",
        },
        # frame 수 -> 오디오 길이.  floor((N - 400) / 160) + 1 의 역이다.
        "bucket_seconds": {"98": 1.0, "298": 3.0, "498": 5.0, "998": 10.0},
        "bucket_selection": "가장 가까운 상위 bucket을 고르고 부족분은 정책에 따라 채운다",
        "short_audio_padding_policy": "undecided",
    }
    postprocess = {
        "schema_version": 2,
        "contract_kind": "speaker_verification_postprocess",
        "implemented_in_model": False,
        "embedding_dim": 192,
        "scoring": "cosine",
        "normalisation": "l2",
        "threshold_status": "must_be_calibrated_from_trials",
        "speaker_templates_in_model": False,
    }

    sections = [
        ModelPackageSection(
            ModelPackageSectionType.PACKED_WEIGHTS, weights_bytes),
        ModelPackageSection(
            ModelPackageSectionType.MODEL_METADATA_JSON,
            canonical_json_bytes(metadata)),
        ModelPackageSection(
            ModelPackageSectionType.FRONTEND_CONTRACT_JSON,
            canonical_json_bytes(frontend)),
        ModelPackageSection(
            ModelPackageSectionType.POSTPROCESS_CONTRACT_JSON,
            canonical_json_bytes(postprocess)),
    ]
    package = build_model_package(
        bucket_frames=min(buckets), sections=sections,
        plans_by_bucket=plans_by_bucket)

    loaded = read_model_package(package)
    if loaded.buckets != tuple(sorted(buckets)):
        raise ModelPackageError(
            f"packed buckets {loaded.buckets} != {tuple(sorted(buckets))}")
    for bucket, payload in plans_by_bucket.items():
        if loaded.plan_for(bucket) != payload:
            raise ModelPackageError(
                f"bucket {bucket} plan did not round-trip")
    if loaded.sections[ModelPackageSectionType.PACKED_WEIGHTS] != weights_bytes:
        raise ModelPackageError("packed weights did not round-trip")

    separate = sum(len(p) for p in plans_by_bucket.values()) \
        + len(weights_bytes) * len(plans_by_bucket)
    report = {
        "schema_version": 2,
        "model_name": model_name,
        "buckets": sorted(buckets),
        "package_size_bytes": len(package),
        "package_sha256": hashlib.sha256(package).hexdigest(),
        "weights_sha256": weights_sha256,
        "plan_sha256": plan_hashes,
        "shared_weights": True,
        "size_if_separate_files_bytes": separate,
        "size_saved_bytes": separate - len(package),
        "source_manifest_kind": source_manifest_kind,
        "source_manifest_sha256": source_manifest_sha256,
        "optimization_suite": optimization_suite,
        "contracts": {
            "buckets": contracts,
            "frontend_embedded": False,
            "postprocess_embedded": False,
            "threshold_calibrated": False,
            "optimization_kernels_embedded": False,
        },
    }
    return package, report
