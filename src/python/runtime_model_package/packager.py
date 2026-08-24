"""Build one fixed-bucket CAM++ speaker-embedding model container."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from runtime_bundle_exporter.format.binary_format_schema import (
    TensorDType,
    TensorStorageType,
)
from runtime_bundle_exporter.writer.execution_plan_writer import (
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


FINAL_98_SUITE = (
    "qconv_layer_hybrid_v3+fused_layer_hybrid_v3+bn_v2_spatial2+"
    "dequant_neon_combined+fused_dqrq_neon+remaining_optimized"
)

SUPPORTED_BUCKET_FRAMES = (98, 298, 498, 998)
AUDIO_SECONDS_BY_BUCKET = {
    98: 1.0,
    298: 3.0,
    498: 5.0,
    998: 10.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelPackageError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ModelPackageError(f"JSON root is not an object: {path}")
    return value


def _tensor_contract(descriptor: Any) -> dict[str, object]:
    try:
        dtype = TensorDType(int(descriptor.dtype)).name.lower()
    except ValueError as exc:
        raise ModelPackageError(f"unknown tensor dtype: {descriptor.dtype}") from exc
    return {
        "dtype": dtype,
        "shape": list(descriptor.dimensions[: descriptor.rank]),
        "layout": "contiguous_logical_tensor",
        "byte_order": "little-endian",
    }


def _source_manifest_hash(
    manifest: Mapping[str, Any], kind: str, bucket_frames: int | None = None,
) -> str | None:
    if manifest.get("strategy") == "offline_first_use_static_blob":
        if manifest.get("bucket_frames") != bucket_frames:
            return None
        output = manifest.get("output")
        if not isinstance(output, dict):
            return None
        key = "weights_sha256" if kind == "weights" else "plan_sha256"
        value = output.get(key)
        return str(value) if value is not None else None
    if kind == "weights":
        value = manifest.get("weights")
        return str(value.get("sha256")) if isinstance(value, dict) else None
    plans = manifest.get("plans")
    if isinstance(plans, list):
        for item in plans:
            if isinstance(item, dict) and item.get("bucket_frames") == bucket_frames:
                value = item.get("sha256")
                return str(value) if value is not None else None
    return None


def _source_manifest_kind(manifest: Mapping[str, Any]) -> str:
    if manifest.get("strategy") == "offline_first_use_static_blob":
        return "bucket_static_weight_plan"
    return "runtime_bundle"


def build_final_bucket_package(
    *,
    bucket_frames: int,
    plan_path: Path,
    weights_path: Path,
    source_manifest_path: Path,
    model_name: str | None = None,
    optimization_suite: str = FINAL_98_SUITE,
) -> tuple[bytes, dict[str, Any]]:
    if bucket_frames not in SUPPORTED_BUCKET_FRAMES:
        raise ModelPackageError(
            f"unsupported model bucket: {bucket_frames}; "
            f"expected one of {SUPPORTED_BUCKET_FRAMES}"
        )
    if model_name is None:
        model_name = f"campp_sv_{bucket_frames}"
    if not plan_path.is_file() or not weights_path.is_file():
        raise ModelPackageError("plan or weights file is missing")
    source_manifest = _load_json(source_manifest_path)
    plan = read_execution_plan(plan_path)
    if plan.header.bucket_frames != bucket_frames:
        raise ModelPackageError(
            "model package bucket differs from execution plan: "
            f"{bucket_frames} != {plan.header.bucket_frames}"
        )

    inputs = [
        item for item in plan.tensors
        if int(item.storage_type) == int(TensorStorageType.INPUT)
    ]
    outputs = [
        item for item in plan.tensors
        if int(item.storage_type) == int(TensorStorageType.OUTPUT)
    ]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ModelPackageError("speaker embedding model requires one input/output")
    input_contract = _tensor_contract(inputs[0])
    output_contract = _tensor_contract(outputs[0])
    expected_input_shape = [1, bucket_frames, 80]
    if (input_contract["dtype"] != "float32" or
            input_contract["shape"] != expected_input_shape):
        raise ModelPackageError(
            f"unexpected bucket-{bucket_frames} input: {input_contract}"
        )
    if output_contract["dtype"] != "float32" or output_contract["shape"] != [1, 192]:
        raise ModelPackageError(f"unexpected embedding output: {output_contract}")

    plan_sha256 = sha256_file(plan_path)
    weights_sha256 = sha256_file(weights_path)
    expected_plan = _source_manifest_hash(
        source_manifest, "plan", bucket_frames
    )
    expected_weights = _source_manifest_hash(
        source_manifest, "weights", bucket_frames
    )
    if expected_plan != plan_sha256:
        raise ModelPackageError(
            f"plan_{bucket_frames} SHA-256 differs from source manifest"
        )
    if expected_weights != weights_sha256:
        raise ModelPackageError("weights SHA-256 differs from source manifest")

    source_manifest_sha256 = sha256_file(source_manifest_path)
    source_manifest_kind = _source_manifest_kind(source_manifest)
    source_metadata = {
        "plan_sha256": plan_sha256,
        "weights_sha256": weights_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_kind": source_manifest_kind,
    }
    if source_manifest_kind == "runtime_bundle":
        source_metadata["bundle_manifest_sha256"] = source_manifest_sha256
    else:
        source_metadata["weight_plan_manifest_sha256"] = source_manifest_sha256

    metadata = {
        "schema_version": 1,
        "model_name": model_name,
        "model_kind": "speaker_embedding",
        "deployment_scope": f"qrb2210_bucket_{bucket_frames}",
        "target": {
            "architecture": "aarch64",
            "device": "QRB2210",
            "required_isa": ["neon"],
            "backend": "cpu_aarch64_o4i4_final",
        },
        "bucket_frames": bucket_frames,
        "audio_seconds": AUDIO_SECONDS_BY_BUCKET[bucket_frames],
        "input": input_contract,
        "output": output_contract,
        "optimization_suite": optimization_suite,
        "operator_count": len(plan.operators),
        "tensor_count": len(plan.tensors),
        "source": source_metadata,
    }
    frontend = {
        "schema_version": 1,
        "contract_kind": "audio_frontend_requirement",
        "implemented_in_model": False,
        "waveform": {
            "sample_rate_hz": 16000,
            "channels": 1,
            "normalization": [
                "pcm_to_float32_-1_1",
                "remove_dc_mean",
                "peak_normalize_0.95",
                "clip_-1_1",
            ],
        },
        "fbank": {
            "implementation_reference": "kaldi_fbank",
            "num_mel_bins": 80,
            "frame_length_ms": 25.0,
            "frame_shift_ms": 10.0,
            "dither": 0.0,
            "energy_floor": 0.0,
            "window_type": "hamming",
            "use_energy": False,
            "snip_edges": True,
        },
        "cmvn": "subtract_time_axis_mean_per_mel_bin",
        "model_input_frames": bucket_frames,
        "long_audio_windowing": "external_pipeline_policy_not_calibrated",
        "short_audio_padding": "external_pipeline_policy_not_calibrated",
    }
    postprocess = {
        "schema_version": 1,
        "contract_kind": "speaker_verification_postprocess_requirement",
        "implemented_in_model": False,
        "embedding_dimension": 192,
        "embedding_dtype": "float32",
        "similarity": "cosine",
        "normalization": "l2_before_aggregation_and_scoring",
        "enrollment_aggregation": "external_pipeline_policy_not_calibrated",
        "decision_threshold": None,
        "threshold_status": "must_be_calibrated_from_trials",
        "speaker_templates_in_model": False,
    }

    plan_bytes = plan_path.read_bytes()
    weights_bytes = weights_path.read_bytes()
    sections = [
        ModelPackageSection(
            ModelPackageSectionType.EXECUTION_PLAN, plan_bytes
        ),
        ModelPackageSection(
            ModelPackageSectionType.PACKED_WEIGHTS, weights_bytes
        ),
        ModelPackageSection(
            ModelPackageSectionType.MODEL_METADATA_JSON,
            canonical_json_bytes(metadata),
        ),
        ModelPackageSection(
            ModelPackageSectionType.FRONTEND_CONTRACT_JSON,
            canonical_json_bytes(frontend),
        ),
        ModelPackageSection(
            ModelPackageSectionType.POSTPROCESS_CONTRACT_JSON,
            canonical_json_bytes(postprocess),
        ),
    ]
    package = build_model_package(
        bucket_frames=bucket_frames, sections=sections
    )
    loaded = read_model_package(package)
    if loaded.sections[ModelPackageSectionType.EXECUTION_PLAN] != plan_bytes:
        raise ModelPackageError("packed execution plan did not round-trip")
    if loaded.sections[ModelPackageSectionType.PACKED_WEIGHTS] != weights_bytes:
        raise ModelPackageError("packed weights did not round-trip")
    report = {
        "schema_version": 1,
        "model_name": model_name,
        "bucket_frames": bucket_frames,
        "audio_seconds": AUDIO_SECONDS_BY_BUCKET[bucket_frames],
        "package_size_bytes": len(package),
        "package_sha256": hashlib.sha256(package).hexdigest(),
        "plan_sha256": plan_sha256,
        "weights_sha256": weights_sha256,
        "source_manifest_kind": source_manifest_kind,
        "source_manifest_sha256": source_manifest_sha256,
        "optimization_suite": optimization_suite,
        "contracts": {
            "input": input_contract,
            "output": output_contract,
            "frontend_embedded": False,
            "postprocess_embedded": False,
            "threshold_calibrated": False,
        },
    }
    return package, report


def build_final_98_package(
    *,
    plan_path: Path,
    weights_path: Path,
    source_manifest_path: Path,
    model_name: str = "campp_sv_98",
    optimization_suite: str = FINAL_98_SUITE,
) -> tuple[bytes, dict[str, Any]]:
    """Backward-compatible wrapper for existing Final-98 callers."""

    return build_final_bucket_package(
        bucket_frames=98,
        plan_path=plan_path,
        weights_path=weights_path,
        source_manifest_path=source_manifest_path,
        model_name=model_name,
        optimization_suite=optimization_suite,
    )
