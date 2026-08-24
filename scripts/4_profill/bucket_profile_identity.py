#!/usr/bin/env python3
"""Validate that an operator profile describes one exact execution plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "src/python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from runtime_bundle_exporter.format.binary_format_schema import (  # noqa: E402
    OperatorCode,
    TensorStorageType,
)
from runtime_bundle_exporter.writer.execution_plan_writer import (  # noqa: E402
    read_execution_plan,
)


class BucketProfileIdentityError(RuntimeError):
    """The profile and target execution plan do not describe the same graph."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _operator_name(opcode: int) -> str:
    try:
        return OperatorCode(opcode).name
    except ValueError:
        return f"UNKNOWN_{opcode}"


def _constant_shapes(plan: Any, descriptor: Any) -> list[list[int]]:
    shapes = []
    for tensor_id in descriptor.input_tensor_ids[: descriptor.input_count]:
        tensor = plan.tensors[tensor_id]
        if tensor.storage_type == TensorStorageType.CONSTANT:
            shapes.append(list(tensor.dimensions[: tensor.rank]))
    return shapes


def _tensor_ids(row: dict[str, Any], field: str) -> list[int]:
    tensors = row.get(field)
    if not isinstance(tensors, list):
        raise BucketProfileIdentityError(f"profile operator {field} is missing")
    try:
        return [int(item["tensor_id"]) for item in tensors]
    except (KeyError, TypeError, ValueError) as exc:
        raise BucketProfileIdentityError(
            f"profile operator {field} is invalid"
        ) from exc


def validate_profile_against_plan(
    profile_path: Path, plan_path: Path, bucket_frames: int,
) -> dict[str, Any]:
    """Validate bucket, plan hash, and every operator identity field."""

    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BucketProfileIdentityError(
            f"cannot read operator profile {profile_path}: {exc}"
        ) from exc
    if not isinstance(profile, dict):
        raise BucketProfileIdentityError("operator profile is not an object")
    configuration = profile.get("configuration")
    measured_bucket = (
        configuration.get("bucket_frames")
        if isinstance(configuration, dict) else None
    )
    if measured_bucket != bucket_frames:
        raise BucketProfileIdentityError(
            f"profile bucket mismatch: expected {bucket_frames}, "
            f"got {measured_bucket}"
        )
    plan = read_execution_plan(plan_path)
    if plan.header.bucket_frames != bucket_frames:
        raise BucketProfileIdentityError(
            f"plan bucket mismatch: expected {bucket_frames}, "
            f"got {plan.header.bucket_frames}"
        )
    artifacts = profile.get("artifacts")
    plan_artifact = artifacts.get("plan") if isinstance(artifacts, dict) else None
    recorded_sha256 = (
        plan_artifact.get("sha256")
        if isinstance(plan_artifact, dict) else None
    )
    actual_sha256 = _sha256(plan_path)
    if recorded_sha256 != actual_sha256:
        raise BucketProfileIdentityError(
            "profile plan SHA-256 does not match the target execution plan"
        )
    rows = profile.get("operators")
    if not isinstance(rows, list):
        raise BucketProfileIdentityError("profile operators are missing")
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "operator_id" not in row:
            raise BucketProfileIdentityError("invalid profile operator row")
        operator_id = int(row["operator_id"])
        if operator_id in by_id:
            raise BucketProfileIdentityError(
                f"duplicate profile operator ID: {operator_id}"
            )
        by_id[operator_id] = row
    expected_ids = {descriptor.operator_id for descriptor in plan.operators}
    if set(by_id) != expected_ids:
        missing = sorted(expected_ids - set(by_id))
        extra = sorted(set(by_id) - expected_ids)
        raise BucketProfileIdentityError(
            f"profile operator set mismatch: missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )

    qlinear_count = 0
    for descriptor in plan.operators:
        operator_id = descriptor.operator_id
        row = by_id[operator_id]
        expected_type = _operator_name(int(descriptor.opcode))
        expected_inputs = list(
            descriptor.input_tensor_ids[: descriptor.input_count]
        )
        expected_outputs = list(
            descriptor.output_tensor_ids[: descriptor.output_count]
        )
        mismatches = []
        if int(row.get("kernel_id", -1)) != descriptor.kernel_id:
            mismatches.append("kernel_id")
        if row.get("operator_type") != expected_type:
            mismatches.append("opcode/operator_type")
        if _tensor_ids(row, "input_tensors") != expected_inputs:
            mismatches.append("input_tensor_ids")
        if _tensor_ids(row, "output_tensors") != expected_outputs:
            mismatches.append("output_tensor_ids")
        if row.get("weight_shapes") != _constant_shapes(plan, descriptor):
            mismatches.append("weight_shapes")
        if not isinstance(row.get("kernel_name"), str) or not row["kernel_name"]:
            mismatches.append("kernel_name")
        if mismatches:
            raise BucketProfileIdentityError(
                f"operator {operator_id} identity mismatch: "
                + ", ".join(mismatches)
            )
        if expected_type == "QLINEAR_CONV":
            qlinear_count += 1
    return {
        "ready": True,
        "bucket_frames": bucket_frames,
        "plan_sha256": actual_sha256,
        "operator_count": len(plan.operators),
        "qlinear_conv_count": qlinear_count,
    }
