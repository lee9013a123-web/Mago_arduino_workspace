"""Reference Runtime 산출물과 검증 증거를 최종 기준선 보고서로 묶는다.

이 모듈은 ONNX나 NumPy를 다시 실행하지 않는다. 앞 단계가 만든 bundle manifest,
Operator 비교 결과, End-to-end 결과를 서로 대조하고 입력 파일의 SHA-256을 기록한다.
따라서 보고서 생성은 빠르고, 같은 입력에서는 같은 결과가 나온다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable, Mapping


EXPECTED_BUCKETS = (98, 298, 498, 998)
GOOD_STATUSES = frozenset(("exact", "within_tolerance"))
CHECKPOINTS = (
    (2281, "input_transform", "입력 Transpose 직후"),
    (2332, "head", "Head 출력"),
    (2647, "dense_block_1", "Dense block 1 경계"),
    (3276, "dense_block_2", "Dense block 2 경계"),
    (3697, "dense_block_3", "Dense block 3 경계"),
    (3712, "statistics_pooling", "Statistics pooling 출력"),
    (3718, "embedding", "최종 embedding"),
)
FINAL_FILENAMES = frozenset(
    (
        "operator_validation.json",
        "end_to_end_validation.json",
        "reference_runtime_report.md",
    )
)
_REGISTRY_OPCODE = re.compile(
    r"CAMPP_REFERENCE_ENTRY\s*\(\s*CAMPP_OP_([A-Z0-9_]+)", re.MULTILINE
)


class FreezeResultError(RuntimeError):
    """필수 검증 증거가 없거나 서로 모순될 때 발생한다."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FreezeResultError(f"필수 파일이 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FreezeResultError(f"JSON 형식이 잘못되었습니다: {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except FileNotFoundError as exc:
        raise FreezeResultError(f"해시를 계산할 파일이 없습니다: {path}") from exc
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _manifest_path(bundle_dir: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise FreezeResultError("manifest의 파일 경로가 비어 있습니다")
    # Windows에서 생성한 manifest도 Linux 보드에서 검증할 수 있게 양쪽 구분자를 받는다.
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    return path if path.is_absolute() else bundle_dir / path


def _verify_manifest_entry(
    *, label: str, path: Path, entry: Mapping[str, Any]
) -> dict[str, Any]:
    if not path.is_file():
        raise FreezeResultError(f"{label} 파일이 없습니다: {path}")
    actual_size = path.stat().st_size
    actual_hash = _sha256(path)
    expected_size = int(entry.get("size_bytes", -1))
    expected_hash = str(entry.get("sha256", ""))
    if actual_size != expected_size:
        raise FreezeResultError(
            f"{label} 크기가 manifest와 다릅니다: {actual_size} != {expected_size}"
        )
    if actual_hash != expected_hash:
        raise FreezeResultError(f"{label} SHA-256이 manifest와 다릅니다")
    return {
        "path": path,
        "size_bytes": actual_size,
        "sha256": actual_hash,
        "verified": True,
    }


def _verify_bundle(bundle_dir: Path, repository_root: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise FreezeResultError("bundle manifest의 최상위 값은 object여야 합니다")

    canonical_entry = manifest.get("canonical_model")
    weights_entry = manifest.get("weights")
    plans = manifest.get("plans")
    if not isinstance(canonical_entry, dict) or not isinstance(weights_entry, dict):
        raise FreezeResultError("manifest에 canonical_model 또는 weights가 없습니다")
    if not isinstance(plans, list):
        raise FreezeResultError("manifest의 plans는 배열이어야 합니다")

    canonical = _verify_manifest_entry(
        label="canonical model",
        path=_manifest_path(bundle_dir, canonical_entry.get("path")),
        entry=canonical_entry,
    )
    weights = _verify_manifest_entry(
        label="weights.bin",
        path=_manifest_path(bundle_dir, weights_entry.get("path")),
        entry=weights_entry,
    )

    plan_results: list[dict[str, Any]] = []
    seen_buckets: set[int] = set()
    for entry in plans:
        if not isinstance(entry, dict):
            raise FreezeResultError("manifest의 plan 항목은 object여야 합니다")
        bucket = int(entry.get("bucket_frames", -1))
        if bucket in seen_buckets:
            raise FreezeResultError(f"manifest에 {bucket}-frame plan이 중복됩니다")
        seen_buckets.add(bucket)
        verified = _verify_manifest_entry(
            label=f"plan_{bucket}.bin",
            path=_manifest_path(bundle_dir, entry.get("path")),
            entry=entry,
        )
        verified.update(
            {
                "bucket_frames": bucket,
                "tensor_count": int(entry.get("tensor_count", -1)),
                "operator_count": int(entry.get("operator_count", -1)),
                "path": _relative(verified["path"], repository_root),
            }
        )
        plan_results.append(verified)
    if seen_buckets != set(EXPECTED_BUCKETS):
        raise FreezeResultError(
            "manifest의 bucket 구성이 다릅니다: "
            f"expected={list(EXPECTED_BUCKETS)}, actual={sorted(seen_buckets)}"
        )
    if len({item["operator_count"] for item in plan_results}) != 1:
        raise FreezeResultError("bucket별 Operator 수가 서로 다릅니다")

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "format_version": int(manifest.get("format_version", -1)),
        "canonical_model": {
            **canonical,
            "path": _relative(canonical["path"], repository_root),
        },
        "weights": {
            **weights,
            "path": _relative(weights["path"], repository_root),
        },
        "plans": sorted(plan_results, key=lambda item: item["bucket_frames"]),
        "operator_count": plan_results[0]["operator_count"],
        "verified": True,
    }


def _finite_metric(items: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for item in items:
        value = item.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return max(values) if values else None


def _compact_comparison(item: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if item is None:
        return None
    keys = (
        "operator_id",
        "operator_name",
        "opcode",
        "tensor_id",
        "tensor_name",
        "reference_dtype",
        "actual_dtype",
        "reference_shape",
        "actual_shape",
        "element_count",
        "status",
        "max_abs_error",
        "mean_abs_error",
        "max_rel_error",
        "cosine_similarity",
        "mismatched_elements",
        "bitwise_differing_elements",
        "bitwise_identical",
    )
    return {key: item[key] for key in keys if key in item}


def _summarize_operator_bucket(
    path: Path,
    bucket: int,
    expected_operator_count: int,
    embedding_cosine_min: float,
    repository_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comparisons = _read_json(path)
    if not isinstance(comparisons, list):
        raise FreezeResultError(f"{path}의 최상위 값은 배열이어야 합니다")
    if len(comparisons) != expected_operator_count:
        raise FreezeResultError(
            f"{bucket}-frame 비교 수가 잘못되었습니다: "
            f"{len(comparisons)} != {expected_operator_count}"
        )
    if not all(isinstance(item, dict) for item in comparisons):
        raise FreezeResultError(f"{path}에 object가 아닌 비교 항목이 있습니다")

    operator_ids = [int(item.get("operator_id", -1)) for item in comparisons]
    if sorted(operator_ids) != list(range(expected_operator_count)):
        raise FreezeResultError(
            f"{bucket}-frame 결과가 Operator 0..{expected_operator_count - 1}을 "
            "정확히 한 번씩 포함하지 않습니다"
        )

    failures = [item for item in comparisons if item.get("status") not in GOOD_STATUSES]
    first_failure = min(
        failures,
        key=lambda item: (int(item.get("operator_id", -1)), int(item.get("tensor_id", -1))),
        default=None,
    )
    embedding = next(
        (item for item in comparisons if item.get("tensor_name") == "embedding"),
        None,
    )
    if embedding is None:
        raise FreezeResultError(f"{bucket}-frame 결과에 embedding이 없습니다")
    cosine_value = embedding.get("cosine_similarity")
    cosine = float(cosine_value) if isinstance(cosine_value, (int, float)) else None
    cosine_passed = cosine is not None and cosine >= embedding_cosine_min

    by_opcode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in comparisons:
        opcode = item.get("opcode")
        if not isinstance(opcode, str) or not opcode:
            raise FreezeResultError(f"{bucket}-frame 결과에 opcode가 없는 항목이 있습니다")
        by_opcode[opcode].append(item)
    opcode_results = []
    for opcode in sorted(by_opcode):
        entries = by_opcode[opcode]
        opcode_failures = [item for item in entries if item.get("status") not in GOOD_STATUSES]
        opcode_results.append(
            {
                "opcode": opcode,
                "output_tensors": len(entries),
                "failed_tensors": len(opcode_failures),
                "max_abs_error": _finite_metric(entries, "max_abs_error"),
                "max_rel_error": _finite_metric(entries, "max_rel_error"),
                "passed": not opcode_failures,
            }
        )

    status_counts = Counter(str(item.get("status")) for item in comparisons)
    summary = {
        "bucket_frames": bucket,
        "compared_tensors": len(comparisons),
        "passed_tensors": len(comparisons) - len(failures),
        "failed_tensors": len(failures),
        "status_counts": dict(sorted(status_counts.items())),
        "max_abs_error": _finite_metric(comparisons, "max_abs_error"),
        "max_rel_error": _finite_metric(comparisons, "max_rel_error"),
        "embedding": _compact_comparison(embedding),
        "embedding_cosine_min": embedding_cosine_min,
        "embedding_cosine_passed": cosine_passed,
        "first_failure": _compact_comparison(first_failure),
        "opcode_results": opcode_results,
        "source": {
            "path": _relative(path, repository_root),
            "sha256": _sha256(path),
        },
        "passed": not failures and cosine_passed,
    }
    return summary, comparisons


def _implemented_opcodes(registry_path: Path) -> list[str]:
    try:
        text = registry_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FreezeResultError(f"kernel registry가 없습니다: {registry_path}") from exc
    opcodes = sorted(set(_REGISTRY_OPCODE.findall(text)))
    if not opcodes:
        raise FreezeResultError(f"kernel registry에서 opcode를 찾지 못했습니다: {registry_path}")
    return opcodes


def _git_metadata(repository_root: Path, output_dir: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", "-c", "core.quotepath=false", *arguments),
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    try:
        sha = run("rev-parse", "HEAD")
        commit_time = run("show", "-s", "--format=%cI", "HEAD")
        status_lines = run("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise FreezeResultError(f"Git revision을 읽지 못했습니다: {exc}") from exc

    ignored = {
        _relative(output_dir / filename, repository_root) for filename in FINAL_FILENAMES
    }
    meaningful_status: list[str] = []
    for line in status_lines:
        candidate = line[3:].replace("\\", "/") if len(line) >= 4 else line
        # rename 행은 "old -> new" 형식이다. 어느 쪽이든 최종 출력이면 제외한다.
        paths = {part.strip('" ') for part in candidate.split(" -> ")}
        if paths and paths.issubset(ignored):
            continue
        meaningful_status.append(line)
    return {
        "sha": sha,
        "commit_time": commit_time,
        "worktree_clean_excluding_generated_reports": not meaningful_status,
        "worktree_changes_excluding_generated_reports": meaningful_status,
    }


def _source_evidence(path: Path, repository_root: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, repository_root),
        "sha256": _sha256(path),
    }


def _checkpoint_results(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_tensor = {int(item.get("tensor_id", -1)): item for item in comparisons}
    output = []
    for tensor_id, key, description in CHECKPOINTS:
        item = by_tensor.get(tensor_id)
        output.append(
            {
                "key": key,
                "description": description,
                "tensor_id": tensor_id,
                "comparison": _compact_comparison(item),
                "present": item is not None,
                "passed": item is not None and item.get("status") in GOOD_STATUSES,
            }
        )
    return output


def _build_end_to_end_bucket(
    *,
    bucket_summary: Mapping[str, Any],
    comparisons: list[dict[str, Any]],
    validation_dir: Path,
    expected_operator_count: int,
    repository_root: Path,
) -> dict[str, Any]:
    bucket = int(bucket_summary["bucket_frames"])
    detail_path = validation_dir / f"end_to_end_{bucket}.json"
    detail: Mapping[str, Any] | None = None
    evidence: dict[str, Any]
    if detail_path.is_file():
        candidate = _read_json(detail_path)
        if not isinstance(candidate, dict) or int(candidate.get("bucket_frames", -1)) != bucket:
            raise FreezeResultError(f"{detail_path}의 bucket_frames가 잘못되었습니다")
        detail = candidate
        evidence = {
            "kind": "dedicated_end_to_end_report",
            **_source_evidence(detail_path, repository_root),
        }
    else:
        source = dict(bucket_summary["source"])
        evidence = {
            "kind": "inferred_from_complete_runtime_tensor_dump_comparison",
            **source,
        }

    compared = int(bucket_summary["compared_tensors"])
    executed = (
        int(detail.get("executed_operator_count", detail.get("operator_count", -1)))
        if detail is not None
        else compared
    )
    execution_passed = executed == expected_operator_count and compared == expected_operator_count
    checkpoints = _checkpoint_results(comparisons)
    return {
        "bucket_frames": bucket,
        "operator_count": expected_operator_count,
        "executed_operator_count": executed,
        "execution_passed": execution_passed,
        "execution_evidence": evidence,
        "compared_tensors": compared,
        "failed_tensors": int(bucket_summary["failed_tensors"]),
        "numerical_passed": bool(bucket_summary["passed"]),
        "embedding": bucket_summary["embedding"],
        "checkpoints": checkpoints,
        "first_failure": bucket_summary["first_failure"],
    }


def _cross_bucket_evidence(
    validation_dir: Path, repository_root: Path
) -> dict[str, Any]:
    path = validation_dir / "end_to_end_summary.json"
    if not path.is_file():
        return {
            "recorded": False,
            "passed": False,
            "reason": "end_to_end_summary.json이 없어 잘못된 bucket 입력 거부 증거를 고정하지 못했습니다",
            "checks": [],
        }
    document = _read_json(path)
    if not isinstance(document, dict):
        raise FreezeResultError(f"{path}의 최상위 값은 object여야 합니다")
    checks = document.get("bucket_mismatch_checks", [])
    errors = document.get("bucket_mismatch_errors", [])
    passed = bool(document.get("cross_bucket_rejection_passed", False))
    return {
        "recorded": True,
        "passed": passed,
        "checks": checks if isinstance(checks, list) else [],
        "errors": errors if isinstance(errors, list) else [],
        "source": _source_evidence(path, repository_root),
    }


def _baseline_id(
    *,
    bundle: Mapping[str, Any],
    evidence_hashes: Iterable[str],
    registry_hash: str,
    config_hash: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(bundle["manifest_sha256"]).encode("ascii"))
    digest.update(registry_hash.encode("ascii"))
    digest.update(config_hash.encode("ascii"))
    for value in sorted(evidence_hashes):
        digest.update(value.encode("ascii"))
    return f"campp-reference-v{bundle['format_version']}-{digest.hexdigest()[:16]}"


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False) + "\n").encode(
        "utf-8"
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _format_number(value: object, digits: int = 9) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{float(value):.{digits}g}"


def _render_report(
    *,
    baseline_id: str,
    git: Mapping[str, Any],
    bundle: Mapping[str, Any],
    operator_document: Mapping[str, Any],
    end_to_end_document: Mapping[str, Any],
    limitations: list[str],
    phase4_reasons: list[str],
) -> str:
    lines = [
        "# CAM++ Reference Runtime 고정 보고서",
        "",
        f"- 기준선 ID: `{baseline_id}`",
        f"- Git SHA: `{git['sha']}`",
        f"- canonical model SHA-256: `{bundle['canonical_model']['sha256']}`",
        f"- binary format version: `{bundle['format_version']}`",
        f"- weights SHA-256: `{bundle['weights']['sha256']}`",
        f"- validation config SHA-256: `{operator_document['validation_config']['sha256']}`",
        f"- 기준선 고정: `{'PASS' if operator_document['baseline_frozen'] else 'FAIL'}`",
        f"- Phase 4 진행 가능: `{'YES' if end_to_end_document['phase4_ready'] else 'NO'}`",
        "",
        "## 고정 범위",
        "",
        "`manifest.json`에 기록된 canonical ONNX, `weights.bin`, 네 execution plan의 "
        "크기와 SHA-256을 다시 계산했습니다. 각 bucket의 1,438개 Operator 출력 비교 파일도 "
        "SHA-256으로 고정했습니다.",
        "",
        "## 구현한 opcode",
        "",
        ", ".join(f"`{opcode}`" for opcode in operator_document["implemented_opcodes"]),
        "",
        "## 허용 오차",
        "",
        "| 항목 | 기준 |",
        "|---|---:|",
        f"| FP32 절대 오차(atol) | {operator_document['tolerances']['float_atol']} |",
        f"| FP32 상대 오차(rtol) | {operator_document['tolerances']['float_rtol']} |",
        f"| embedding cosine 최솟값 | {operator_document['tolerances']['embedding_cosine_min']} |",
        "| 정수 Tensor | 원소 단위 완전 일치 |",
        "",
        "## Bucket별 결과",
        "",
        "| frames | 실행 | 비교 Tensor | 실패 Tensor | embedding cosine | 수치 판정 |",
        "|---:|:---:|---:|---:|---:|:---:|",
    ]
    e2e_by_bucket = {
        int(item["bucket_frames"]): item for item in end_to_end_document["buckets"]
    }
    for item in operator_document["buckets"]:
        bucket = int(item["bucket_frames"])
        end = e2e_by_bucket[bucket]
        embedding = item.get("embedding") or {}
        lines.append(
            f"| {bucket} | {'PASS' if end['execution_passed'] else 'FAIL'} | "
            f"{item['compared_tensors']} | {item['failed_tensors']} | "
            f"{_format_number(embedding.get('cosine_similarity'), 15)} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )

    lines.extend(("", "## 알려진 제한 사항", ""))
    lines.extend(f"- {item}" for item in limitations)
    lines.extend(("", "## Phase 4 판정", ""))
    if phase4_reasons:
        lines.append("전체 네 bucket을 대상으로 한 Phase 4 최적화 기준선 승인은 보류합니다.")
        lines.append("")
        lines.extend(f"- {reason}" for reason in phase4_reasons)
    else:
        lines.append("네 bucket 모두 고정 기준을 만족하여 Phase 4 최적화를 진행할 수 있습니다.")
    lines.extend(
        (
            "",
            "최적화 kernel은 같은 입력과 plan으로 실행한 뒤 이 기준선의 dtype, shape, 정수 "
            "완전 일치 및 FP32 허용 오차를 그대로 적용해야 합니다. 이 보고서는 성능 기준이 "
            "아니라 정확도 기준입니다.",
            "",
        )
    )
    return "\n".join(lines)


def freeze_reference_results(
    *,
    repository_root: Path,
    bundle_dir: Path,
    validation_dir: Path,
    config_path: Path,
    registry_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """검증 증거를 확인하고 최종 JSON 두 개와 Markdown 보고서를 원자적으로 쓴다."""

    repository_root = repository_root.resolve()
    bundle_dir = bundle_dir.resolve()
    validation_dir = validation_dir.resolve()
    output_dir = output_dir.resolve()
    config_path = config_path.resolve()
    registry_path = registry_path.resolve()

    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise FreezeResultError("runtime config의 최상위 값은 object여야 합니다")
    configured_buckets = tuple(int(value) for value in config.get("buckets", []))
    if configured_buckets != EXPECTED_BUCKETS:
        raise FreezeResultError(
            f"config buckets는 {list(EXPECTED_BUCKETS)}여야 합니다: {configured_buckets}"
        )
    tolerances = config.get("tolerances")
    if not isinstance(tolerances, dict):
        raise FreezeResultError("config에 tolerances가 없습니다")
    float_atol = float(tolerances.get("float_atol", -1.0))
    float_rtol = float(tolerances.get("float_rtol", -1.0))
    embedding_cosine_min = float(tolerances.get("embedding_cosine_min", -1.0))
    if float_atol < 0.0 or float_rtol < 0.0 or not 0.0 <= embedding_cosine_min <= 1.0:
        raise FreezeResultError("runtime config의 허용 오차가 유효하지 않습니다")

    bundle = _verify_bundle(bundle_dir, repository_root)
    implemented = _implemented_opcodes(registry_path)
    registry_hash = _sha256(registry_path)
    config_hash = _sha256(config_path)
    git = _git_metadata(repository_root, output_dir)

    bucket_summaries: list[dict[str, Any]] = []
    comparisons_by_bucket: dict[int, list[dict[str, Any]]] = {}
    graph_opcodes: set[str] = set()
    for bucket in EXPECTED_BUCKETS:
        summary, comparisons = _summarize_operator_bucket(
            validation_dir / f"compare_{bucket}.json",
            bucket,
            int(bundle["operator_count"]),
            embedding_cosine_min,
            repository_root,
        )
        bucket_summaries.append(summary)
        comparisons_by_bucket[bucket] = comparisons
        graph_opcodes.update(str(item["opcode"]) for item in comparisons)

    missing_opcodes = sorted(graph_opcodes - set(implemented))
    unused_implemented_opcodes = sorted(set(implemented) - graph_opcodes)
    all_buckets_complete = all(
        int(item["compared_tensors"]) == int(bundle["operator_count"])
        for item in bucket_summaries
    )
    baseline_frozen = bool(bundle["verified"] and all_buckets_complete and not missing_opcodes)
    evidence_hashes = [str(item["source"]["sha256"]) for item in bucket_summaries]
    baseline_id = _baseline_id(
        bundle=bundle,
        evidence_hashes=evidence_hashes,
        registry_hash=registry_hash,
        config_hash=config_hash,
    )

    common = {
        "schema_version": 1,
        "baseline_id": baseline_id,
        "source_revision": git,
        "model": {
            "canonical_path": bundle["canonical_model"]["path"],
            "canonical_sha256": bundle["canonical_model"]["sha256"],
            "manifest_path": _relative(bundle["manifest_path"], repository_root),
            "manifest_sha256": bundle["manifest_sha256"],
            "weights_path": bundle["weights"]["path"],
            "weights_sha256": bundle["weights"]["sha256"],
            "binary_format_version": bundle["format_version"],
            "plans": [
                {
                    "bucket_frames": item["bucket_frames"],
                    "path": item["path"],
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                    "tensor_count": item["tensor_count"],
                    "operator_count": item["operator_count"],
                }
                for item in bundle["plans"]
            ],
        },
        "validation_config": {
            "path": _relative(config_path, repository_root),
            "sha256": config_hash,
        },
    }
    operator_document: dict[str, Any] = {
        **common,
        "validation_kind": "operator_output_accuracy",
        "tolerances": {
            "float_atol": float_atol,
            "float_rtol": float_rtol,
            "embedding_cosine_min": embedding_cosine_min,
            "integer_policy": "exact_element_match",
        },
        "kernel_registry": {
            "path": _relative(registry_path, repository_root),
            "sha256": registry_hash,
        },
        "implemented_opcodes": implemented,
        "graph_opcodes": sorted(graph_opcodes),
        "missing_opcodes": missing_opcodes,
        "unused_implemented_opcodes": unused_implemented_opcodes,
        "buckets": bucket_summaries,
        "all_buckets_complete": all_buckets_complete,
        "all_buckets_numerically_passed": all(item["passed"] for item in bucket_summaries),
        "baseline_frozen": baseline_frozen,
    }

    end_buckets = [
        _build_end_to_end_bucket(
            bucket_summary=summary,
            comparisons=comparisons_by_bucket[int(summary["bucket_frames"])],
            validation_dir=validation_dir,
            expected_operator_count=int(bundle["operator_count"]),
            repository_root=repository_root,
        )
        for summary in bucket_summaries
    ]
    cross_bucket = _cross_bucket_evidence(validation_dir, repository_root)
    dedicated_complete = all(
        item["execution_evidence"]["kind"] == "dedicated_end_to_end_report"
        for item in end_buckets
    )
    phase4_reasons: list[str] = []
    if not baseline_frozen:
        phase4_reasons.append("bundle/opcode/Operator 증거 기준선이 완전하지 않습니다.")
    failed_buckets = [
        int(item["bucket_frames"]) for item in end_buckets if not item["numerical_passed"]
    ]
    if failed_buckets:
        phase4_reasons.append(
            f"수치 허용 기준을 통과하지 못한 bucket이 있습니다: {failed_buckets}."
        )
    if not dedicated_complete:
        phase4_reasons.append(
            "네 bucket의 전용 end_to_end_*.json이 모두 고정되지 않았습니다. "
            "없는 bucket은 전체 Tensor dump 비교로 실행 완료를 추론했습니다."
        )
    if not cross_bucket["passed"]:
        phase4_reasons.append("잘못된 frame 수를 BUCKET_MISMATCH로 거부한 증거가 고정되지 않았습니다.")
    if not git["worktree_clean_excluding_generated_reports"]:
        phase4_reasons.append("생성 보고서를 제외한 Git working tree가 깨끗하지 않습니다.")

    phase4_ready = not phase4_reasons
    end_to_end_document: dict[str, Any] = {
        **common,
        "validation_kind": "multi_bucket_end_to_end",
        "tolerances": operator_document["tolerances"],
        "shared_runtime_binary_requirement": True,
        "shared_weights_requirement": True,
        "buckets": end_buckets,
        "all_buckets_executed": all(item["execution_passed"] for item in end_buckets),
        "all_buckets_numerically_passed": not failed_buckets,
        "dedicated_end_to_end_reports_complete": dedicated_complete,
        "cross_bucket_rejection": cross_bucket,
        "phase4_ready": phase4_ready,
        "phase4_blockers": phase4_reasons,
    }

    limitations = [
        "이 기준선은 scalar CPU Reference backend의 정확도를 기록하며 성능을 보증하지 않습니다.",
    ]
    if 998 in failed_buckets:
        first = next(
            item["first_failure"]
            for item in end_buckets
            if item["bucket_frames"] == 998
        )
        limitations.append(
            "998-frame은 FP32 1-ULP 차이가 QuantizeLinear 반올림 경계를 넘은 뒤 증폭되어 "
            f"{next(item['failed_tensors'] for item in end_buckets if item['bucket_frames'] == 998)}개 "
            f"Tensor가 실패합니다. 첫 엄격 불일치는 Operator #{first.get('operator_id')} "
            f"{first.get('opcode')}이며, 현재 기준에서는 PASS로 간주하지 않습니다."
        )
    if not dedicated_complete:
        missing_detail = [
            item["bucket_frames"]
            for item in end_buckets
            if item["execution_evidence"]["kind"] != "dedicated_end_to_end_report"
        ]
        limitations.append(f"전용 End-to-end 상세 보고서가 없는 bucket: {missing_detail}.")
    if not cross_bucket["recorded"]:
        limitations.append("현재 results/runtime에는 교차-bucket 거부 summary가 없습니다.")

    report = _render_report(
        baseline_id=baseline_id,
        git=git,
        bundle=bundle,
        operator_document=operator_document,
        end_to_end_document=end_to_end_document,
        limitations=limitations,
        phase4_reasons=phase4_reasons,
    )

    output_paths = {
        "operator_validation": output_dir / "operator_validation.json",
        "end_to_end_validation": output_dir / "end_to_end_validation.json",
        "reference_runtime_report": output_dir / "reference_runtime_report.md",
    }
    _atomic_write(output_paths["operator_validation"], _json_bytes(operator_document))
    _atomic_write(output_paths["end_to_end_validation"], _json_bytes(end_to_end_document))
    _atomic_write(output_paths["reference_runtime_report"], report.encode("utf-8"))
    return {
        "baseline_id": baseline_id,
        "baseline_frozen": baseline_frozen,
        "phase4_ready": phase4_ready,
        "failed_buckets": failed_buckets,
        "output_paths": output_paths,
    }
