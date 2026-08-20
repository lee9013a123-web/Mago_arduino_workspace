#!/usr/bin/env python3
"""Generate bucket-aware V3 convolution dispatch tables from measured plans."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    ROOT / "src/c/profill/optimization/candidates/conv_layer_hybrid"
    / "conv_layer_hybrid_plan.c"
)
DEFAULT_MANIFEST = (
    ROOT / "results/profiling/conv_layer_hybrid_multibucket.json"
)
QCONV_MODES = {"mac_fixed", "v4", "v5"}
FUSED_MODES = {"combined_fixed", "combined_hybrid", "combined_v5"}


class MultibucketPlanError(RuntimeError):
    """A measured plan cannot safely be translated into C dispatch data."""


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def parse_plan_spec(value: str) -> tuple[int, Path]:
    bucket_text, separator, path_text = value.partition("=")
    if not separator or not bucket_text.isdigit() or not path_text:
        raise argparse.ArgumentTypeError("plan must use BUCKET=PATH")
    bucket = int(bucket_text)
    if bucket <= 0:
        raise argparse.ArgumentTypeError("plan bucket must be positive")
    return bucket, _path(path_text)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MultibucketPlanError(f"cannot read {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise MultibucketPlanError(f"plan is not an object: {path}")
    return document


def _family(document: dict[str, Any], name: str) -> dict[str, Any]:
    families = document.get("families")
    if not isinstance(families, list):
        raise MultibucketPlanError("plan families are missing")
    matches = [
        item for item in families
        if isinstance(item, dict) and item.get("family") == name
    ]
    if len(matches) != 1:
        raise MultibucketPlanError(f"plan must contain one {name} family")
    return matches[0]


def _selected_ids(
    family: dict[str, Any], allowed_modes: set[str], bucket: int,
) -> dict[str, list[int]]:
    operators = family.get("operators")
    if not isinstance(operators, list) or not operators:
        raise MultibucketPlanError(
            f"bucket {bucket} family {family.get('family')} has no operators"
        )
    selected: dict[str, list[int]] = {mode: [] for mode in allowed_modes}
    seen: set[int] = set()
    for row in operators:
        if not isinstance(row, dict):
            raise MultibucketPlanError(f"bucket {bucket} has invalid operator row")
        operator_id = int(row["operator_id"])
        mode = str(row["selected"])
        if operator_id in seen:
            raise MultibucketPlanError(
                f"bucket {bucket} duplicates operator {operator_id}"
            )
        if mode not in allowed_modes:
            raise MultibucketPlanError(
                f"bucket {bucket} selects unsupported mode {mode}"
            )
        modes = row.get("modes")
        selected_result = modes.get(mode) if isinstance(modes, dict) else None
        if not isinstance(selected_result, dict) or (
            selected_result.get("bitwise") is not True
        ):
            raise MultibucketPlanError(
                f"bucket {bucket} operator {operator_id} selected mode is not bitwise-valid"
            )
        seen.add(operator_id)
        selected[mode].append(operator_id)
    for ids in selected.values():
        ids.sort()
    return selected


def load_bucket_plan(bucket: int, path: Path) -> dict[str, Any]:
    document = _load_json(path)
    declared_bucket = document.get("bucket_frames")
    if declared_bucket is not None and int(declared_bucket) != bucket:
        raise MultibucketPlanError(
            f"plan bucket mismatch: argument={bucket}, document={declared_bucket}"
        )
    rule = document.get("selection_rule")
    correctness = rule.get("correctness") if isinstance(rule, dict) else None
    if not isinstance(correctness, str) or "bitwise" not in correctness:
        raise MultibucketPlanError(
            f"bucket {bucket} plan does not declare a bitwise selection gate"
        )
    qconv = _selected_ids(
        _family(document, "qlinear_conv"), QCONV_MODES, bucket
    )
    fused = _selected_ids(
        _family(document, "fused_quant_qconv"), FUSED_MODES, bucket
    )
    return {
        "bucket_frames": bucket,
        "path": path,
        "qconv": qconv,
        "fused": fused,
    }


def _symbol(prefix: str, bucket: int) -> str:
    return f"CAMPP_{prefix}_{bucket}"


def _format_id_array(symbol: str, ids: Sequence[int]) -> str:
    lines = [f"static const uint32_t {symbol}[] = {{"]
    row: list[str] = []
    for operator_id in ids:
        row.append(f"{operator_id}u")
        if len(row) == 10:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    lines.append("};")
    return "\n".join(lines)


def _table_ref(symbol: str, ids: Sequence[int]) -> tuple[str, str]:
    if not ids:
        return "NULL", "0u"
    return symbol, f"sizeof({symbol}) / sizeof({symbol}[0])"


def render_source(plans: Sequence[dict[str, Any]]) -> str:
    if not plans:
        raise MultibucketPlanError("at least one bucket plan is required")
    ordered = sorted(plans, key=lambda item: int(item["bucket_frames"]))
    buckets = [int(item["bucket_frames"]) for item in ordered]
    if len(set(buckets)) != len(buckets):
        raise MultibucketPlanError("duplicate bucket plan")

    sections = [
        '#include "conv_layer_hybrid_plan.h"',
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "/* Generated by 17_generate_multibucket_v3_source.py. */",
    ]
    for plan in ordered:
        bucket = int(plan["bucket_frames"])
        arrays = (
            ("QCONV_MAC_FIXED_IDS", plan["qconv"]["mac_fixed"]),
            ("QCONV_V5_IDS", plan["qconv"]["v5"]),
            ("FUSED_QCONV_FIXED_IDS", plan["fused"]["combined_fixed"]),
            ("FUSED_QCONV_V5_IDS", plan["fused"]["combined_v5"]),
        )
        for prefix, ids in arrays:
            if ids:
                sections.extend(("", _format_id_array(_symbol(prefix, bucket), ids)))

    sections.extend((
        "",
        "typedef struct {",
        "    uint32_t bucket_frames;",
        "    const uint32_t *qconv_mac_fixed_ids;",
        "    size_t qconv_mac_fixed_count;",
        "    const uint32_t *qconv_v5_ids;",
        "    size_t qconv_v5_count;",
        "    const uint32_t *fused_fixed_ids;",
        "    size_t fused_fixed_count;",
        "    const uint32_t *fused_v5_ids;",
        "    size_t fused_v5_count;",
        "} CamppConvLayerHybridBucketPlan;",
        "",
        "static const CamppConvLayerHybridBucketPlan CAMPP_BUCKET_PLANS[] = {",
    ))
    for plan in ordered:
        bucket = int(plan["bucket_frames"])
        qfixed = _table_ref(
            _symbol("QCONV_MAC_FIXED_IDS", bucket),
            plan["qconv"]["mac_fixed"],
        )
        qv5 = _table_ref(
            _symbol("QCONV_V5_IDS", bucket), plan["qconv"]["v5"]
        )
        ffixed = _table_ref(
            _symbol("FUSED_QCONV_FIXED_IDS", bucket),
            plan["fused"]["combined_fixed"],
        )
        fv5 = _table_ref(
            _symbol("FUSED_QCONV_V5_IDS", bucket),
            plan["fused"]["combined_v5"],
        )
        sections.extend((
            "    {",
            f"        {bucket}u,",
            f"        {qfixed[0]}, {qfixed[1]},",
            f"        {qv5[0]}, {qv5[1]},",
            f"        {ffixed[0]}, {ffixed[1]},",
            f"        {fv5[0]}, {fv5[1]},",
            "    },",
        ))
    sections.extend((
        "};",
        "",
        "static int campp_id_in_sorted_table(",
        "    uint32_t operator_id, const uint32_t *ids, size_t count)",
        "{",
        "    size_t lower = 0u;",
        "    size_t upper = count;",
        "",
        "    while (lower < upper) {",
        "        const size_t middle = lower + (upper - lower) / 2u;",
        "        if (ids[middle] < operator_id) {",
        "            lower = middle + 1u;",
        "        } else {",
        "            upper = middle;",
        "        }",
        "    }",
        "    return lower < count && ids[lower] == operator_id;",
        "}",
        "",
        "static const CamppConvLayerHybridBucketPlan *campp_find_bucket_plan(",
        "    uint32_t bucket_frames)",
        "{",
        "    size_t index;",
        "    for (index = 0u; index < sizeof(CAMPP_BUCKET_PLANS) /",
        "             sizeof(CAMPP_BUCKET_PLANS[0]); ++index) {",
        "        if (CAMPP_BUCKET_PLANS[index].bucket_frames == bucket_frames) {",
        "            return &CAMPP_BUCKET_PLANS[index];",
        "        }",
        "    }",
        "    return NULL;",
        "}",
        "",
        "int campp_conv_layer_hybrid_has_bucket_plan(uint32_t bucket_frames)",
        "{",
        "    return campp_find_bucket_plan(bucket_frames) != NULL;",
        "}",
        "",
        "size_t campp_conv_layer_hybrid_bucket_plan_count(void)",
        "{",
        "    return sizeof(CAMPP_BUCKET_PLANS) / sizeof(CAMPP_BUCKET_PLANS[0]);",
        "}",
        "",
        "uint32_t campp_conv_layer_hybrid_bucket_plan_at(size_t index)",
        "{",
        "    return index < campp_conv_layer_hybrid_bucket_plan_count()",
        "        ? CAMPP_BUCKET_PLANS[index].bucket_frames",
        "        : 0u;",
        "}",
        "",
        "CamppQconvCandidateMode campp_conv_layer_hybrid_select_qconv(",
        "    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op)",
        "{",
        "    const CamppConvLayerHybridBucketPlan *plan;",
        "    if (model == NULL || op == NULL) {",
        "        return CAMPP_QCONV_CANDIDATE_V4;",
        "    }",
        "    plan = campp_find_bucket_plan(model->bucket_frames);",
        "    if (plan == NULL) {",
        "        return CAMPP_QCONV_CANDIDATE_V4;",
        "    }",
        "    if (campp_id_in_sorted_table(op->operator_id,",
        "            plan->qconv_mac_fixed_ids, plan->qconv_mac_fixed_count)) {",
        "        return CAMPP_QCONV_CANDIDATE_MAC_FIXED;",
        "    }",
        "    if (campp_id_in_sorted_table(op->operator_id,",
        "            plan->qconv_v5_ids, plan->qconv_v5_count)) {",
        "        return CAMPP_QCONV_CANDIDATE_V5;",
        "    }",
        "    return CAMPP_QCONV_CANDIDATE_V4;",
        "}",
        "",
        "CamppFusedQconvCandidateMode campp_conv_layer_hybrid_select_fused_qconv(",
        "    const CamppRuntimeModel *model, const CamppOperatorDescriptor *op)",
        "{",
        "    const CamppConvLayerHybridBucketPlan *plan;",
        "    if (model == NULL || op == NULL) {",
        "        return CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_HYBRID;",
        "    }",
        "    plan = campp_find_bucket_plan(model->bucket_frames);",
        "    if (plan == NULL) {",
        "        return CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_HYBRID;",
        "    }",
        "    if (campp_id_in_sorted_table(op->operator_id,",
        "            plan->fused_fixed_ids, plan->fused_fixed_count)) {",
        "        return CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_FIXED;",
        "    }",
        "    if (campp_id_in_sorted_table(op->operator_id,",
        "            plan->fused_v5_ids, plan->fused_v5_count)) {",
        "        return CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_V5;",
        "    }",
        "    return CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_HYBRID;",
        "}",
    ))
    return "\n".join(sections) + "\n"


def build_manifest(plans: Sequence[dict[str, Any]], source: Path) -> dict[str, Any]:
    rows = []
    for plan in sorted(plans, key=lambda item: int(item["bucket_frames"])):
        rows.append({
            "bucket_frames": int(plan["bucket_frames"]),
            "measured_plan": _display_path(plan["path"]),
            "selection_counts": {
                "qconv_mac_fixed": len(plan["qconv"]["mac_fixed"]),
                "qconv_v4": len(plan["qconv"]["v4"]),
                "qconv_v5": len(plan["qconv"]["v5"]),
                "fused_combined_fixed": len(
                    plan["fused"]["combined_fixed"]
                ),
                "fused_combined_hybrid": len(
                    plan["fused"]["combined_hybrid"]
                ),
                "fused_combined_v5": len(plan["fused"]["combined_v5"]),
            },
        })
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "bucket-aware measured V3 convolution dispatch",
        "fallback": {
            "qlinear_conv": "v4",
            "fused_quant_qconv": "combined_hybrid",
        },
        "source": _display_path(source),
        "buckets": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", action="append", type=parse_plan_spec, required=True,
        metavar="BUCKET=PATH",
    )
    parser.add_argument("--source", type=_path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=_path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        specs = args.plan
        buckets = [bucket for bucket, _ in specs]
        if len(set(buckets)) != len(buckets):
            raise MultibucketPlanError("duplicate --plan bucket")
        plans = [load_bucket_plan(bucket, path) for bucket, path in specs]
        source_text = render_source(plans)
        manifest = build_manifest(plans, args.source)
        if args.check_only:
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0
        existing = [path for path in (args.source, args.manifest) if path.exists()]
        if existing and not args.force:
            raise MultibucketPlanError(
                f"output exists: {existing[0]} (use --force)"
            )
        args.source.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.source.write_text(source_text, encoding="utf-8", newline="\n")
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        print(f"multibucket V3 source: {_display_path(args.source)}")
        print(f"multibucket V3 manifest: {_display_path(args.manifest)}")
        return 0
    except (MultibucketPlanError, OSError, ValueError, KeyError) as exc:
        print(f"multibucket V3 generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
