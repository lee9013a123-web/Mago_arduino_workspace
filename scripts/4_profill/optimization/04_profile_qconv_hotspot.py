#!/usr/bin/env python3
"""QConv 3x3/1x1의 target-only cycle을 address/MAC/requant로 세분화한다."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
DIAGNOSIS_SCRIPT = Path(__file__).with_name("02_diagnose_top4.py")
COMMON_SCRIPT = Path(__file__).with_name("_perf_hotspot_common.py")
QCONV_SOURCE = (
    ROOT
    / "src/c/runtime/backends/cpu_aarch64/int8_neon/qlinear_convolution_neon.c"
)
REQUANT_SOURCE = (
    ROOT
    / "src/c/runtime/backends/cpu_aarch64/int8_neon/requantization_neon.c"
)
TARGET_CASE_NAMES = ("qconv_3x3", "qconv_1x1")
# MAC v2 lives in the candidate tree, so profiling it needs its own source map:
# the production classifier would drop nearly every v2 sample as unclassified.
V2_MAC_SOURCE = (
    ROOT
    / "src/c/profill/optimization/candidates/qlinear_conv/qconv_mac_neon.c"
)
V2_CANDIDATE_SOURCE = (
    ROOT
    / "src/c/profill/optimization/candidates/qlinear_conv/qconv_candidate.c"
)
V2_ADDRESS_SOURCE = (
    ROOT
    / "src/c/profill/optimization/candidates/qlinear_conv"
    / "qconv_address_fastpath.c"
)
FIXED_MAC_SOURCE = (
    ROOT
    / "src/c/profill/optimization/candidates/qlinear_conv/microkernels"
    / "qconv_mac_4x8_intrinsics.c"
)
FIXED_DISPATCH_SOURCE = (
    ROOT
    / "src/c/profill/optimization/candidates/qlinear_conv/microkernels"
    / "qconv_mac_4x8.c"
)
FIXED_ASSEMBLY_SOURCE = (
    ROOT
    / "src/c/profill/optimization/candidates/qlinear_conv/microkernels"
    / "qconv_mac_4x8_aarch64.S"
)
FIXED_INTRINSICS_SYMBOL = "campp_qconv_mac_4x8_intrinsics_raw"
FIXED_ASSEMBLY_SYMBOL = "campp_qconv_mac_4x8_aarch64_raw"
V2_MAC_SYMBOL = "campp_qconv_mac_neon_tile_v2"
PRODUCTION_MAC_SYMBOL = "campp_aarch64_qlinear_conv_o4i4"
FUSED_QUANTIZE_SYMBOL = "campp_fused_input_quantize_neon"
CATEGORIES = (
    "address_load_control",
    "mac_reduction",
    "requant_write",
    "setup_other",
    "foreign_symbol",
    "unclassified",
)
V2_CATEGORIES = (
    "v2_input_address",
    "v2_weight_transform",
    "v2_mac_smlal",
    "v2_requant",
    "v2_output_store",
    "setup_other",
    "foreign_symbol",
    "unclassified",
)
V2_CORE_CATEGORIES = (
    "v2_input_address",
    "v2_weight_transform",
    "v2_mac_smlal",
    "v2_requant",
    "v2_output_store",
)
V2_DETAIL_CATEGORIES = (
    "address_output_coordinates",
    "address_kernel_coordinates",
    "address_padding_bounds",
    "address_offset",
    "address_point_table",
    "address_input_load_transform",
    "mac_dispatch_guard",
    "mac_weight_load_transform",
    "mac_accumulate",
    "mac_accumulator_store",
    "mac_bias_tail",
    "requant_parameter_load",
    "requant_dispatch",
    "requant_scale_multiply",
    "requant_round_clamp_narrow",
    "requant_output_address",
    "requant_tail_control",
    "requant_output_store",
    "requant_scalar_fallback",
    "setup_other",
    "foreign_symbol",
    "unclassified",
)
V2_DETAIL_AREAS = ("address", "mac", "requant", "setup", "foreign", "unclassified")
V2_DETAIL_PARENT = {
    "address_output_coordinates": "v2_input_address",
    "address_kernel_coordinates": "v2_input_address",
    "address_padding_bounds": "v2_input_address",
    "address_offset": "v2_input_address",
    "address_point_table": "v2_input_address",
    "address_input_load_transform": "v2_input_address",
    "mac_dispatch_guard": "setup_other",
    "mac_weight_load_transform": "v2_weight_transform",
    "mac_accumulate": "v2_mac_smlal",
    "mac_accumulator_store": "v2_mac_smlal",
    "mac_bias_tail": "v2_mac_smlal",
    "requant_parameter_load": "v2_requant",
    "requant_dispatch": "v2_requant",
    "requant_scale_multiply": "v2_requant",
    "requant_round_clamp_narrow": "v2_requant",
    "requant_output_address": "v2_output_store",
    "requant_tail_control": "v2_output_store",
    "requant_output_store": "v2_output_store",
    "requant_scalar_fallback": "v2_requant",
    "setup_other": "setup_other",
    "foreign_symbol": "foreign_symbol",
    "unclassified": "unclassified",
}


def _detail_area(category: str) -> str:
    if category.startswith("address_"):
        return "address"
    if category.startswith("mac_"):
        return "mac"
    if category.startswith("requant_"):
        return "requant"
    return {
        "setup_other": "setup",
        "foreign_symbol": "foreign",
        "unclassified": "unclassified",
    }[category]


# Second line of defence behind the perf control FIFO: samples raised by another
# kernel or by the measurement harness must never reach a QConv bucket. Inlined
# headers report the *enclosing* symbol, so matching on the symbol is what keeps
# foreign work out of address_load_control / mac_reduction.
FOREIGN_SYMBOL_TOKENS = (
    "sha256",
    "runtime_model_load",
    "fused_bn_relu_quant",
)
PERF_LINE = re.compile(
    r"^\s*(?P<period>\d+)\s+"
    r"(?P<ip>(?:0x)?[0-9a-fA-F]+)\s+"
    r"(?P<symbol>\S+)\s+"
    r"(?P<source>.*):(?P<line>\d+)\s*$"
)
PERF_HEADER_LINE = re.compile(
    r"^\s*(?P<period>\d+)\s+"
    r"(?P<ip>(?:0x)?[0-9a-fA-F]+)\s+"
    r"(?P<symbol>\S+)\s*$"
)
PERF_SRCLINE_LINE = re.compile(r"^\s+(?P<source>\S.*):(?P<line>\d+)\s*$")


class HotspotError(RuntimeError):
    """QConv sampling 입력, perf 실행 또는 결과가 유효하지 않다."""


def _load_diagnosis_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "e7_optimization_diagnosis", DIAGNOSIS_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise HotspotError(f"cannot import {DIAGNOSIS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_common_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "e7_perf_hotspot_common_qconv", COMMON_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise HotspotError(f"cannot import {COMMON_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIAGNOSIS = _load_diagnosis_module()
COMMON = _load_common_module()


def select_qconv_cases(
    operators: Sequence[dict[str, Any]], *, case_scope: str
) -> list[dict[str, Any]]:
    """Select one representative per shape or every ordinary QConv operator."""

    if case_scope == "representative":
        selected = DIAGNOSIS.select_representative_cases(operators)
        cases = [
            dict(case)
            for case in selected
            if case["case_name"] in TARGET_CASE_NAMES
        ]
        for case in cases:
            case["shape"] = case["case_name"].removeprefix("qconv_")
        return cases
    if case_scope != "all":
        raise HotspotError(f"unsupported case scope: {case_scope}")

    cases = []
    for operator in operators:
        if operator.get("kernel_name") != "qlinear_conv_o4i4_neon":
            continue
        weight_shape = DIAGNOSIS._meaningful_weight_shape(operator)
        kernel_shape = weight_shape[2:]
        if kernel_shape == (3, 3):
            shape = "3x3"
        elif kernel_shape and all(value == 1 for value in kernel_shape):
            shape = "1x1"
        else:
            continue
        operator_id = int(operator["operator_id"])
        cases.append(
            {
                "case_name": f"qconv_{shape}_op_{operator_id}",
                "shape": shape,
                "operator_id": operator_id,
                "kernel_id": int(operator["kernel_id"]),
                "kernel_name": str(operator["kernel_name"]),
                "operator_type": str(operator["operator_type"]),
                "weight_shape": list(weight_shape),
                "profile_mean_ms": float(operator["mean_ms"]),
                "profile_share_pct": float(operator["end_to_end_share_pct"]),
            }
        )
    cases.sort(
        key=lambda case: (
            0 if case["shape"] == "3x3" else 1,
            -float(case["profile_mean_ms"]),
            int(case["operator_id"]),
        )
    )
    shapes = {case["shape"] for case in cases}
    if shapes != {"3x3", "1x1"}:
        raise HotspotError("profile does not contain both 3x3 and 1x1 QConv")
    return cases


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _function_span(lines: Sequence[str], signature: str) -> tuple[int, int]:
    start_index = next(
        (index for index, line in enumerate(lines) if signature in line), None
    )
    if start_index is None:
        raise HotspotError(f"source signature not found: {signature}")
    depth = 0
    opened = False
    for index in range(start_index, len(lines)):
        depth += lines[index].count("{")
        if "{" in lines[index]:
            opened = True
        depth -= lines[index].count("}")
        if opened and depth == 0:
            return start_index + 1, index + 1
    raise HotspotError(f"unterminated C function: {signature}")


def _find_line(
    lines: Sequence[str], text: str, *, start: int = 1, reverse: bool = False
) -> int:
    indices = range(start - 1, len(lines))
    if reverse:
        indices = range(len(lines) - 1, start - 2, -1)
    for index in indices:
        if text in lines[index]:
            return index + 1
    raise HotspotError(f"source marker not found: {text}")


def _marker_span(
    lines: Sequence[str], begin: str, end: str, *, start: int = 1
) -> tuple[int, int]:
    begin_line = _find_line(lines, begin, start=start)
    return begin_line, _find_line(lines, end, start=begin_line)


def build_v2_source_map(
    mac_source: Path = V2_MAC_SOURCE,
    candidate_source: Path = V2_CANDIDATE_SOURCE,
    address_source: Path = V2_ADDRESS_SOURCE,
    requant_source: Path = REQUANT_SOURCE,
    fixed_mac_source: Path = FIXED_MAC_SOURCE,
    fixed_dispatch_source: Path = FIXED_DISPATCH_SOURCE,
    fixed_assembly_source: Path = FIXED_ASSEMBLY_SOURCE,
) -> dict[str, Any]:
    """Function spans for the MAC v2 candidate.

    The v2 helpers are `static` and get inlined, so perf reports the enclosing
    symbol. Source-line spans are therefore what separates weight transform,
    SMLAL body, requant and store. Register spill cannot be split this way at
    all -- it is instruction level, so it is measured from `perf annotate`.
    """
    mac_lines = mac_source.read_text(encoding="utf-8").splitlines()
    cand_lines = candidate_source.read_text(encoding="utf-8").splitlines()
    address_lines = address_source.read_text(encoding="utf-8").splitlines()
    requant_lines = requant_source.read_text(encoding="utf-8").splitlines()
    fixed_lines = fixed_mac_source.read_text(encoding="utf-8").splitlines()
    fixed_dispatch_lines = fixed_dispatch_source.read_text(
        encoding="utf-8"
    ).splitlines()
    assembly_lines = fixed_assembly_source.read_text(
        encoding="utf-8"
    ).splitlines()

    cand_run_v2_span = _function_span(
        cand_lines, "static CamppStatus campp_qconv_candidate_run_v2("
    )
    cand_address_output_span = _marker_span(
        cand_lines,
        "campp_qconv_address_output_tile(",
        "output_coordinates);",
        start=cand_run_v2_span[0],
    )
    cand_kernel_coordinates_span = _marker_span(
        cand_lines,
        "campp_qconv_candidate_spatial_coordinates(",
        "kernel_coordinates);",
        start=cand_run_v2_span[0],
    )
    cand_point_table_span = _marker_span(
        cand_lines,
        "input_points[kernel_index][tile] =",
        "mode == CAMPP_QCONV_CANDIDATE_COMBINED);",
        start=cand_run_v2_span[0],
    )
    cand_mac_dispatch_span = _marker_span(
        cand_lines,
        "if (mode == CAMPP_QCONV_CANDIDATE_MAC_FIXED ||",
        "CAMPP_OPT_STAGE_QCONV_MAC_ADDRESS,",
        start=cand_run_v2_span[0],
    )
    cand_requant_dispatch_span = _marker_span(
        cand_lines,
        "CAMPP_OPTIMIZATION_STAGE_BEGIN(requant_started_ns)",
        "requant_started_ns);",
        start=cand_run_v2_span[0],
    )

    address_input_span = _function_span(
        address_lines, "const uint8_t *campp_qconv_address_input_base("
    )
    address_coordinate_begin = _find_line(
        address_lines,
        "for (axis = 0u; axis < plan->spatial_rank; ++axis)",
        start=address_input_span[0],
    )
    address_direct_begin = _find_line(
        address_lines, "if (direct_offset)", start=address_coordinate_begin
    )
    address_check_begin = _find_line(
        address_lines,
        "if (offset >= plan->input->storage_span_bytes)",
        start=address_direct_begin,
    )

    fixed_main_span = _function_span(
        fixed_lines, "int campp_qconv_mac_4x8_intrinsics_raw("
    )
    fixed_weight_macro_begin = _find_line(
        fixed_lines, "#define CAMPP_QCONV_LOAD_COLUMNS"
    )
    fixed_accumulate_macro_begin = _find_line(
        fixed_lines, "#define CAMPP_QCONV_ACCUMULATE"
    )
    fixed_weight_calls = [
        index + 1
        for index, line in enumerate(fixed_lines)
        if fixed_main_span[0] <= index + 1 <= fixed_main_span[1]
        and "CAMPP_QCONV_LOAD_COLUMNS(" in line
    ]
    fixed_accumulate_calls = [
        index + 1
        for index, line in enumerate(fixed_lines)
        if fixed_main_span[0] <= index + 1 <= fixed_main_span[1]
        and "CAMPP_QCONV_ACCUMULATE(" in line
    ]

    half_tile_v2_span = _function_span(
        mac_lines, "static int campp_qconv_mac_neon_half_tile_v2("
    )
    half_store_begin = _find_line(
        mac_lines, "vst1q_s32(output[tile_offset]", start=half_tile_v2_span[0]
    )
    half_bias_begin = _find_line(
        mac_lines,
        "for (tile = tile_offset; tile < tile_offset + 4u; ++tile)",
        start=half_store_begin,
    )
    tile_v2_span = _function_span(
        mac_lines, "int campp_qconv_mac_neon_tile_v2("
    )
    tile_tail_store_begin = _find_line(
        mac_lines,
        "vst1q_s32(accumulators[tile]",
        start=tile_v2_span[0],
    )
    tile_tail_bias_begin = _find_line(
        mac_lines,
        "const int64_t value =",
        start=tile_tail_store_begin,
    )

    scalar_requant_span = _function_span(
        requant_lines, "static CamppStatus campp_qconv_requantize_scalar("
    )
    neon_requant_span = _function_span(
        requant_lines, "static uint32_t campp_qconv_requantize_neon4("
    )
    store4_span = _function_span(
        requant_lines, "CamppStatus campp_aarch64_qconv_requantize_store4("
    )
    requant_direct_lookup_span = _marker_span(
        requant_lines,
        "if (campp_qconv_channel_store_offset(",
        "&direct_offset, &channel_capacity))",
        start=store4_span[0],
    )
    requant_vector_call_span = _marker_span(
        requant_lines,
        "#if defined(__aarch64__) && defined(__ARM_NEON)",
        "#endif",
        start=requant_direct_lookup_span[1],
    )
    requant_tail_span = _marker_span(
        requant_lines,
        "A non-aliased channel-packed tensor owns its padded channel block",
        "return CAMPP_STATUS_BUFFER_OVERFLOW;",
        start=requant_vector_call_span[1],
    )
    requant_fallback_begin = _find_line(
        requant_lines,
        "for (lane = 0u; lane < valid_outputs; ++lane)",
        start=requant_tail_span[1],
    )

    requant_tile_span = _function_span(
        mac_lines, "int campp_qconv_requantize_store_neon_tile("
    )
    requant_tile_address_span = _marker_span(
        mac_lines,
        "const uint32_t spatial_index =",
        "output_coordinates[tile][1];",
        start=requant_tile_span[0],
    )
    requant_tile_tail_span = _marker_span(
        mac_lines,
        "for (output_block = 0u;",
        "if (lanes == 0u) continue;",
        start=requant_tile_address_span[1],
    )
    requant_tile_call_span = _marker_span(
        mac_lines,
        "if (campp_aarch64_qconv_requantize_store4(",
        "CAMPP_STATUS_OK)",
        start=requant_tile_tail_span[1],
    )

    assembly_function_span = _marker_span(
        assembly_lines,
        "campp_qconv_mac_4x8_aarch64_raw:",
        ".size campp_qconv_mac_4x8_aarch64_raw",
    )
    return {
        "mac_source_name": mac_source.name,
        "candidate_source_name": candidate_source.name,
        "address_source_name": address_source.name,
        "requant_source_name": requant_source.name,
        "fixed_mac_source_name": fixed_mac_source.name,
        "fixed_dispatch_source_name": fixed_dispatch_source.name,
        "fixed_assembly_source_name": fixed_assembly_source.name,
        "mac_line_count": len(mac_lines),
        "candidate_line_count": len(cand_lines),
        "address_line_count": len(address_lines),
        "requant_line_count": len(requant_lines),
        "fixed_mac_line_count": len(fixed_lines),
        "fixed_dispatch_line_count": len(fixed_dispatch_lines),
        "fixed_assembly_line_count": len(assembly_lines),
        # qconv_mac_neon.c spans
        "neon_input_span": _function_span(
            mac_lines, "static int16x4_t campp_qconv_neon_input("
        ),
        "dot_input_span": _function_span(
            mac_lines, "static int8x8_t campp_qconv_dot_input("
        ),
        "weight_columns_span": _function_span(
            mac_lines, "static void campp_qconv_neon_weight_columns("
        ),
        "dot_weight_span": _function_span(
            mac_lines, "static int8x16_t campp_qconv_dot_weight("
        ),
        "dot_weight_zero_span": _function_span(
            mac_lines, "static int32x4_t campp_qconv_dot_weight_zero("
        ),
        "dot_accumulate_span": _function_span(
            mac_lines, "static int32x4_t campp_qconv_dot_accumulate("
        ),
        "smlal_accumulate_span": _function_span(
            mac_lines, "static int32x4_t campp_qconv_smlal_accumulate("
        ),
        "half_tile_v2_span": half_tile_v2_span,
        "half_tile_store_span": (half_store_begin, half_bias_begin - 1),
        "half_tile_bias_span": (half_bias_begin, half_tile_v2_span[1]),
        "scalar_tile_v2_span": _function_span(
            mac_lines, "static int campp_qconv_mac_scalar_tile_v2("
        ),
        "tile_v2_span": tile_v2_span,
        "tile_tail_store_span": (
            tile_tail_store_begin,
            tile_tail_bias_begin - 1,
        ),
        "tile_tail_bias_span": (tile_tail_bias_begin, tile_v2_span[1]),
        "requantize_neon4_span": neon_requant_span,
        "requantize_neon4_scale_span": _marker_span(
            requant_lines,
            "float32x4_t scaled = vmulq_f32(",
            "vld1q_f32(multipliers));",
            start=neon_requant_span[0],
        ),
        "requantize_scalar_span": scalar_requant_span,
        "requantize_scalar_scale_span": _marker_span(
            requant_lines,
            "scaled = (float)accumulator * multiplier;",
            "scaled = (float)accumulator * multiplier;",
            start=scalar_requant_span[0],
        ),
        "requantize_store4_span": store4_span,
        "requantize_channel_offset_span": _function_span(
            requant_lines, "static bool campp_qconv_channel_store_offset("
        ),
        "requantize_direct_lookup_span": requant_direct_lookup_span,
        "requantize_vector_call_span": requant_vector_call_span,
        "requantize_tail_span": requant_tail_span,
        "requantize_direct_store_span": _marker_span(
            requant_lines,
            "memcpy((uint8_t *)output->data + direct_offset",
            "return CAMPP_STATUS_OK;",
            start=requant_tail_span[1],
        ),
        "requantize_fallback_span": (
            requant_fallback_begin,
            store4_span[1],
        ),
        "requantize_store_span": requant_tile_span,
        "requantize_tile_address_span": requant_tile_address_span,
        "requantize_tile_tail_span": requant_tile_tail_span,
        "requantize_tile_call_span": requant_tile_call_span,
        "candidate_read_span": _function_span(
            mac_lines, "static int32_t campp_qconv_candidate_read("
        ),
        # qconv_candidate.c spans
        "cand_spatial_span": _function_span(
            cand_lines, "static void campp_qconv_candidate_spatial_coordinates("
        ),
        "cand_write_span": _function_span(
            cand_lines, "static CamppStatus campp_qconv_candidate_write("
        ),
        "cand_prepare_span": _function_span(
            cand_lines, "static CamppStatus campp_qconv_candidate_prepare("
        ),
        "cand_load_parameters_span": _function_span(
            cand_lines, "static CamppStatus campp_qconv_candidate_load_parameters("
        ),
        "cand_run_v2_span": cand_run_v2_span,
        "cand_address_output_span": cand_address_output_span,
        "cand_kernel_coordinates_span": cand_kernel_coordinates_span,
        "cand_point_table_span": cand_point_table_span,
        "cand_mac_dispatch_span": cand_mac_dispatch_span,
        "cand_requant_dispatch_span": cand_requant_dispatch_span,
        # qconv_address_fastpath.c spans
        "address_plan_span": _function_span(
            address_lines, "bool campp_qconv_address_plan_create("
        ),
        "address_output_tile_span": _function_span(
            address_lines, "void campp_qconv_address_output_tile("
        ),
        "address_input_span": address_input_span,
        "address_padding_span": (
            address_coordinate_begin,
            address_direct_begin - 1,
        ),
        "address_offset_span": (address_direct_begin, address_input_span[1]),
        "address_offset_check_span": (
            address_check_begin,
            address_input_span[1],
        ),
        # qconv_mac_4x8_intrinsics.c spans
        "fixed_load_input_span": _function_span(
            fixed_lines, "static inline int16x4_t campp_qconv_mac_4x8_load_input("
        ),
        "fixed_weight_macro_span": (
            fixed_weight_macro_begin,
            fixed_accumulate_macro_begin - 1,
        ),
        "fixed_accumulate_macro_span": _marker_span(
            fixed_lines, "#define CAMPP_QCONV_ACCUMULATE", "} while (0)"
        ),
        "fixed_main_span": fixed_main_span,
        "fixed_input_calls_span": _marker_span(
            fixed_lines,
            "const int16x4_t input0 =",
            "points[3] + channel, input_zero);",
            start=fixed_main_span[0],
        ),
        "fixed_weight_call_lines": fixed_weight_calls,
        "fixed_accumulate_call_lines": fixed_accumulate_calls,
        "fixed_accumulator_store_span": _marker_span(
            fixed_lines,
            "vst1q_s32(params->output, accumulator00);",
            "vst1q_s32(params->output + 28u, accumulator13);",
            start=fixed_main_span[0],
        ),
        # qconv_mac_4x8.c spans
        "fixed_support_span": _function_span(
            fixed_dispatch_lines, "static int campp_qconv_mac_4x8_is_supported("
        ),
        "fixed_try_tile_span": _function_span(
            fixed_dispatch_lines, "CamppQconvMac4x8Result campp_qconv_mac_4x8_try_tile("
        ),
        # qconv_mac_4x8_aarch64.S spans
        "assembly_weight_macro_span": _marker_span(
            assembly_lines,
            ".macro CAMPP_QCONV_LOAD_COLUMNS",
            ".endm",
        ),
        "assembly_accumulate_macro_span": _marker_span(
            assembly_lines,
            ".macro CAMPP_QCONV_ACCUMULATE",
            ".endm",
            start=_find_line(assembly_lines, ".macro CAMPP_QCONV_ACCUMULATE"),
        ),
        "assembly_function_span": assembly_function_span,
        "assembly_point_table_span": _marker_span(
            assembly_lines,
            ".Lcampp_qconv_4x8_kernel:",
            "mov w15, w7",
            start=assembly_function_span[0],
        ),
        "assembly_input_load_span": _marker_span(
            assembly_lines,
            ".Lcampp_qconv_4x8_input_block:",
            "sub v27.4h, v27.4h, v31.4h",
            start=assembly_function_span[0],
        ),
        "assembly_weight_call_lines": [
            index + 1
            for index, line in enumerate(assembly_lines)
            if index + 1 >= assembly_function_span[0]
            and "CAMPP_QCONV_LOAD_COLUMNS" in line
        ],
        "assembly_accumulate_call_lines": [
            index + 1
            for index, line in enumerate(assembly_lines)
            if index + 1 >= assembly_function_span[0]
            and "CAMPP_QCONV_ACCUMULATE" in line
        ],
        "assembly_accumulator_store_span": _marker_span(
            assembly_lines,
            "stp q16, q20, [x5]",
            "stp q19, q23, [x5, #96]",
            start=assembly_function_span[0],
        ),
    }


def classify_v2_detail_sample(
    sample: dict[str, Any], source_map: dict[str, Any]
) -> str:
    symbol = str(sample["symbol"]).lower()
    source_name = Path(str(sample["source"])).name
    line = int(sample["line"])

    if any(token in symbol for token in FOREIGN_SYMBOL_TOKENS):
        return "foreign_symbol"
    # Production kernel means the shape fell back out of the candidate path.
    if "campp_aarch64_qlinear_conv_o4i4" in symbol:
        return "foreign_symbol"

    if source_name == source_map["address_source_name"]:
        if _inside(line, source_map["address_output_tile_span"]):
            return "address_output_coordinates"
        if _inside(line, source_map["address_padding_span"]):
            return "address_padding_bounds"
        if _inside(line, source_map["address_offset_span"]):
            return "address_offset"
        if _inside(line, source_map["address_plan_span"]):
            return "setup_other"
        return "unclassified"

    if source_name == source_map["fixed_mac_source_name"]:
        if _inside(line, source_map["fixed_load_input_span"]) or _inside(
            line, source_map["fixed_input_calls_span"]
        ):
            return "address_input_load_transform"
        if _inside(line, source_map["fixed_weight_macro_span"]) or line in (
            source_map["fixed_weight_call_lines"]
        ):
            return "mac_weight_load_transform"
        if _inside(line, source_map["fixed_accumulate_macro_span"]) or line in (
            source_map["fixed_accumulate_call_lines"]
        ):
            return "mac_accumulate"
        if _inside(line, source_map["fixed_accumulator_store_span"]):
            return "mac_accumulator_store"
        if _inside(line, source_map["fixed_main_span"]):
            return "mac_dispatch_guard"

    if source_name == source_map["fixed_assembly_source_name"]:
        if _inside(line, source_map["assembly_input_load_span"]):
            return "address_input_load_transform"
        if _inside(line, source_map["assembly_point_table_span"]):
            return "address_point_table"
        if _inside(line, source_map["assembly_weight_macro_span"]) or line in (
            source_map["assembly_weight_call_lines"]
        ):
            return "mac_weight_load_transform"
        if _inside(line, source_map["assembly_accumulate_macro_span"]) or line in (
            source_map["assembly_accumulate_call_lines"]
        ):
            return "mac_accumulate"
        if _inside(line, source_map["assembly_accumulator_store_span"]):
            return "mac_accumulator_store"
        if _inside(line, source_map["assembly_function_span"]):
            return "mac_dispatch_guard"

    if source_name == source_map["fixed_dispatch_source_name"]:
        if _inside(line, source_map["fixed_support_span"]) or _inside(
            line, source_map["fixed_try_tile_span"]
        ):
            return "mac_dispatch_guard"
        return "setup_other"

    if source_name == source_map["mac_source_name"]:
        for key, category in (
            ("weight_columns_span", "mac_weight_load_transform"),
            ("dot_weight_span", "mac_weight_load_transform"),
            ("dot_weight_zero_span", "mac_weight_load_transform"),
            ("neon_input_span", "address_input_load_transform"),
            ("dot_input_span", "address_input_load_transform"),
            ("candidate_read_span", "address_input_load_transform"),
            ("smlal_accumulate_span", "mac_accumulate"),
            ("dot_accumulate_span", "mac_accumulate"),
            ("half_tile_store_span", "mac_accumulator_store"),
            ("half_tile_bias_span", "mac_bias_tail"),
            ("tile_tail_store_span", "mac_accumulator_store"),
            ("tile_tail_bias_span", "mac_bias_tail"),
            ("requantize_tile_address_span", "requant_output_address"),
            ("requantize_tile_tail_span", "requant_tail_control"),
            ("requantize_tile_call_span", "requant_dispatch"),
        ):
            if _inside(line, source_map[key]):
                return category
        if _inside(line, source_map["requantize_store_span"]):
            return "requant_dispatch"
        if _inside(line, source_map["half_tile_v2_span"]) or _inside(
            line, source_map["scalar_tile_v2_span"]
        ):
            return "mac_accumulate"
        if _inside(line, source_map["tile_v2_span"]):
            return "mac_dispatch_guard"
        return "setup_other"

    if source_name == source_map["candidate_source_name"]:
        if _inside(line, source_map["cand_address_output_span"]):
            return "address_output_coordinates"
        if _inside(line, source_map["cand_kernel_coordinates_span"]):
            return "address_kernel_coordinates"
        if _inside(line, source_map["cand_point_table_span"]):
            return "address_point_table"
        if _inside(line, source_map["cand_mac_dispatch_span"]):
            return "mac_dispatch_guard"
        if _inside(line, source_map["cand_requant_dispatch_span"]):
            return "requant_dispatch"
        if _inside(line, source_map["cand_spatial_span"]):
            return "address_kernel_coordinates"
        if _inside(line, source_map["cand_write_span"]):
            return "requant_scalar_fallback"
        if _inside(line, source_map["cand_load_parameters_span"]):
            return "requant_parameter_load"
        if _inside(line, source_map["cand_prepare_span"]):
            return "setup_other"
        if _inside(line, source_map["cand_run_v2_span"]):
            return "setup_other"
        return "setup_other"

    if source_name == source_map["requant_source_name"]:
        if _inside(line, source_map["requantize_neon4_scale_span"]) or _inside(
            line, source_map["requantize_scalar_scale_span"]
        ):
            return "requant_scale_multiply"
        if _inside(line, source_map["requantize_neon4_span"]):
            return "requant_round_clamp_narrow"
        if _inside(line, source_map["requantize_scalar_span"]):
            return "requant_round_clamp_narrow"
        if _inside(line, source_map["requantize_channel_offset_span"]) or _inside(
            line, source_map["requantize_direct_lookup_span"]
        ):
            return "requant_output_address"
        if _inside(line, source_map["requantize_vector_call_span"]):
            return "requant_dispatch"
        if _inside(line, source_map["requantize_tail_span"]):
            return "requant_tail_control"
        if _inside(line, source_map["requantize_direct_store_span"]):
            return "requant_output_store"
        if _inside(line, source_map["requantize_fallback_span"]):
            return "requant_scalar_fallback"
        if _inside(line, source_map["requantize_store4_span"]):
            return "requant_dispatch"
        return "setup_other"

    # tensor_view.h / arm_neon.h inline into the candidate; attribute by intent.
    if source_name == "tensor_view.h":
        return "address_offset"
    if source_name == "arm_neon.h":
        if "requant" in symbol:
            return "requant_round_clamp_narrow"
        if "weight" in symbol:
            return "mac_weight_load_transform"
        if "input" in symbol:
            return "address_input_load_transform"
        return "mac_accumulate"
    if "campp_qconv_mac_4x8_" in symbol:
        return "mac_accumulate"
    if "campp_qconv_mac_neon_tile_v2" in symbol:
        return "mac_accumulate"
    if "campp_qconv_address_output_tile" in symbol:
        return "address_output_coordinates"
    if "campp_qconv_address_input_base" in symbol:
        return "address_offset"
    if "campp_qconv_requantize" in symbol or "nearbyint" in symbol:
        return "requant_round_clamp_narrow"
    return "unclassified"


def classify_v2_sample(
    sample: dict[str, Any], source_map: dict[str, Any]
) -> str:
    """Return the legacy top-level category for one detailed sample."""

    detail = classify_v2_detail_sample(sample, source_map)
    return V2_DETAIL_PARENT[detail]


def classify_v2_perf_script(
    text: str, source_map: dict[str, Any]
) -> dict[str, Any]:
    periods = {category: 0 for category in V2_CATEGORIES}
    counts = {category: 0 for category in V2_CATEGORIES}
    detail_periods = {category: 0 for category in V2_DETAIL_CATEGORIES}
    detail_counts = {category: 0 for category in V2_DETAIL_CATEGORIES}
    area_periods = {area: 0 for area in V2_DETAIL_AREAS}
    area_symbol_periods: dict[str, dict[str, int]] = {
        area: {} for area in V2_DETAIL_AREAS
    }
    line_periods: dict[tuple[str, int, str, str, str, str], int] = {}
    fixed_microkernel_sample_count = 0
    samples, malformed = COMMON.iter_perf_samples(text)
    for sample in samples:
        if "campp_qconv_mac_4x8_" in str(sample["symbol"]).lower():
            fixed_microkernel_sample_count += 1
        detail = classify_v2_detail_sample(sample, source_map)
        category = V2_DETAIL_PARENT[detail]
        area = _detail_area(detail)
        period = int(sample["period"])
        periods[category] += period
        counts[category] += 1
        detail_periods[detail] += period
        detail_counts[detail] += 1
        area_periods[area] += period
        symbol = str(sample["symbol"])
        area_symbol_periods[area][symbol] = (
            area_symbol_periods[area].get(symbol, 0) + period
        )
        key = (
            Path(str(sample["source"])).name,
            int(sample["line"]),
            symbol,
            category,
            detail,
            area,
        )
        line_periods[key] = line_periods.get(key, 0) + period
    total = sum(periods.values())
    if total == 0:
        raise HotspotError("perf script contains no attributable cycle samples")
    core = sum(periods[name] for name in V2_CORE_CATEGORIES)
    detailed_core = sum(area_periods[name] for name in ("address", "mac", "requant"))
    top = sorted(line_periods.items(), key=lambda item: item[1], reverse=True)
    return {
        "parsed_sample_count": sum(counts.values()),
        "fixed_microkernel_sample_count": fixed_microkernel_sample_count,
        "malformed_line_count": malformed,
        "total_sample_period": total,
        "category_sample_counts": counts,
        "category_periods": periods,
        "category_share_pct": {
            name: value / total * 100.0 for name, value in periods.items()
        },
        "classified_core_period": core,
        "classified_core_share_pct": core / total * 100.0 if total else 0.0,
        "detail_category_sample_counts": detail_counts,
        "detail_category_periods": detail_periods,
        "detail_category_share_pct": {
            name: value / total * 100.0
            for name, value in detail_periods.items()
        },
        "area_periods": area_periods,
        "area_share_pct": {
            name: value / total * 100.0 for name, value in area_periods.items()
        },
        "area_share_of_classified_pct": {
            name: area_periods[name] / detailed_core * 100.0
            if detailed_core
            else 0.0
            for name in ("address", "mac", "requant")
        },
        "top_symbols_by_area": {
            area: [
                {
                    "symbol": symbol,
                    "sample_period": period,
                    "area_share_pct": period / area_periods[area] * 100.0
                    if area_periods[area]
                    else 0.0,
                    "total_share_pct": period / total * 100.0,
                }
                for symbol, period in sorted(
                    symbol_periods.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
            ]
            for area, symbol_periods in area_symbol_periods.items()
        },
        "top_lines": [
            {
                "source": source,
                "line": line,
                "symbol": symbol,
                "category": category,
                "detail_category": detail,
                "area": area,
                "sample_period": period,
                "total_share_pct": period / total * 100.0,
            }
            for (source, line, symbol, category, detail, area), period in top[:40]
        ],
    }


def measure_spill_share(annotate_text: str) -> dict[str, Any]:
    """Quantify stack spill/reload traffic from `perf annotate` output.

    Source lines cannot separate spill from real work, so this reads the
    disassembly and sums the sampled percentage of ldr/str against [sp].
    """
    spill = 0.0
    vector_memory = 0.0
    total = 0.0
    for raw in annotate_text.splitlines():
        match = re.match(r"\s*(\d+\.\d+)\s*:?\s+[0-9a-f]+:\s+(\S+)\s+(.*)$", raw)
        if match is None:
            continue
        percent = float(match.group(1))
        mnemonic = match.group(2).lower()
        operands = match.group(3).lower()
        total += percent
        if mnemonic.startswith(("ldr", "str", "ldp", "stp")):
            if "[sp" in operands or "[x29" in operands:
                spill += percent
            else:
                vector_memory += percent
    return {
        "annotated_percent_total": total,
        "stack_spill_percent": spill,
        "other_memory_percent": vector_memory,
        "stack_spill_share_of_annotated_pct": (
            spill / total * 100.0 if total else 0.0
        ),
    }


def select_mac_annotate_target(
    perf_script_text: str, *, qconv_candidate: str = "baseline",
    fused_qconv_candidate: str = "baseline",
) -> dict[str, Any]:
    """Select the symbol that actually executed, including fixed fallback."""

    if qconv_candidate != "baseline" and fused_qconv_candidate != "baseline":
        raise HotspotError("QConv and fused QConv candidates are exclusive")
    samples, _ = COMMON.iter_perf_samples(perf_script_text)
    symbols = [str(sample["symbol"]).lower() for sample in samples]

    requested = qconv_candidate
    if requested == "hybrid":
        requested = "mac_fixed"
    if fused_qconv_candidate != "baseline":
        requested = {
            "mac": "mac",
            "combined": "combined",
            "mac_fixed": "mac_fixed",
            "combined_fixed": "mac_fixed",
            "combined_v4": "mac_fixed",
            "combined_hybrid": "mac_fixed",
            "quant_neon": "baseline",
        }.get(fused_qconv_candidate, "baseline")

    if requested == "mac_fixed":
        if any(FIXED_INTRINSICS_SYMBOL in symbol for symbol in symbols):
            return {
                "execution_path": "fixed_4x8_intrinsics",
                "symbol": FIXED_INTRINSICS_SYMBOL,
            }
        return {"execution_path": "fallback_v2", "symbol": V2_MAC_SYMBOL}
    if requested == "mac_asm":
        if any(FIXED_ASSEMBLY_SYMBOL in symbol for symbol in symbols):
            return {
                "execution_path": "fixed_4x8_assembly",
                "symbol": FIXED_ASSEMBLY_SYMBOL,
            }
        return {"execution_path": "fallback_v2", "symbol": V2_MAC_SYMBOL}
    if requested != "baseline":
        return {"execution_path": "v2", "symbol": V2_MAC_SYMBOL}
    return {
        "execution_path": "production",
        "symbol": PRODUCTION_MAC_SYMBOL,
    }


def build_source_map(
    source: Path = QCONV_SOURCE,
    requant_source: Path = REQUANT_SOURCE,
) -> dict[str, Any]:
    lines = source.read_text(encoding="utf-8").splitlines()
    requant_lines = requant_source.read_text(encoding="utf-8").splitlines()
    dot_span = _function_span(lines, "static int32_t campp_dot4_i16(")
    read_span = _function_span(lines, "static int32_t campp_read_qbyte(")
    spatial_span = _function_span(lines, "static void campp_spatial_coordinates(")
    requant_span = _function_span(
        requant_lines, "CamppStatus campp_aarch64_qconv_requantize_store4("
    )
    main_span = _function_span(lines, "CamppStatus campp_aarch64_qlinear_conv_o4i4(")
    core_begin = _find_line(
        lines, "for (kernel_index = 0u;", start=main_span[0]
    )
    mac_call = _find_line(
        lines, "accum[tile][output_lane] += campp_dot4_i16(",
        start=core_begin,
    )
    output_loops = [
        index + 1
        for index, line in enumerate(lines[:mac_call])
        if "for (output_lane = 0u;" in line and index + 1 >= core_begin
    ]
    if not output_loops:
        raise HotspotError("MAC output loop marker not found")
    mac_loop_begin = output_loops[-1]
    core_end = _find_line(
        lines,
        "CAMPP_OPT_STAGE_QCONV_MAC_ADDRESS, mac_started_ns",
        start=mac_call,
    )
    requant_begin = _find_line(
        lines, "CAMPP_OPTIMIZATION_STAGE_BEGIN(requant_started_ns)",
        start=core_end,
    )
    requant_end = _find_line(
        lines, "requant_started_ns);", start=requant_begin
    )
    return {
        "source": source,
        "source_name": source.name,
        "requant_source_name": requant_source.name,
        "line_count": len(lines),
        "dot_span": dot_span,
        "read_span": read_span,
        "spatial_span": spatial_span,
        "requant_span": requant_span,
        "main_span": main_span,
        "core_begin": core_begin,
        "mac_loop_begin": mac_loop_begin,
        "core_end": core_end,
        "requant_begin": requant_begin,
        "requant_end": requant_end,
    }


def parse_perf_script_line(line: str) -> dict[str, Any] | None:
    match = PERF_LINE.match(line)
    if match is None:
        return None
    return {
        "period": int(match.group("period")),
        "ip": match.group("ip"),
        "symbol": match.group("symbol"),
        "source": match.group("source"),
        "line": int(match.group("line")),
    }


def _parse_perf_header_line(line: str) -> dict[str, Any] | None:
    match = PERF_HEADER_LINE.match(line)
    if match is None:
        return None
    return {
        "period": int(match.group("period")),
        "ip": match.group("ip"),
        "symbol": match.group("symbol"),
    }


def _parse_perf_srcline_line(line: str) -> tuple[str, int] | None:
    match = PERF_SRCLINE_LINE.match(line)
    if match is None:
        return None
    return match.group("source"), int(match.group("line"))


def _inside(line: int, span: tuple[int, int]) -> bool:
    return span[0] <= line <= span[1]


def classify_sample(sample: dict[str, Any], source_map: dict[str, Any]) -> str:
    symbol = str(sample["symbol"])
    source_name = Path(str(sample["source"])).name
    line = int(sample["line"])
    lower_symbol = symbol.lower()

    if any(token in lower_symbol for token in FOREIGN_SYMBOL_TOKENS):
        return "foreign_symbol"
    if "campp_dot4_i16" in lower_symbol:
        return "mac_reduction"
    if any(
        name in lower_symbol
        for name in ("qconv_requantize", "nearbyint", "write_quantized")
    ):
        return "requant_write"
    if any(
        name in lower_symbol
        for name in (
            "campp_spatial_coordinates",
            "campp_tensor_view_byte_offset",
            "campp_read_qbyte",
        )
    ):
        return "address_load_control"
    if source_name == source_map["requant_source_name"]:
        return (
            "requant_write"
            if _inside(line, source_map["requant_span"])
            else "setup_other"
        )
    if source_name != source_map["source_name"]:
        return "unclassified"
    if _inside(line, source_map["dot_span"]):
        return "mac_reduction"
    if _inside(line, source_map["read_span"]) or _inside(
        line, source_map["spatial_span"]
    ):
        return "address_load_control"
    if _inside(line, source_map["requant_span"]):
        return "requant_write"
    if source_map["core_begin"] <= line < source_map["mac_loop_begin"]:
        return "address_load_control"
    if source_map["mac_loop_begin"] <= line <= source_map["core_end"]:
        return "mac_reduction"
    if source_map["requant_begin"] <= line <= source_map["requant_end"]:
        return "requant_write"
    if _inside(line, source_map["main_span"]):
        return "setup_other"
    return "unclassified"


def classify_perf_script(
    text: str, source_map: dict[str, Any]
) -> dict[str, Any]:
    periods = {category: 0 for category in CATEGORIES}
    sample_counts = {category: 0 for category in CATEGORIES}
    line_periods: dict[tuple[str, int, str, str], int] = {}
    samples, malformed_lines = COMMON.iter_perf_samples(text)
    for sample in samples:
        category = classify_sample(sample, source_map)
        period = int(sample["period"])
        periods[category] += period
        sample_counts[category] += 1
        key = (
            str(sample["source"]),
            int(sample["line"]),
            str(sample["symbol"]),
            category,
        )
        line_periods[key] = line_periods.get(key, 0) + period

    total_period = sum(periods.values())
    if not samples or total_period == 0:
        raise HotspotError("perf script contains no attributable cycle samples")
    address_period = periods["address_load_control"]
    mac_period = periods["mac_reduction"]
    core_period = address_period + mac_period
    top_lines = sorted(line_periods.items(), key=lambda item: item[1], reverse=True)
    return {
        "parsed_sample_count": len(samples),
        "malformed_line_count": malformed_lines,
        "total_sample_period": total_period,
        "category_sample_counts": sample_counts,
        "category_periods": periods,
        "category_share_pct": {
            category: value / total_period * 100.0
            for category, value in periods.items()
        },
        "address_vs_mac": {
            "classified_core_period": core_period,
            "address_pct": address_period / core_period * 100.0
            if core_period
            else None,
            "mac_pct": mac_period / core_period * 100.0 if core_period else None,
        },
        "top_lines": [
            {
                "source": Path(source).name,
                "line": line,
                "symbol": symbol,
                "category": category,
                "sample_period": period,
                "total_share_pct": period / total_period * 100.0,
            }
            for (source, line, symbol, category), period in top_lines[:20]
        ],
    }


def _pin_cpu_zero() -> None:
    os.sched_setaffinity(0, {0})


def _run(
    command: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        return COMMON.run_command(command, root=ROOT, check=check)
    except RuntimeError as exc:
        raise HotspotError(str(exc)) from exc


def _microbench_command(
    binary: Path,
    *,
    plan: Path,
    weights: Path,
    feature: Path,
    operator_id: int,
    warmup: int,
    repeat: int,
    qconv_candidate: str = "baseline",
    fused_qconv_candidate: str = "baseline",
) -> list[str]:
    command = DIAGNOSIS._command(
        binary,
        plan=plan,
        weights=weights,
        feature=feature,
        operator_id=operator_id,
        warmup=warmup,
        repeat=repeat,
    )
    if qconv_candidate != "baseline":
        command.extend(["--qconv-candidate", qconv_candidate])
    if fused_qconv_candidate != "baseline":
        command.extend(
            ["--fused-qconv-candidate", fused_qconv_candidate]
        )
    command.append("--perf-window")
    return command


def _run_one(
    *,
    perf: str,
    binary: Path,
    plan: Path,
    weights: Path,
    feature: Path,
    case: dict[str, Any],
    warmup: int,
    repeat: int,
    sample_period: int,
    output_dir: Path,
    source_map: dict[str, Any],
    qconv_candidate: str = "baseline",
    fused_qconv_candidate: str = "baseline",
    v2_source_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    perf_data = output_dir / "perf.data"
    payload_path = output_dir / "microbench.json"
    script_path = output_dir / "perf_script.txt"
    annotate_path = output_dir / "perf_annotate.txt"
    quantize_annotate_path = output_dir / "perf_annotate_quantize.txt"
    report_path = output_dir / "perf_report.txt"
    command_path = output_dir / "command.json"
    record_stderr_path = output_dir / "perf_record.stderr.txt"
    microbench = _microbench_command(
        binary,
        plan=plan,
        weights=weights,
        feature=feature,
        operator_id=int(case["operator_id"]),
        warmup=warmup,
        repeat=repeat,
        qconv_candidate=qconv_candidate,
        fused_qconv_candidate=fused_qconv_candidate,
    )
    # prctl cannot gate events owned by an external `perf record`, so the
    # prelude used to leak in. perf starts disabled (-D -1) and the microbench
    # turns sampling on through perf's own control FIFO instead.
    control_fifo = output_dir / "perf_ctl.fifo"
    ack_fifo = output_dir / "perf_ack.fifo"
    for fifo in (control_fifo, ack_fifo):
        if fifo.exists():
            fifo.unlink()
        os.mkfifo(fifo)
    microbench = [
        *microbench,
        "--perf-control",
        str(control_fifo),
        "--perf-ack",
        str(ack_fifo),
    ]
    record_command = [
        perf,
        "record",
        "--quiet",
        "--no-buildid",
        "--output",
        str(perf_data),
        "--event",
        "cycles:u",
        "--count",
        str(sample_period),
        "--delay",
        "-1",
        "--control",
        f"fifo:{control_fifo},{ack_fifo}",
        "--",
        *microbench,
    ]
    command_path.write_text(
        json.dumps(
            {
                "record": record_command,
                "cpu_affinity": [0],
                "measurement_scope": "target kernel only",
                "isolation": (
                    "perf control FIFO (-D -1) plus target symbol filter"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    try:
        completed = _run(record_command)
    finally:
        for fifo in (control_fifo, ack_fifo):
            if fifo.exists():
                fifo.unlink()
    record_stderr_path.write_text(
        completed.stderr, encoding="utf-8", newline="\n"
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HotspotError("hotspot binary did not return JSON") from exc
    operator = payload.get("operator", {})
    window = payload.get("perf_window", {})
    if (
        operator.get("operator_id") != case["operator_id"]
        or operator.get("kernel_name") != case["kernel_name"]
        or payload.get("output_hash_matches") is not True
        or payload.get("qconv_candidate") != qconv_candidate
        or payload.get("fused_qconv_candidate") != fused_qconv_candidate
        or window.get("requested") is not True
        or window.get("supported") is not True
        or window.get("enabled_at_exit") is not False
    ):
        raise HotspotError(f"invalid hotspot payload for {case['case_name']}")
    payload["input"] = {"path": _display_path(feature)}
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not perf_data.is_file() or perf_data.stat().st_size == 0:
        raise HotspotError("perf record did not create data")

    script_command = [
        perf,
        "script",
        "--input",
        str(perf_data),
        "--fields",
        "period,ip,sym,srcline",
    ]
    script = _run(script_command)
    script_path.write_text(script.stdout, encoding="utf-8", newline="\n")

    annotate_target = select_mac_annotate_target(
        script.stdout,
        qconv_candidate=qconv_candidate,
        fused_qconv_candidate=fused_qconv_candidate,
    )
    annotate_symbol = str(annotate_target["symbol"])
    annotate = _run(
        [
            perf,
            "annotate",
            "--stdio",
            "--input",
            str(perf_data),
            "--symbol",
            annotate_symbol,
        ],
        check=False,
    )
    annotate_path.write_text(
        annotate.stdout + annotate.stderr, encoding="utf-8", newline="\n"
    )
    quantize_spill = None
    if fused_qconv_candidate in (
        "quant_neon", "combined_fixed", "combined_v4", "combined_hybrid"
    ):
        quantize_annotate = _run(
            [
                perf,
                "annotate",
                "--stdio",
                "--input",
                str(perf_data),
                "--symbol",
                FUSED_QUANTIZE_SYMBOL,
            ],
            check=False,
        )
        quantize_annotate_path.write_text(
            quantize_annotate.stdout + quantize_annotate.stderr,
            encoding="utf-8",
            newline="\n",
        )
        quantize_spill = measure_spill_share(quantize_annotate.stdout)
    report = _run(
        [
            perf,
            "report",
            "--stdio",
            "--input",
            str(perf_data),
            "--sort",
            "symbol,srcline",
            "--percent-limit",
            "0",
        ],
        check=False,
    )
    report_path.write_text(
        report.stdout + report.stderr, encoding="utf-8", newline="\n"
    )
    candidate_active = (
        qconv_candidate != "baseline"
        or fused_qconv_candidate not in ("baseline", "quant_neon")
    )
    if candidate_active and v2_source_map is not None:
        classification = classify_v2_perf_script(script.stdout, v2_source_map)
    else:
        classification = classify_perf_script(script.stdout, source_map)
    classification["spill"] = measure_spill_share(annotate.stdout)
    classification.update(
        {
            "execution_path": annotate_target["execution_path"],
            "mac_annotate_symbol": annotate_symbol,
            "quantize_annotate_symbol": (
                FUSED_QUANTIZE_SYMBOL if quantize_spill is not None else None
            ),
            "quantize_spill": quantize_spill,
            "input": _display_path(feature),
            "output_hash": payload["output_hash"],
            "artifacts": {
                "perf_data": _display_path(perf_data),
                "perf_record_stderr": _display_path(record_stderr_path),
                "perf_script": _display_path(script_path),
                "perf_annotate": _display_path(annotate_path),
                "perf_annotate_quantize": (
                    _display_path(quantize_annotate_path)
                    if quantize_spill is not None else None
                ),
                "perf_report": _display_path(report_path),
                "microbench": _display_path(payload_path),
            },
        }
    )
    return classification


def _aggregate_v2_case(
    case: dict[str, Any], inputs: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    periods = {category: 0 for category in V2_CATEGORIES}
    detail_periods = {category: 0 for category in V2_DETAIL_CATEGORIES}
    area_periods = {area: 0 for area in V2_DETAIL_AREAS}
    for item in inputs:
        for category in V2_CATEGORIES:
            periods[category] += int(item["category_periods"][category])
        for category in V2_DETAIL_CATEGORIES:
            detail_periods[category] += int(
                item["detail_category_periods"][category]
            )
        for area in V2_DETAIL_AREAS:
            area_periods[area] += int(item["area_periods"][area])
    total = sum(periods.values())
    input_winners = [
        max(V2_CORE_CATEGORIES, key=lambda name: item["category_periods"][name])
        for item in inputs
    ]
    unclassified_shares = [
        float(item["category_share_pct"]["unclassified"]) for item in inputs
    ]
    foreign_shares = [
        float(item["category_share_pct"]["foreign_symbol"]) for item in inputs
    ]
    spill_shares = [
        float(item.get("spill", {}).get("stack_spill_share_of_annotated_pct", 0.0))
        for item in inputs
    ]
    fixed_sample_count = sum(
        int(item.get("fixed_microkernel_sample_count", 0)) for item in inputs
    )
    winner = input_winners[0] if len(set(input_winners)) == 1 else "mixed"
    area_input_winners = [
        max(
            ("address", "mac", "requant"),
            key=lambda name: item["area_periods"][name],
        )
        for item in inputs
    ]
    detail_core_categories = [
        name
        for name in V2_DETAIL_CATEGORIES
        if _detail_area(name) in ("address", "mac", "requant")
    ]
    detail_input_winners = [
        max(
            detail_core_categories,
            key=lambda name: item["detail_category_periods"][name],
        )
        for item in inputs
    ]
    area_winner = (
        area_input_winners[0]
        if len(set(area_input_winners)) == 1
        else "mixed"
    )
    detail_winner = (
        detail_input_winners[0]
        if len(set(detail_input_winners)) == 1
        else "mixed"
    )
    stable = (
        winner != "mixed"
        and max(unclassified_shares) <= 20.0
        and max(foreign_shares) <= 5.0
    )
    detail_stable = (
        area_winner != "mixed"
        and detail_winner != "mixed"
        and max(unclassified_shares) <= 20.0
        and max(foreign_shares) <= 5.0
    )
    ranked = sorted(
        V2_CORE_CATEGORIES, key=lambda name: periods[name], reverse=True
    )
    core = sum(periods[name] for name in V2_CORE_CATEGORIES)
    cumulative = 0
    top_set: list[str] = []
    for name in ranked:
        if periods[name] == 0:
            continue
        cumulative += periods[name]
        top_set.append(name)
        if core and cumulative / core >= 0.8:
            break
    detailed_core = sum(
        area_periods[name] for name in ("address", "mac", "requant")
    )
    detail_ranked = sorted(
        detail_core_categories,
        key=lambda name: detail_periods[name],
        reverse=True,
    )
    detail_cumulative = 0
    detail_top_set: list[str] = []
    for name in detail_ranked:
        if detail_periods[name] == 0:
            continue
        detail_cumulative += detail_periods[name]
        detail_top_set.append(name)
        if detailed_core and detail_cumulative / detailed_core >= 0.8:
            break
    detail_by_area = {}
    for area in ("address", "mac", "requant"):
        area_categories = [
            name for name in detail_core_categories if _detail_area(name) == area
        ]
        area_total = area_periods[area]
        detail_by_area[area] = [
            {
                "category": name,
                "sample_period": detail_periods[name],
                "area_share_pct": (
                    detail_periods[name] / area_total * 100.0
                    if area_total
                    else 0.0
                ),
                "total_share_pct": (
                    detail_periods[name] / total * 100.0 if total else 0.0
                ),
            }
            for name in sorted(
                area_categories,
                key=lambda category: detail_periods[category],
                reverse=True,
            )
            if detail_periods[name] > 0
        ]
    return {
        **case,
        "shape": case.get(
            "shape", case["case_name"].removeprefix("qconv_")
        ),
        "inputs": list(inputs),
        "aggregate": {
            "total_sample_period": total,
            "category_periods": periods,
            "category_share_pct": {
                name: value / total * 100.0 if total else 0.0
                for name, value in periods.items()
            },
            "classified_core_period": core,
            "classified_core_share_pct": core / total * 100.0 if total else 0.0,
            "top_bottleneck_set": top_set,
            "detail_category_periods": detail_periods,
            "detail_category_share_pct": {
                name: value / total * 100.0 if total else 0.0
                for name, value in detail_periods.items()
            },
            "area_periods": area_periods,
            "area_share_pct": {
                name: value / total * 100.0 if total else 0.0
                for name, value in area_periods.items()
            },
            "area_share_of_classified_pct": {
                name: area_periods[name] / detailed_core * 100.0
                if detailed_core
                else 0.0
                for name in ("address", "mac", "requant")
            },
            "detail_top_bottleneck_set": detail_top_set,
            "detail_by_area": detail_by_area,
            "stack_spill_share_of_annotated_pct": (
                sum(spill_shares) / len(spill_shares) if spill_shares else 0.0
            ),
            "fixed_microkernel_sample_count": fixed_sample_count,
        },
        "decision": {
            "winner": winner,
            "stable_across_inputs": stable,
            "input_winners": input_winners,
            "area_winner": area_winner,
            "area_input_winners": area_input_winners,
            "detail_winner": detail_winner,
            "detail_input_winners": detail_input_winners,
            "detail_stable_across_inputs": detail_stable,
            "maximum_unclassified_share_pct": max(unclassified_shares),
            "maximum_foreign_symbol_share_pct": max(foreign_shares),
            "rule": (
                "same top v2 category on 3 inputs, unclassified <=20%, "
                "foreign_symbol <=5%"
            ),
            "detail_rule": (
                "same address/MAC/requant area and same detailed winner on "
                "3 inputs, unclassified <=20%, foreign_symbol <=5%"
            ),
        },
    }


def _aggregate_case(
    case: dict[str, Any], inputs: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    periods = {category: 0 for category in CATEGORIES}
    for item in inputs:
        for category in CATEGORIES:
            periods[category] += int(item["category_periods"][category])
    total = sum(periods.values())
    address = periods["address_load_control"]
    mac = periods["mac_reduction"]
    core = address + mac
    input_winners = []
    input_winner_shares = []
    for item in inputs:
        pair = item["address_vs_mac"]
        address_pct = pair["address_pct"]
        mac_pct = pair["mac_pct"]
        if address_pct is None or mac_pct is None:
            input_winners.append("unknown")
            input_winner_shares.append(0.0)
        elif address_pct >= mac_pct:
            input_winners.append("address_load_control")
            input_winner_shares.append(float(address_pct))
        else:
            input_winners.append("mac_reduction")
            input_winner_shares.append(float(mac_pct))
    unclassified_shares = [
        float(item["category_share_pct"]["unclassified"]) for item in inputs
    ]
    # A non-trivial foreign share means the perf window failed to isolate the
    # target kernel, so the address/mac split cannot be trusted no matter how
    # stable the winner looks.
    foreign_shares = [
        float(item["category_share_pct"]["foreign_symbol"]) for item in inputs
    ]
    winner = input_winners[0] if len(set(input_winners)) == 1 else "mixed"
    stable = (
        winner not in ("unknown", "mixed")
        and min(input_winner_shares) >= 55.0
        and max(unclassified_shares) <= 20.0
        and max(foreign_shares) <= 5.0
    )
    return {
        **case,
        "inputs": list(inputs),
        "aggregate": {
            "total_sample_period": total,
            "category_periods": periods,
            "category_share_pct": {
                category: value / total * 100.0 if total else 0.0
                for category, value in periods.items()
            },
            "address_vs_mac": {
                "classified_core_period": core,
                "address_pct": address / core * 100.0 if core else None,
                "mac_pct": mac / core * 100.0 if core else None,
            },
        },
        "decision": {
            "winner": winner,
            "stable_across_inputs": stable,
            "input_winners": input_winners,
            "minimum_winner_share_pct": min(input_winner_shares),
            "maximum_unclassified_share_pct": max(unclassified_shares),
            "maximum_foreign_symbol_share_pct": max(foreign_shares),
            "rule": "same winner on 3 inputs, winner >=55%, unclassified <=20%, foreign_symbol <=5%",
        },
    }


def _build_v2_shape_breakdown(
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for case in cases:
        shape = str(case.get("shape", case["case_name"]))
        bucket = grouped.setdefault(
            shape,
            {
                "operator_ids": [],
                "profile_mean_ms_total": 0.0,
                "profile_share_pct_total": 0.0,
                "total_sample_period": 0,
                "area_periods": {area: 0 for area in V2_DETAIL_AREAS},
                "detail_category_periods": {
                    name: 0 for name in V2_DETAIL_CATEGORIES
                },
            },
        )
        bucket["operator_ids"].append(int(case["operator_id"]))
        bucket["profile_mean_ms_total"] += float(
            case.get("profile_mean_ms", 0.0)
        )
        bucket["profile_share_pct_total"] += float(
            case.get("profile_share_pct", 0.0)
        )
        aggregate = case["aggregate"]
        bucket["total_sample_period"] += int(aggregate["total_sample_period"])
        for area in V2_DETAIL_AREAS:
            bucket["area_periods"][area] += int(
                aggregate["area_periods"][area]
            )
        for name in V2_DETAIL_CATEGORIES:
            bucket["detail_category_periods"][name] += int(
                aggregate["detail_category_periods"][name]
            )

    breakdown = {}
    detail_core_categories = [
        name
        for name in V2_DETAIL_CATEGORIES
        if _detail_area(name) in ("address", "mac", "requant")
    ]
    for shape, bucket in grouped.items():
        total = int(bucket["total_sample_period"])
        area_periods = bucket["area_periods"]
        detail_periods = bucket["detail_category_periods"]
        detailed_core = sum(
            area_periods[name] for name in ("address", "mac", "requant")
        )
        detail_by_area = {}
        for area in ("address", "mac", "requant"):
            area_total = area_periods[area]
            names = [
                name for name in detail_core_categories if _detail_area(name) == area
            ]
            detail_by_area[area] = [
                {
                    "category": name,
                    "sample_period": detail_periods[name],
                    "area_share_pct": (
                        detail_periods[name] / area_total * 100.0
                        if area_total
                        else 0.0
                    ),
                    "total_share_pct": (
                        detail_periods[name] / total * 100.0 if total else 0.0
                    ),
                }
                for name in sorted(
                    names,
                    key=lambda category: detail_periods[category],
                    reverse=True,
                )
                if detail_periods[name] > 0
            ]
        breakdown[shape] = {
            "operator_count": len(bucket["operator_ids"]),
            "operator_ids": sorted(bucket["operator_ids"]),
            "profile_mean_ms_total": bucket["profile_mean_ms_total"],
            "profile_share_pct_total": bucket["profile_share_pct_total"],
            "total_sample_period": total,
            "area_periods": area_periods,
            "area_share_pct": {
                name: area_periods[name] / total * 100.0 if total else 0.0
                for name in V2_DETAIL_AREAS
            },
            "area_share_of_classified_pct": {
                name: area_periods[name] / detailed_core * 100.0
                if detailed_core
                else 0.0
                for name in ("address", "mac", "requant")
            },
            "detail_category_periods": detail_periods,
            "detail_category_share_pct": {
                name: detail_periods[name] / total * 100.0 if total else 0.0
                for name in V2_DETAIL_CATEGORIES
            },
            "detail_by_area": detail_by_area,
            "area_winner": max(
                ("address", "mac", "requant"),
                key=lambda name: area_periods[name],
            ),
            "detail_winner": max(
                detail_core_categories,
                key=lambda name: detail_periods[name],
            ),
        }
    return breakdown


def build_summary(
    cases: Sequence[dict[str, Any]],
    *,
    warmup: int,
    repeat: int,
    sample_period: int,
    elapsed_seconds: float,
    source_map: dict[str, Any],
    qconv_candidate: str = "baseline",
    v2_source_map: dict[str, Any] | None = None,
    case_scope: str = "representative",
) -> dict[str, Any]:
    winners = [case["decision"]["winner"] for case in cases]
    stable = all(case["decision"]["stable_across_inputs"] for case in cases)
    if stable and len(set(winners)) == 1:
        overall = winners[0]
    elif stable:
        overall = "shape_specific"
    else:
        overall = "inconclusive"
    shape_breakdown = (
        _build_v2_shape_breakdown(cases)
        if qconv_candidate != "baseline"
        else None
    )
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "separate address, MAC and requant sub-stages for 3x3/1x1 QConv"
            if qconv_candidate != "baseline"
            else "separate address/load/control cycles from MAC cycles in E7 QConv"
        ),
        "configuration": {
            "bucket_frames": 98,
            "threads": 1,
            "cpu_affinity": [0],
            "warmup": warmup,
            "repeat_per_input": repeat,
            "input_count": 3,
            "perf_event": "cycles:u",
            "sample_period": sample_period,
            "stage_probe_compiled": False,
            "qconv_candidate": qconv_candidate,
            "case_scope": case_scope,
            "elapsed_seconds": elapsed_seconds,
        },
        "v2_source_map": v2_source_map,
        "detail_classification": (
            {
                "categories": list(V2_DETAIL_CATEGORIES),
                "areas": list(V2_DETAIL_AREAS),
                "parent_categories": V2_DETAIL_PARENT,
                "measurement": "perf cycles:u source-line samples",
                "limitation": (
                    "macro-expanded instructions can share one source line; "
                    "inspect perf_annotate for the winning detail category"
                ),
            }
            if qconv_candidate != "baseline"
            else None
        ),
        "source_map": {
            key: value
            for key, value in source_map.items()
            if key not in ("source", "source_name")
        }
        | {"source": _display_path(source_map["source"])},
        "cases": list(cases),
        "shape_breakdown": shape_breakdown,
        "decision": {
            "ready": stable,
            "detail_ready": (
                all(
                    case["decision"].get("detail_stable_across_inputs", False)
                    for case in cases
                )
                if qconv_candidate != "baseline"
                else None
            ),
            "overall": overall,
            "next_candidate": {
                "address_load_control": "QConv address fast path",
                "mac_reduction": "common O4I4 NEON microkernel",
                "shape_specific": "separate 3x3 and 1x1 candidates",
                "inconclusive": "inspect perf_annotate and reduce unclassified samples",
                # MAC v2 categories
                "v2_mac_smlal": "widen the SMLAL body: more accumulators in flight",
                "v2_input_address": "hoist input point construction out of the tile loop",
                "v2_weight_transform": "hoist or cache the weight column transform",
                "v2_requant": "vectorise the requantize path",
                "v2_output_store": "widen the output store",
                "mixed": "shape-specific v2 follow-up",
            }[overall],
        },
    }


def _write_v2_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "case_name",
        "shape",
        "operator_id",
        "input",
        "execution_path",
        "fixed_microkernel_sample_count",
        *[f"{name}_pct" for name in V2_CATEGORIES],
        "classified_core_share_pct",
        *[f"area_{name}_pct" for name in V2_DETAIL_AREAS],
        *[
            f"area_{name}_pct_of_classified"
            for name in ("address", "mac", "requant")
        ],
        *[f"detail_{name}_pct" for name in V2_DETAIL_CATEGORIES],
        "stack_spill_share_of_annotated_pct",
        "winner",
        "area_winner",
        "detail_winner",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        for case in summary["cases"]:
            for item in case["inputs"]:
                shares = item["category_share_pct"]
                area_shares = item["area_share_pct"]
                area_classified = item["area_share_of_classified_pct"]
                detail_shares = item["detail_category_share_pct"]
                detail_core_categories = [
                    name
                    for name in V2_DETAIL_CATEGORIES
                    if _detail_area(name) in ("address", "mac", "requant")
                ]
                writer.writerow(
                    {
                        "case_name": case["case_name"],
                        "shape": case.get("shape", case["case_name"]),
                        "operator_id": case["operator_id"],
                        "input": item["input"],
                        "execution_path": item.get("execution_path"),
                        "fixed_microkernel_sample_count": item.get(
                            "fixed_microkernel_sample_count", 0
                        ),
                        **{
                            f"{name}_pct": f"{shares[name]:.6f}"
                            for name in V2_CATEGORIES
                        },
                        "classified_core_share_pct": (
                            f"{item['classified_core_share_pct']:.6f}"
                        ),
                        **{
                            f"area_{name}_pct": f"{area_shares[name]:.6f}"
                            for name in V2_DETAIL_AREAS
                        },
                        **{
                            f"area_{name}_pct_of_classified": (
                                f"{area_classified[name]:.6f}"
                            )
                            for name in ("address", "mac", "requant")
                        },
                        **{
                            f"detail_{name}_pct": f"{detail_shares[name]:.6f}"
                            for name in V2_DETAIL_CATEGORIES
                        },
                        "stack_spill_share_of_annotated_pct": (
                            f"{item.get('spill', {}).get('stack_spill_share_of_annotated_pct', 0.0):.6f}"
                        ),
                        "winner": max(
                            V2_CORE_CATEGORIES,
                            key=lambda name: item["category_periods"][name],
                        ),
                        "area_winner": max(
                            ("address", "mac", "requant"),
                            key=lambda name: item["area_periods"][name],
                        ),
                        "detail_winner": max(
                            detail_core_categories,
                            key=lambda name: item["detail_category_periods"][name],
                        ),
                    }
                )


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "case_name",
        "operator_id",
        "input",
        "address_load_control_pct",
        "mac_reduction_pct",
        "requant_write_pct",
        "setup_other_pct",
        "unclassified_pct",
        "address_pct_of_classified_core",
        "mac_pct_of_classified_core",
        "winner",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        for case in summary["cases"]:
            for item in case["inputs"]:
                shares = item["category_share_pct"]
                pair = item["address_vs_mac"]
                winner = (
                    "address_load_control"
                    if (pair["address_pct"] or 0) >= (pair["mac_pct"] or 0)
                    else "mac_reduction"
                )
                writer.writerow(
                    {
                        "case_name": case["case_name"],
                        "operator_id": case["operator_id"],
                        "input": item["input"],
                        "address_load_control_pct": shares["address_load_control"],
                        "mac_reduction_pct": shares["mac_reduction"],
                        "requant_write_pct": shares["requant_write"],
                        "setup_other_pct": shares["setup_other"],
                        "unclassified_pct": shares["unclassified"],
                        "address_pct_of_classified_core": pair["address_pct"],
                        "mac_pct_of_classified_core": pair["mac_pct"],
                        "winner": winner,
                    }
                )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=_path,
        default=ROOT / "results/profiling/e7_98/operator_profile.json",
    )
    parser.add_argument(
        "--binary",
        type=_path,
        default=ROOT / "build/profill/optimization/campp_operator_hotspot",
    )
    parser.add_argument("--plan", type=_path)
    parser.add_argument("--weights", type=_path)
    parser.add_argument(
        "--features", type=_path, nargs="+", default=list(DIAGNOSIS.EXPECTED_INPUTS)
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--sample-period", type=int, default=100_000)
    parser.add_argument(
        "--runs-dir",
        type=_path,
        default=ROOT / "runs/profiling/e7_98/optimization/qconv_hotspot",
    )
    parser.add_argument(
        "--results-dir",
        type=_path,
        default=ROOT / "results/profiling/e7_98/optimization/qconv_hotspot",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--case-scope",
        choices=("representative", "all"),
        default="representative",
        help="profile the hottest 3x3/1x1 pair or every ordinary QConv layer",
    )
    parser.add_argument(
        "--qconv-candidate",
        choices=(
            "baseline", "address", "mac", "combined", "mac_fixed", "mac_asm",
            "v4", "hybrid",
        ),
        default="baseline",
        help="profile a candidate instead of the production kernel; "
        "non-baseline modes select the candidate classifier",
    )
    args = parser.parse_args(argv)

    try:
        if args.warmup < 0 or args.repeat <= 0 or args.sample_period <= 0:
            raise HotspotError("warmup, repeat or sample period is invalid")
        if os.name != "posix" and not args.preflight_only:
            raise HotspotError("actual perf sampling requires Linux/QRB2210")
        perf = shutil.which("perf")
        if perf is None:
            raise HotspotError("Linux perf executable not found")
        if not args.profile.is_file():
            raise HotspotError(f"profile not found: {args.profile}")
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        cases = select_qconv_cases(
            profile.get("operators", []), case_scope=args.case_scope
        )
        if args.case_scope == "representative" and [
            case["case_name"] for case in cases
        ] != list(TARGET_CASE_NAMES):
            raise HotspotError("QConv 3x3/1x1 cases were not selected")
        plan = args.plan or DIAGNOSIS._resolve_profile_artifact(profile, "plan")
        weights = args.weights or DIAGNOSIS._resolve_profile_artifact(
            profile, "weights"
        )
        required = [
            args.binary,
            plan,
            weights,
            QCONV_SOURCE,
            REQUANT_SOURCE,
            *args.features,
        ]
        if args.qconv_candidate != "baseline":
            required.extend(
                [
                    V2_MAC_SOURCE,
                    V2_CANDIDATE_SOURCE,
                    V2_ADDRESS_SOURCE,
                    FIXED_MAC_SOURCE,
                    FIXED_DISPATCH_SOURCE,
                    FIXED_ASSEMBLY_SOURCE,
                ]
            )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise HotspotError(
                "required files are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
            )
        if len(args.features) != 3 or any(
            path.stat().st_size != 98 * 80 * 4 for path in args.features
        ):
            raise HotspotError("exactly three float32 [1,98,80] inputs are required")
        summary_path = args.results_dir / "qconv_hotspot.json"
        csv_path = args.results_dir / "qconv_hotspot.csv"
        if summary_path.exists() and not args.force and not args.preflight_only:
            raise HotspotError(f"output exists: {summary_path} (use --force)")
        capabilities = DIAGNOSIS._run_payload([str(args.binary), "--capabilities"])
        if capabilities.get("perf_sample_window") is not True:
            raise HotspotError("binary does not support target-only perf windows")
        if capabilities.get("stage_probe") is not False:
            raise HotspotError("hotspot binary must be built without stage probes")
        source_map = build_source_map()
        v2_source_map = (
            build_v2_source_map() if args.qconv_candidate != "baseline" else None
        )
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "ready": True,
                        "perf": perf,
                        "binary": _display_path(args.binary),
                        "plan": _display_path(plan),
                        "weights": _display_path(weights),
                        "features": [_display_path(path) for path in args.features],
                        "cases": cases,
                        "case_scope": args.case_scope,
                        "qconv_candidate": args.qconv_candidate,
                        "source_map": {
                            key: value
                            for key, value in source_map.items()
                            if key not in ("source", "source_name")
                        },
                        "v2_source_map": v2_source_map,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        estimated_seconds = 25.0 + sum(
            float(case["profile_mean_ms"]) / 1000.0
            * (args.warmup + args.repeat + 1)
            * len(args.features)
            for case in cases
        )
        print(f"예상 시간: 약 {max(1, math.ceil(estimated_seconds / 60.0))}분")
        started = time.monotonic()
        results = []
        for case in cases:
            inputs = []
            for feature in args.features:
                print("진행 중", flush=True)
                output_dir = (
                    args.runs_dir / case["case_name"] / feature.stem
                )
                if output_dir.exists() and args.force:
                    for generated in output_dir.iterdir():
                        if generated.is_file():
                            generated.unlink()
                inputs.append(
                    _run_one(
                        perf=perf,
                        binary=args.binary,
                        plan=plan,
                        weights=weights,
                        feature=feature,
                        case=case,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        sample_period=args.sample_period,
                        output_dir=output_dir,
                        source_map=source_map,
                        qconv_candidate=args.qconv_candidate,
                        v2_source_map=v2_source_map,
                    )
                )
            results.append(
                _aggregate_v2_case(case, inputs)
                if args.qconv_candidate != "baseline"
                else _aggregate_case(case, inputs)
            )
        summary = build_summary(
            results,
            warmup=args.warmup,
            repeat=args.repeat,
            sample_period=args.sample_period,
            elapsed_seconds=time.monotonic() - started,
            source_map=source_map,
            qconv_candidate=args.qconv_candidate,
            v2_source_map=v2_source_map,
            case_scope=args.case_scope,
        )
        summary["artifacts"] = {
            "profile": _display_path(args.profile),
            "plan": _display_path(plan),
            "weights": _display_path(weights),
            "raw_dir": _display_path(args.runs_dir),
        }
        args.results_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if args.qconv_candidate != "baseline":
            _write_v2_csv(csv_path, summary)
        else:
            _write_csv(csv_path, summary)
        print(f"QConv hotspot 완료: {_display_path(summary_path)}")
        for case in summary["cases"]:
            if args.qconv_candidate != "baseline":
                aggregate = case["aggregate"]
                print(f"  {case['case_name']}:")
                for name in V2_CATEGORIES:
                    print(
                        f"    {name}: "
                        f"{aggregate['category_share_pct'][name]:.2f}%"
                    )
                print(
                    "    stack_spill(of annotated): "
                    f"{aggregate['stack_spill_share_of_annotated_pct']:.2f}%"
                )
                area = aggregate["area_share_of_classified_pct"]
                print(
                    "    detailed areas(classified): "
                    f"address={area['address']:.2f}% "
                    f"mac={area['mac']:.2f}% "
                    f"requant={area['requant']:.2f}%"
                )
                print(
                    f"    top={aggregate['top_bottleneck_set']} "
                    f"winner={case['decision']['winner']} "
                    f"detail={case['decision']['detail_winner']}"
                )
                print(
                    "    detail_top="
                    f"{aggregate['detail_top_bottleneck_set']}"
                )
                continue
            pair = case["aggregate"]["address_vs_mac"]
            print(
                f"  {case['case_name']}: address={pair['address_pct']:.2f}% "
                f"mac={pair['mac_pct']:.2f}% "
                f"winner={case['decision']['winner']}"
            )
        print(f"  next: {summary['decision']['next_candidate']}")
        return 0 if summary["decision"]["ready"] else 3
    except (HotspotError, OSError, ValueError, KeyError) as exc:
        print(f"QConv hotspot failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
