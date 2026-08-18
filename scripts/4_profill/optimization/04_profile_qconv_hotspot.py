#!/usr/bin/env python3
"""QConv 3x3/1x1의 target-only cycle sample을 MAC과 address로 분리한다."""

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


def build_v2_source_map(
    mac_source: Path = V2_MAC_SOURCE,
    candidate_source: Path = V2_CANDIDATE_SOURCE,
) -> dict[str, Any]:
    """Function spans for the MAC v2 candidate.

    The v2 helpers are `static` and get inlined, so perf reports the enclosing
    symbol. Source-line spans are therefore what separates weight transform,
    SMLAL body, requant and store. Register spill cannot be split this way at
    all -- it is instruction level, so it is measured from `perf annotate`.
    """
    mac_lines = mac_source.read_text(encoding="utf-8").splitlines()
    cand_lines = candidate_source.read_text(encoding="utf-8").splitlines()
    return {
        "mac_source_name": mac_source.name,
        "candidate_source_name": candidate_source.name,
        "mac_line_count": len(mac_lines),
        "candidate_line_count": len(cand_lines),
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
        "half_tile_v2_span": _function_span(
            mac_lines, "static int campp_qconv_mac_neon_half_tile_v2("
        ),
        "scalar_tile_v2_span": _function_span(
            mac_lines, "static int campp_qconv_mac_scalar_tile_v2("
        ),
        "tile_v2_span": _function_span(
            mac_lines, "int campp_qconv_mac_neon_tile_v2("
        ),
        "requantize_neon4_span": _function_span(
            mac_lines, "static uint32_t campp_qconv_requantize_neon4("
        ),
        "requantize_scalar_span": _function_span(
            mac_lines, "static uint8_t campp_qconv_requantize_scalar("
        ),
        "requantize_store_span": _function_span(
            mac_lines, "int campp_qconv_requantize_store_neon_tile("
        ),
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
        "cand_run_v2_span": _function_span(
            cand_lines, "static CamppStatus campp_qconv_candidate_run_v2("
        ),
    }


def classify_v2_sample(
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

    if source_name == source_map["mac_source_name"]:
        for key, category in (
            ("weight_columns_span", "v2_weight_transform"),
            ("dot_weight_span", "v2_weight_transform"),
            ("dot_weight_zero_span", "v2_weight_transform"),
            ("neon_input_span", "v2_input_address"),
            ("dot_input_span", "v2_input_address"),
            ("candidate_read_span", "v2_input_address"),
            ("smlal_accumulate_span", "v2_mac_smlal"),
            ("dot_accumulate_span", "v2_mac_smlal"),
            ("half_tile_v2_span", "v2_mac_smlal"),
            ("scalar_tile_v2_span", "v2_mac_smlal"),
            ("tile_v2_span", "v2_mac_smlal"),
            ("requantize_neon4_span", "v2_requant"),
            ("requantize_scalar_span", "v2_requant"),
            ("requantize_store_span", "v2_output_store"),
        ):
            if _inside(line, source_map[key]):
                return category
        return "setup_other"

    if source_name == source_map["candidate_source_name"]:
        if _inside(line, source_map["cand_spatial_span"]):
            return "v2_input_address"
        if _inside(line, source_map["cand_write_span"]):
            return "v2_output_store"
        if _inside(line, source_map["cand_run_v2_span"]):
            return "v2_input_address"
        if _inside(line, source_map["cand_prepare_span"]) or _inside(
            line, source_map["cand_load_parameters_span"]
        ):
            return "setup_other"
        return "setup_other"

    # tensor_view.h / arm_neon.h inline into the candidate; attribute by intent.
    if source_name == "tensor_view.h":
        return "v2_input_address"
    if source_name == "arm_neon.h":
        return "v2_mac_smlal"
    return "unclassified"


def classify_v2_perf_script(
    text: str, source_map: dict[str, Any]
) -> dict[str, Any]:
    periods = {category: 0 for category in V2_CATEGORIES}
    counts = {category: 0 for category in V2_CATEGORIES}
    line_periods: dict[tuple[str, int, str, str], int] = {}
    samples, malformed = COMMON.iter_perf_samples(text)
    for sample in samples:
        category = classify_v2_sample(sample, source_map)
        period = int(sample["period"])
        periods[category] += period
        counts[category] += 1
        key = (
            Path(str(sample["source"])).name,
            int(sample["line"]),
            str(sample["symbol"]),
            category,
        )
        line_periods[key] = line_periods.get(key, 0) + period
    total = sum(periods.values())
    if total == 0:
        raise HotspotError("perf script contains no attributable cycle samples")
    core = sum(periods[name] for name in V2_CORE_CATEGORIES)
    top = sorted(line_periods.items(), key=lambda item: item[1], reverse=True)
    return {
        "parsed_sample_count": sum(counts.values()),
        "malformed_line_count": malformed,
        "total_sample_period": total,
        "category_sample_counts": counts,
        "category_periods": periods,
        "category_share_pct": {
            name: value / total * 100.0 for name, value in periods.items()
        },
        "classified_core_period": core,
        "classified_core_share_pct": core / total * 100.0 if total else 0.0,
        "top_lines": [
            {
                "source": source,
                "line": line,
                "symbol": symbol,
                "category": category,
                "sample_period": period,
                "total_share_pct": period / total * 100.0,
            }
            for (source, line, symbol, category), period in top[:20]
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


def build_source_map(source: Path = QCONV_SOURCE) -> dict[str, Any]:
    lines = source.read_text(encoding="utf-8").splitlines()
    dot_span = _function_span(lines, "static int32_t campp_dot4_i16(")
    read_span = _function_span(lines, "static int32_t campp_read_qbyte(")
    spatial_span = _function_span(lines, "static void campp_spatial_coordinates(")
    requant_span = _function_span(lines, "static CamppStatus campp_qconv_write(")
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
        for name in ("campp_qconv_write", "nearbyint", "write_quantized")
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
    v2_source_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    perf_data = output_dir / "perf.data"
    payload_path = output_dir / "microbench.json"
    script_path = output_dir / "perf_script.txt"
    annotate_path = output_dir / "perf_annotate.txt"
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

    annotate_symbol = (
        "campp_qconv_mac_neon_tile_v2"
        if qconv_candidate != "baseline"
        else "campp_aarch64_qlinear_conv_o4i4"
    )
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
    if qconv_candidate != "baseline" and v2_source_map is not None:
        classification = classify_v2_perf_script(script.stdout, v2_source_map)
        classification["spill"] = measure_spill_share(annotate.stdout)
    else:
        classification = classify_perf_script(script.stdout, source_map)
    classification.update(
        {
            "input": _display_path(feature),
            "output_hash": payload["output_hash"],
            "artifacts": {
                "perf_data": _display_path(perf_data),
                "perf_record_stderr": _display_path(record_stderr_path),
                "perf_script": _display_path(script_path),
                "perf_annotate": _display_path(annotate_path),
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
    for item in inputs:
        for category in V2_CATEGORIES:
            periods[category] += int(item["category_periods"][category])
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
    winner = input_winners[0] if len(set(input_winners)) == 1 else "mixed"
    stable = (
        winner != "mixed"
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
    return {
        **case,
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
            "stack_spill_share_of_annotated_pct": (
                sum(spill_shares) / len(spill_shares) if spill_shares else 0.0
            ),
        },
        "decision": {
            "winner": winner,
            "stable_across_inputs": stable,
            "input_winners": input_winners,
            "maximum_unclassified_share_pct": max(unclassified_shares),
            "maximum_foreign_symbol_share_pct": max(foreign_shares),
            "rule": (
                "same top v2 category on 3 inputs, unclassified <=20%, "
                "foreign_symbol <=5%"
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
) -> dict[str, Any]:
    winners = [case["decision"]["winner"] for case in cases]
    stable = all(case["decision"]["stable_across_inputs"] for case in cases)
    if stable and len(set(winners)) == 1:
        overall = winners[0]
    elif stable:
        overall = "shape_specific"
    else:
        overall = "inconclusive"
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "locate the internal bottleneck of the MAC v2 candidate"
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
            "elapsed_seconds": elapsed_seconds,
        },
        "v2_source_map": v2_source_map,
        "source_map": {
            key: value
            for key, value in source_map.items()
            if key not in ("source", "source_name")
        }
        | {"source": _display_path(source_map["source"])},
        "cases": list(cases),
        "decision": {
            "ready": stable,
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
        "operator_id",
        "input",
        *[f"{name}_pct" for name in V2_CATEGORIES],
        "classified_core_share_pct",
        "stack_spill_share_of_annotated_pct",
        "winner",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        for case in summary["cases"]:
            for item in case["inputs"]:
                shares = item["category_share_pct"]
                writer.writerow(
                    {
                        "case_name": case["case_name"],
                        "operator_id": case["operator_id"],
                        "input": item["input"],
                        **{
                            f"{name}_pct": f"{shares[name]:.6f}"
                            for name in V2_CATEGORIES
                        },
                        "classified_core_share_pct": (
                            f"{item['classified_core_share_pct']:.6f}"
                        ),
                        "stack_spill_share_of_annotated_pct": (
                            f"{item.get('spill', {}).get('stack_spill_share_of_annotated_pct', 0.0):.6f}"
                        ),
                        "winner": max(
                            V2_CORE_CATEGORIES,
                            key=lambda name: item["category_periods"][name],
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
        "--qconv-candidate",
        choices=("baseline", "address", "mac", "combined"),
        default="baseline",
        help="profile a candidate instead of the production kernel; "
        "'mac'/'combined' select the MAC v2 classifier",
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
        selected = DIAGNOSIS.select_representative_cases(profile.get("operators", []))
        cases = [case for case in selected if case["case_name"] in TARGET_CASE_NAMES]
        if [case["case_name"] for case in cases] != list(TARGET_CASE_NAMES):
            raise HotspotError("QConv 3x3/1x1 cases were not selected")
        plan = args.plan or DIAGNOSIS._resolve_profile_artifact(profile, "plan")
        weights = args.weights or DIAGNOSIS._resolve_profile_artifact(
            profile, "weights"
        )
        required = [args.binary, plan, weights, QCONV_SOURCE, *args.features]
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
                print(
                    f"    top={aggregate['top_bottleneck_set']} "
                    f"winner={case['decision']['winner']}"
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
