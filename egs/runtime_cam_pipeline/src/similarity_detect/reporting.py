"""Stable terminal report required by the microphone demo."""

from __future__ import annotations

from voice_embedding.runtime import RuntimeMetrics


def _mib(value: int) -> float:
    return value / (1024.0 * 1024.0)


def _optional_mib(value: int | None) -> str:
    return "unavailable" if value is None else f"{_mib(value):.2f} MiB"


def format_terminal_report(score: float, metrics: RuntimeMetrics) -> str:
    return "\n".join([
        "=================================",
        "=================================",
        "=====                         ======",
        f"=====      final score: {score:.6f}      ======",
        "=====                         ======",
        "=================================",
        "=================================",
        "[report]",
        f"Backend: {metrics.backend_name}",
        "Peak RAM",
        (
            "- pipeline total peak: "
            f"{_optional_mib(metrics.pipeline_total_peak_rss_bytes)}"
        ),
        (
            "- Python host peak: "
            f"{_optional_mib(metrics.python_host_peak_rss_bytes)}"
        ),
        (
            f"- {metrics.runtime_peak_label} peak: "
            f"{_optional_mib(metrics.peak_rss_bytes)}"
        ),
        f"- {metrics.weight_label}: {_optional_mib(metrics.weight_bytes)}",
        (
            f"- {metrics.activation_label}: "
            f"{_optional_mib(metrics.activation_bytes)}"
        ),
        f"RTF: {metrics.rtf:.6f}",
    ])
