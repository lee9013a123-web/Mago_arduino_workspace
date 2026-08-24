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
        "Peak RAM",
        (
            "- pipeline total peak: "
            f"{_optional_mib(metrics.pipeline_total_peak_rss_bytes)}"
        ),
        (
            "- Python/Torch peak: "
            f"{_optional_mib(metrics.python_torch_peak_rss_bytes)}"
        ),
        f"- C runtime peak: {_mib(metrics.peak_rss_bytes):.2f} MiB",
        f"- logical weight: {_mib(metrics.weight_bytes):.2f} MiB",
        f"- logical activation: {_mib(metrics.activation_bytes):.2f} MiB",
        f"RTF: {metrics.rtf:.6f}",
    ])
