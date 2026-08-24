#!/usr/bin/env python3
"""Compatibility entry point for the generic weight-streaming builder."""

from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).with_name("08_build_weight_streaming.py")),
    run_name="__main__",
)
