"""
Ground-truth check for the statically resolved shapes.

The IR's shapes come from ONNX + ORT symbolic inference, which is inference --
not measurement.  Every downstream decision (arena offsets, packed weight
layouts, tile sizes) is built on those numbers, so they get verified once
against an actual ORT run with all intermediates promoted to graph outputs.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import onnx
from onnx import helper

from graph_ir import GraphIR, freeze_input_shape


def _expose_all_intermediates(model: onnx.ModelProto) -> onnx.ModelProto:
    """Add every internal tensor to graph.output so ORT hands back its value."""
    exposed = onnx.ModelProto()
    exposed.CopyFrom(model)
    graph = exposed.graph

    existing = {o.name for o in graph.output}
    initializers = {i.name for i in graph.initializer}
    for node in graph.node:
        for out in node.output:
            if not out or out in existing or out in initializers:
                continue
            graph.output.append(
                helper.make_empty_tensor_value_info(out)
            )
            existing.add(out)
    return exposed


def verify_shapes(
    ir: GraphIR,
    max_report: int = 20,
    sample: Optional[np.ndarray] = None,
) -> Tuple[int, int, List[str]]:
    """Run the frozen graph once and compare measured vs inferred shapes.

    Returns (checked, mismatched, list of human-readable mismatch lines).
    """
    import onnxruntime as ort

    raw = onnx.load(ir.model_path)
    frozen = freeze_input_shape(raw, batch=ir.batch, frames=ir.frames)
    exposed = _expose_all_intermediates(frozen)

    opts = ort.SessionOptions()
    # No fusion: fused kernels drop the intermediate tensors we want to see.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        exposed.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )

    inp = sess.get_inputs()[0]
    if sample is None:
        rng = np.random.default_rng(0)
        sample = rng.standard_normal((ir.batch, ir.frames, 80)).astype(np.float32)
    feeds = {inp.name: sample}

    names = [o.name for o in sess.get_outputs()]
    values = sess.run(names, feeds)

    checked = 0
    mismatched = 0
    bad: List[str] = []
    for name, val in zip(names, values):
        t = ir.tensors.get(name)
        if t is None or t.shape is None:
            continue
        measured = list(np.shape(val))
        checked += 1
        if measured != list(t.shape):
            mismatched += 1
            if len(bad) < max_report:
                bad.append(f"{name}: inferred={t.shape} measured={measured}")

    return checked, mismatched, bad
