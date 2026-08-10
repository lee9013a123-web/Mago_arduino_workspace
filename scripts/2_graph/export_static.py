#!/usr/bin/env python3
"""
Emit a runnable, fully static ONNX per length bucket.

``ir_*.json`` is an analysis artifact -- it describes the static graph but
cannot be executed.  This produces the thing ORT can actually load, so the
"55% of nodes are shape plumbing" claim can be measured instead of asserted:

    campp_static_98.onnx      1 s
    campp_static_298.onnx     3 s
    campp_static_498.onnx     5 s
    campp_static_998.onnx    10 s

Each one has

  * input shape fully pinned to (1, frames, 80)
  * every shape-domain node evaluated once and replaced by an initializer
  * dead nodes and orphaned initializers dropped
  * all weights and quantization parameters (scale / zero_point) untouched
  * output still ``embedding[1, 192]``

These are intermediate artifacts for validating the static-shape effect inside
ORT, not the deployment format.  The deployment format is the packed bucket
plan built from the IR.

Usage
-----
    venv/bin/python campp_acceleration/graph/export_static.py \
        --model models/campplus_int8_static_qop.onnx \
        --seconds 1 3 5 10 \
        --out-dir campp_acceleration/out/static
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from graph_ir import _mark_static, freeze_input_shape, frames_for_seconds  # noqa: E402

DEFAULT_MODEL = "models/campplus_int8_static_qop.onnx"
DEFAULT_BUCKETS = (1.0, 3.0, 5.0, 10.0)


@dataclass
class ExportStats:
    frames: int
    seconds: float
    path: Path
    nodes_before: int
    nodes_after: int
    folded: int
    dead: int
    initializers_before: int
    initializers_after: int
    bytes_before: int
    bytes_after: int
    max_abs_diff: Optional[float] = None
    cosine: Optional[float] = None


# ---------------------------------------------------------------------------
# constant folding of the shape domain
# ---------------------------------------------------------------------------


def _boundary_tensors(
    graph: onnx.GraphProto,
    node_static: List[bool],
    initializers: Set[str],
) -> List[str]:
    """Static tensors that a surviving node (or the graph output) still reads.

    Everything else in the shape domain is pure scaffolding and disappears with
    its producer.
    """
    produced_static: Dict[str, int] = {}
    for idx, node in enumerate(graph.node):
        if node_static[idx]:
            for out in node.output:
                if out:
                    produced_static[out] = idx

    needed: List[str] = []
    seen: Set[str] = set()
    for idx, node in enumerate(graph.node):
        if node_static[idx]:
            continue
        for inp in node.input:
            if inp and inp in produced_static and inp not in initializers:
                if inp not in seen:
                    seen.add(inp)
                    needed.append(inp)
    for out in graph.output:
        if out.name in produced_static and out.name not in seen:
            seen.add(out.name)
            needed.append(out.name)
    return needed


def _evaluate(
    model: onnx.ModelProto,
    tensors: List[str],
    batch: int,
    frames: int,
) -> Dict[str, np.ndarray]:
    """Run the frozen graph once to read out the shape-domain values.

    Safe to feed zeros: every tensor in this set derives from initializers and
    ``Shape`` outputs, so its value depends on the frozen shape, never on the
    audio.
    """
    if not tensors:
        return {}

    import onnxruntime as ort

    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    existing = {o.name for o in probe.graph.output}
    for name in tensors:
        if name not in existing:
            probe.graph.output.append(helper.make_empty_tensor_value_info(name))

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        probe.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )
    inp = sess.get_inputs()[0]
    feed = {inp.name: np.zeros((batch, frames, 80), dtype=np.float32)}
    values = sess.run(tensors, feed)
    return dict(zip(tensors, values))


# ---------------------------------------------------------------------------
# dead node elimination
# ---------------------------------------------------------------------------


def _prune(graph: onnx.GraphProto) -> int:
    """Drop nodes and initializers that no graph output depends on.

    Returns the number of nodes removed.
    """
    producer: Dict[str, int] = {}
    for idx, node in enumerate(graph.node):
        for out in node.output:
            if out:
                producer[out] = idx

    live_nodes: Set[int] = set()
    live_tensors: Set[str] = set()
    stack = [o.name for o in graph.output]
    while stack:
        name = stack.pop()
        if name in live_tensors:
            continue
        live_tensors.add(name)
        idx = producer.get(name)
        if idx is None or idx in live_nodes:
            continue
        live_nodes.add(idx)
        for inp in graph.node[idx].input:
            if inp:
                stack.append(inp)

    removed = len(graph.node) - len(live_nodes)
    kept = [n for i, n in enumerate(graph.node) if i in live_nodes]
    del graph.node[:]
    graph.node.extend(kept)

    kept_init = [i for i in graph.initializer if i.name in live_tensors]
    del graph.initializer[:]
    graph.initializer.extend(kept_init)

    kept_vi = [
        vi for vi in graph.value_info
        if vi.name in live_tensors and vi.name not in {i.name for i in kept_init}
    ]
    del graph.value_info[:]
    graph.value_info.extend(kept_vi)

    return removed


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def export_static(
    model_path: Path,
    frames: int,
    batch: int = 1,
) -> Tuple[onnx.ModelProto, Dict[str, int]]:
    raw = onnx.load(str(model_path))
    frozen = freeze_input_shape(raw, batch=batch, frames=frames)
    graph = frozen.graph

    init_names = {i.name for i in graph.initializer}
    _, node_static = _mark_static(graph, init_names)

    boundary = _boundary_tensors(graph, node_static, init_names)
    folded_values = _evaluate(frozen, boundary, batch, frames)

    nodes_before = len(graph.node)
    init_before = len(graph.initializer)

    # replace the shape domain with the values it always produces
    survivors = [n for i, n in enumerate(graph.node) if not node_static[i]]
    folded = nodes_before - len(survivors)
    del graph.node[:]
    graph.node.extend(survivors)

    for name, value in folded_values.items():
        graph.initializer.append(numpy_helper.from_array(np.asarray(value), name))

    dead = _prune(graph)

    frozen.producer_name = "campp_acceleration.export_static"
    frozen.producer_version = f"frames={frames}"
    onnx.checker.check_model(frozen)

    return frozen, {
        "nodes_before": nodes_before,
        "nodes_after": len(graph.node),
        "folded": folded,
        "dead": dead,
        "initializers_before": init_before,
        "initializers_after": len(graph.initializer),
    }


# ---------------------------------------------------------------------------
# numerical equivalence against the original dynamic model
# ---------------------------------------------------------------------------


def compare_against_original(
    model_path: Path,
    static_model: onnx.ModelProto,
    frames: int,
    batch: int = 1,
    seed: int = 0,
) -> Tuple[float, float]:
    """Feed both models the same audio-shaped input; return (max|diff|, cosine)."""
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    sample = rng.standard_normal((batch, frames, 80)).astype(np.float32) * 5.0

    def run(model_bytes_or_path) -> np.ndarray:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            model_bytes_or_path, opts, providers=["CPUExecutionProvider"]
        )
        name = sess.get_inputs()[0].name
        return sess.run(None, {name: sample})[0]

    ref = run(str(model_path))
    got = run(static_model.SerializeToString())

    ref_f = np.asarray(ref, dtype=np.float64).ravel()
    got_f = np.asarray(got, dtype=np.float64).ravel()
    max_abs = float(np.max(np.abs(ref_f - got_f)))
    denom = np.linalg.norm(ref_f) * np.linalg.norm(got_f)
    cosine = float(ref_f @ got_f / denom) if denom > 0 else float("nan")
    return max_abs, cosine


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    ap.add_argument("--seconds", nargs="+", type=float, default=list(DEFAULT_BUCKETS))
    ap.add_argument("--frames", nargs="*", type=int, default=None,
                    help="frame counts to use directly instead of --seconds")
    ap.add_argument("--out-dir", default="campp_acceleration/out/static", type=Path)
    ap.add_argument("--no-compare", action="store_true",
                    help="skip the numerical check against the original model")
    args = ap.parse_args(argv)

    if not args.model.exists():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 1

    buckets: List[Tuple[int, float]]
    if args.frames:
        buckets = [(f, (f - 1) * 0.01 + 0.025) for f in args.frames]
    else:
        buckets = [(frames_for_seconds(s), s) for s in args.seconds]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    src_bytes = args.model.stat().st_size

    stats: List[ExportStats] = []
    for frames, seconds in buckets:
        print(f"[export_static] frames={frames} ({seconds:g}s) ...", flush=True)
        model, counts = export_static(args.model, frames=frames)
        out_path = args.out_dir / f"campp_static_{frames}.onnx"
        onnx.save(model, str(out_path))

        st = ExportStats(
            frames=frames,
            seconds=seconds,
            path=out_path,
            bytes_before=src_bytes,
            bytes_after=out_path.stat().st_size,
            **counts,
        )
        if not args.no_compare:
            st.max_abs_diff, st.cosine = compare_against_original(
                args.model, model, frames=frames
            )
        stats.append(st)

    print()
    print(f"{'frames':>7} {'nodes':>14} {'folded':>7} {'dead':>5} "
          f"{'init':>12} {'size':>10}  {'max|diff|':>10} {'cosine':>10}")
    for s in stats:
        diff = "-" if s.max_abs_diff is None else f"{s.max_abs_diff:.3e}"
        cos = "-" if s.cosine is None else f"{s.cosine:.9f}"
        print(f"{s.frames:>7} {s.nodes_before:>6} -> {s.nodes_after:<4} "
              f"{s.folded:>7} {s.dead:>5} "
              f"{s.initializers_before:>5} -> {s.initializers_after:<4} "
              f"{s.bytes_after / 1e6:>8.2f}M  {diff:>10} {cos:>10}")
    print(f"\nwrote {len(stats)} models to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
