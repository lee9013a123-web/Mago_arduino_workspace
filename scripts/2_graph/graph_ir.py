"""
CAM++ ONNX graph -> static execution IR.

The exported CAM++ model is a *dynamic shape* graph: every Reshape/Slice/Expand
inside the model recomputes its own shape at runtime out of Shape/Gather/Concat
nodes.  A static engine picks one input length (a bucket), resolves every shape
once on the PC, and only keeps the nodes that actually touch activations.

This module does that resolution and exposes it as a plain dict IR so the rest
of the toolchain (memory planner, kernel selection, DOT rendering) does not have
to know about ONNX protobufs.

Terminology used here:

  static tensor    a tensor whose value is fully determined by weights + the
                   frozen input shape (initializers, Constant outputs, and the
                   whole Shape/Gather/Concat shape-arithmetic domain)
  runtime tensor   everything else -- the actual activations
  static node      a node whose inputs are all static; it can be folded away
                   ahead of time and never executed on the board
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import onnx
from onnx import TensorProto, shape_inference

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

ELEM_SIZE = {
    TensorProto.FLOAT: 4,
    TensorProto.UINT8: 1,
    TensorProto.INT8: 1,
    TensorProto.UINT16: 2,
    TensorProto.INT16: 2,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.BOOL: 1,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.UINT32: 4,
    TensorProto.UINT64: 8,
}

# Nodes that manufacture shape information out of thin air.  Their outputs seed
# the "static" domain even though their input is a runtime activation.
SHAPE_SOURCE_OPS = {"Shape"}

# Ops that carry real arithmetic weight.  Anything outside this set is either
# shape plumbing or a pure data move.
COMPUTE_OPS = {
    "Conv", "QLinearConv", "ConvInteger",
    "Gemm", "MatMul", "QLinearMatMul", "MatMulInteger",
    "BatchNormalization", "InstanceNormalization", "LayerNormalization",
    "Relu", "Sigmoid", "Tanh", "Softmax", "Erf", "HardSigmoid",
    "Add", "Sub", "Mul", "Div", "Pow", "Sqrt",
    "ReduceMean", "ReduceSum", "ReduceMax", "ReduceProd",
    "AveragePool", "MaxPool", "GlobalAveragePool",
    "QuantizeLinear", "DequantizeLinear",
}

# Pure data movement -- free in a static engine if the memory planner aliases
# the buffers instead of copying.
MOVE_OPS = {
    "Reshape", "Transpose", "Concat", "Slice", "Squeeze", "Unsqueeze",
    "Identity", "Expand", "Split", "Pad", "Cast", "Gather", "Flatten",
}

FBANK_FRAME_LENGTH_MS = 25.0
FBANK_FRAME_SHIFT_MS = 10.0


def frames_for_seconds(seconds: float, sample_rate: int = 16000) -> int:
    """Kaldi/torchaudio fbank frame count for a given clip length.

    Mirrors ``extract_fbank`` in pipeline_experiment/core.py (25 ms window,
    10 ms hop, snip_edges=True).
    """
    n = int(round(seconds * sample_rate))
    win = int(round(FBANK_FRAME_LENGTH_MS * sample_rate / 1000.0))
    hop = int(round(FBANK_FRAME_SHIFT_MS * sample_rate / 1000.0))
    if n < win:
        return 0
    return (n - win) // hop + 1


# ---------------------------------------------------------------------------
# IR dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TensorInfo:
    name: str
    dtype: int = TensorProto.UNDEFINED
    shape: Optional[List[int]] = None      # None => shape inference failed
    is_initializer: bool = False
    is_static: bool = False
    producer: Optional[int] = None         # node index
    consumers: List[int] = field(default_factory=list)

    @property
    def elems(self) -> Optional[int]:
        if self.shape is None or any(d < 0 for d in self.shape):
            return None
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> Optional[int]:
        e = self.elems
        if e is None:
            return None
        return e * ELEM_SIZE.get(self.dtype, 4)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": onnx.TensorProto.DataType.Name(self.dtype)
            if self.dtype else "UNDEFINED",
            "shape": self.shape,
            "bytes": self.nbytes,
            "is_initializer": self.is_initializer,
            "is_static": self.is_static,
            "producer": self.producer,
            "consumers": self.consumers,
        }


@dataclass
class NodeInfo:
    index: int
    name: str
    op_type: str
    scope: str
    inputs: List[str]
    outputs: List[str]
    is_static: bool = False
    kind: str = "compute"                  # compute | move | shape | other
    macs: int = 0
    weight_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "op_type": self.op_type,
            "scope": self.scope,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "is_static": self.is_static,
            "kind": self.kind,
            "macs": self.macs,
            "weight_bytes": self.weight_bytes,
        }


@dataclass
class GraphIR:
    model_path: str
    batch: int
    frames: int
    seconds: float
    opset: int
    nodes: List[NodeInfo]
    tensors: Dict[str, TensorInfo]
    inputs: List[str]
    outputs: List[str]

    # -- convenience views ---------------------------------------------------

    @property
    def runtime_nodes(self) -> List[NodeInfo]:
        return [n for n in self.nodes if not n.is_static]

    @property
    def static_nodes(self) -> List[NodeInfo]:
        return [n for n in self.nodes if n.is_static]

    def op_histogram(self, runtime_only: bool = False) -> "OrderedDict[str, int]":
        src = self.runtime_nodes if runtime_only else self.nodes
        return OrderedDict(Counter(n.op_type for n in src).most_common())

    def total_macs(self) -> int:
        return sum(n.macs for n in self.nodes)

    def unresolved_tensors(self) -> List[str]:
        return [
            t.name for t in self.tensors.values()
            if not t.is_static and t.shape is None and t.producer is not None
        ]

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "batch": self.batch,
            "frames": self.frames,
            "seconds": self.seconds,
            "opset": self.opset,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": [n.to_dict() for n in self.nodes],
            "tensors": {k: v.to_dict() for k, v in self.tensors.items()},
        }


# ---------------------------------------------------------------------------
# shape freezing
# ---------------------------------------------------------------------------


def freeze_input_shape(
    model: onnx.ModelProto,
    batch: int,
    frames: int,
) -> onnx.ModelProto:
    """Replace the symbolic input dims with concrete values and re-infer.

    ``batch_size`` and any frame-length dim (``frame_num``/``T``/...) get pinned;
    everything already concrete is left alone.  Existing value_info is dropped
    first, otherwise stale symbolic entries win over the fresh inference.
    """
    frozen = onnx.ModelProto()
    frozen.CopyFrom(model)
    model = frozen
    graph = model.graph

    for inp in graph.input:
        dims = inp.type.tensor_type.shape.dim
        for pos, dim in enumerate(dims):
            if dim.HasField("dim_value"):
                continue
            param = dim.dim_param or ""
            if pos == 0 or "batch" in param.lower():
                dim.Clear()
                dim.dim_value = batch
            else:
                dim.Clear()
                dim.dim_value = frames

    del graph.value_info[:]
    for out in graph.output:
        for dim in out.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value") and dim.dim_param:
                if "batch" in dim.dim_param.lower():
                    dim.Clear()
                    dim.dim_value = batch

    # data_prop lets the inference engine constant-fold Shape/Gather/Concat
    # chains, which is exactly what resolves the dynamic Reshape targets.
    model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)

    # Stock ONNX inference gives up at the first Reshape whose target comes out
    # of a Shape->Slice->Concat chain, and CAM++'s CAM layers are full of them.
    # ORT's symbolic engine folds those, so run it as a second pass.
    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        model = SymbolicShapeInference.infer_shapes(
            model, auto_merge=True, guess_output_rank=False,
        )
    except Exception as exc:                      # pragma: no cover - best effort
        print(f"[graph_ir] symbolic shape inference skipped: {exc}")

    return model


# ---------------------------------------------------------------------------
# static-domain propagation
# ---------------------------------------------------------------------------


def _mark_static(
    graph: onnx.GraphProto,
    initializers: Sequence[str],
) -> Tuple[set, List[bool]]:
    """Forward-propagate the 'value is known at compile time' property.

    Returns (static_tensor_names, per-node static flags).
    """
    static: set = set(initializers)
    node_static: List[bool] = []

    for node in graph.node:
        if node.op_type in SHAPE_SOURCE_OPS:
            # Shape(x) is compile-time known once the input shape is frozen,
            # regardless of whether x itself is an activation.
            is_static = True
        else:
            is_static = all(
                (inp == "") or (inp in static) for inp in node.input
            )
        node_static.append(is_static)
        if is_static:
            static.update(o for o in node.output if o)

    return static, node_static


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------


def _conv_macs(
    node: onnx.NodeProto,
    tensors: Dict[str, TensorInfo],
) -> int:
    """MAC count for Conv / QLinearConv.

    QLinearConv input order is (x, x_scale, x_zp, w, w_scale, w_zp, y_scale,
    y_zp, [B]); plain Conv is (x, w, [B]).
    """
    w_name = node.input[3] if node.op_type == "QLinearConv" else node.input[1]
    w = tensors.get(w_name)
    out = tensors.get(node.output[0]) if node.output else None
    if w is None or w.shape is None or out is None or out.shape is None:
        return 0
    if any(d < 0 for d in out.shape) or len(w.shape) < 2:
        return 0

    # w: (M, C/group, *kernel)
    cin_per_group = w.shape[1]
    ksize = 1
    for d in w.shape[2:]:
        ksize *= d
    # out: (N, M, *spatial)
    spatial = 1
    for d in out.shape[2:]:
        spatial *= d
    batch = out.shape[0] if out.shape else 1
    m = out.shape[1] if len(out.shape) > 1 else w.shape[0]
    return int(batch * m * spatial * cin_per_group * ksize)


def _matmul_macs(
    node: onnx.NodeProto,
    tensors: Dict[str, TensorInfo],
) -> int:
    a = tensors.get(node.input[0])
    out = tensors.get(node.output[0]) if node.output else None
    if a is None or a.shape is None or out is None or out.shape is None:
        return 0
    if any(d < 0 for d in out.shape) or not a.shape:
        return 0
    k = a.shape[-1]
    n = 1
    for d in out.shape:
        n *= d
    return int(n * k)


def _classify(op_type: str) -> str:
    if op_type in COMPUTE_OPS:
        return "compute"
    if op_type in MOVE_OPS:
        return "move"
    if op_type in ("Shape", "ConstantOfShape", "Equal", "Where", "Range",
                   "NonZero", "Constant"):
        return "shape"
    return "other"


_SCOPE_RE = re.compile(r"^/?(.*)/[^/]+$")


def scope_of(node_name: str) -> str:
    """Module path of a node, from the exporter-generated ``/a/b/c`` name."""
    if not node_name or "/" not in node_name:
        return ""
    m = _SCOPE_RE.match(node_name)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def build_ir(
    model_path: Path,
    frames: int,
    batch: int = 1,
    seconds: Optional[float] = None,
) -> GraphIR:
    raw = onnx.load(str(model_path))
    model = freeze_input_shape(raw, batch=batch, frames=frames)
    graph = model.graph

    tensors: Dict[str, TensorInfo] = {}

    def touch(name: str) -> TensorInfo:
        t = tensors.get(name)
        if t is None:
            t = TensorInfo(name=name)
            tensors[name] = t
        return t

    for init in graph.initializer:
        t = touch(init.name)
        t.dtype = init.data_type
        t.shape = list(init.dims)
        t.is_initializer = True
        t.is_static = True

    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        t = touch(vi.name)
        tt = vi.type.tensor_type
        t.dtype = tt.elem_type
        if not tt.HasField("shape"):
            # no shape field at all == unknown rank, which is *not* a scalar
            t.shape = None
            continue
        dims = []
        ok = True
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            else:
                ok = False
                break
        t.shape = dims if ok else None

    # Constant nodes hold their value in an attribute, not in value_info.
    for node in graph.node:
        if node.op_type == "Constant" and node.output:
            t = touch(node.output[0])
            for attr in node.attribute:
                if attr.name == "value":
                    t.dtype = attr.t.data_type
                    t.shape = list(attr.t.dims)

    static_names, node_static = _mark_static(
        graph, [i.name for i in graph.initializer]
    )

    nodes: List[NodeInfo] = []
    for idx, node in enumerate(graph.node):
        info = NodeInfo(
            index=idx,
            name=node.name or f"{node.op_type}_{idx}",
            op_type=node.op_type,
            scope=scope_of(node.name),
            inputs=[i for i in node.input if i],
            outputs=[o for o in node.output if o],
            is_static=node_static[idx],
            kind=_classify(node.op_type),
        )
        for name in info.inputs:
            touch(name).consumers.append(idx)
        for name in info.outputs:
            t = touch(name)
            t.producer = idx

        if node.op_type in ("Conv", "QLinearConv"):
            info.macs = _conv_macs(node, tensors)
        elif node.op_type in ("MatMul", "Gemm"):
            info.macs = _matmul_macs(node, tensors)

        info.weight_bytes = sum(
            tensors[n].nbytes or 0
            for n in info.inputs
            if n in tensors and tensors[n].is_initializer
        )
        nodes.append(info)

    for name in static_names:
        touch(name).is_static = True

    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 0)

    return GraphIR(
        model_path=str(model_path),
        batch=batch,
        frames=frames,
        seconds=seconds if seconds is not None else round(frames * 0.01 + 0.015, 3),
        opset=opset,
        nodes=nodes,
        tensors=tensors,
        inputs=[i.name for i in graph.input],
        outputs=[o.name for o in graph.output],
    )
