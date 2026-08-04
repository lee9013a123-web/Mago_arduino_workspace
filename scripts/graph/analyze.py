"""
Analysis passes over the CAM++ static IR.

Two things the static engine needs to know before any kernel gets written:

  1. which node chains can collapse into one fused kernel
     (-> how many intermediate tensors never have to reach LPDDR)
  2. how big the Tensor Arena has to be
     (-> the single up-front allocation that replaces every malloc/free)
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from graph_ir import GraphIR, NodeInfo, TensorInfo


# ---------------------------------------------------------------------------
# fusion candidates
# ---------------------------------------------------------------------------


@dataclass
class FusionPattern:
    """A named kernel we intend to write, and what it is actually allowed to do.

    The names matter: several of these look like textbook fusions but are not,
    and getting the name wrong would put the wrong kernel in the plan.
    """
    name: str
    ops: Tuple[str, ...]       # contiguous op subsequence that identifies it
    note: str


# Ordered most-specific first -- the first match wins.
FUSION_PATTERNS: Tuple[FusionPattern, ...] = (
    FusionPattern(
        name="DenseConcatBNReluQuantConvPack",
        ops=("Concat", "BatchNormalization", "Relu", "QuantizeLinear", "QLinearConv"),
        note=(
            "Dense connection followed by the same packing epilogue as "
            "BNAffineReluQuantConvPack. The Concat itself is removable "
            "separately: preallocate the full channel block once and have each "
            "unit write its output into its own slice, so nothing is copied."
        ),
    ),
    FusionPattern(
        name="BNAffineReluQuantConvPack",
        ops=("BatchNormalization", "Relu", "QuantizeLinear", "QLinearConv"),
        note=(
            "NOT a BN-into-conv-weight fold. A ReLU sits between the BN and the "
            "conv, and BN's own input is the dense Concat, so BN cannot be "
            "folded forward into the conv weights nor backward into the "
            "producers. What fuses is the input side: BN affine + ReLU + "
            "quantize become one pass that writes the int8 conv input directly, "
            "so neither the float BN output nor the float ReLU output is ever "
            "materialized."
        ),
    ),
    FusionPattern(
        name="SegmentContextConsumerFusion",
        ops=("AveragePool", "Unsqueeze", "Expand", "Reshape", "Slice"),
        note=(
            "Segment-level context broadcast back onto the frame axis. This "
            "collapses to address arithmetic only if the consumer is fused in "
            "too -- the Add, and through it the CAM mask conv. Deleting Expand "
            "while still handing the result to a plain ORT tensor does not "
            "work: the consumer requires a materialized contiguous tensor, "
            "which is exactly the write we are trying to avoid."
        ),
    ),
    FusionPattern(
        name="CAMMaskGen",
        ops=("Add", "QuantizeLinear", "QLinearConv", "DequantizeLinear", "Relu",
             "QuantizeLinear", "QLinearConv", "DequantizeLinear", "Sigmoid"),
        note=(
            "The whole CAM mask bottleneck (C -> C/2 -> C/4 -> sigmoid). Small "
            "channel counts, so kernel launch overhead dominates the arithmetic."
        ),
    ),
    FusionPattern(
        name="StatsPoolingStd",
        ops=("Sub", "Mul", "ReduceMean", "Mul", "Div", "Sqrt"),
        note="Standard-deviation branch of the statistics pooling layer.",
    ),
    FusionPattern(
        name="ConvReluRequant",
        ops=("QLinearConv", "DequantizeLinear", "Relu", "QuantizeLinear"),
        note=(
            "Dequantize -> ReLU -> requantize around a conv output. ReLU on an "
            "affine-quantized tensor is a clamp at the zero point, so this is "
            "a scale adjustment plus a clamp folded into the conv epilogue."
        ),
    ),
    FusionPattern(
        name="RequantBoundary",
        ops=("QuantizeLinear", "QLinearConv", "DequantizeLinear"),
        note=(
            "Pure quantization boundary. Once neighbouring kernels agree on a "
            "scale it disappears entirely."
        ),
    ),
)


def _match_pattern(ops: List[str]) -> Optional[FusionPattern]:
    for pat in FUSION_PATTERNS:
        n = len(pat.ops)
        for i in range(len(ops) - n + 1):
            if tuple(ops[i:i + n]) == pat.ops:
                return pat
    return None


@dataclass
class Chain:
    """A maximal straight-line run of runtime nodes with no fan-out."""
    node_indices: List[int]
    ops: List[str]
    scope: str
    interm_bytes: int          # bytes written+reread purely to hand over

    @property
    def signature(self) -> str:
        return " -> ".join(self.ops)

    @property
    def pattern(self) -> Optional[FusionPattern]:
        return _match_pattern(self.ops)


def find_chains(ir: GraphIR, min_len: int = 2) -> List[Chain]:
    """Split the runtime graph into maximal fusable straight-line segments.

    A node joins the current chain when its producer's output feeds exactly one
    runtime consumer and that consumer is this node.  Anything with fan-out ends
    a chain -- a fused kernel cannot keep a value in registers if someone else
    needs it later.
    """
    runtime = ir.runtime_nodes
    runtime_idx = {n.index: n for n in runtime}

    # runtime consumers per tensor
    consumers: Dict[str, List[int]] = {}
    for n in runtime:
        for name in n.inputs:
            consumers.setdefault(name, []).append(n.index)

    # the single runtime successor of a node, if it has exactly one
    def sole_successor(n: NodeInfo) -> Optional[int]:
        succ = set()
        for out in n.outputs:
            if out in ir.outputs:
                return None
            for c in consumers.get(out, ()):
                succ.add(c)
        if len(succ) != 1:
            return None
        return next(iter(succ))

    # a node can only be appended if *all* of its runtime inputs come from the
    # predecessor (or are static/weights)
    def joins(pred: NodeInfo, node: NodeInfo) -> bool:
        pred_outs = set(pred.outputs)
        for name in node.inputs:
            t = ir.tensors.get(name)
            if t is None or t.is_static or t.is_initializer:
                continue
            if name not in pred_outs:
                return False
        return True

    visited: set = set()
    chains: List[Chain] = []

    for n in runtime:
        if n.index in visited:
            continue
        chain = [n]
        visited.add(n.index)
        cur = n
        while True:
            nxt_idx = sole_successor(cur)
            if nxt_idx is None or nxt_idx in visited:
                break
            nxt = runtime_idx.get(nxt_idx)
            if nxt is None or not joins(cur, nxt):
                break
            chain.append(nxt)
            visited.add(nxt.index)
            cur = nxt

        if len(chain) < min_len:
            continue

        interm = 0
        for link in chain[:-1]:
            for out in link.outputs:
                t = ir.tensors.get(out)
                if t is not None and t.nbytes:
                    interm += t.nbytes

        chains.append(Chain(
            node_indices=[c.index for c in chain],
            ops=[c.op_type for c in chain],
            scope=_common_scope([c.scope for c in chain]),
            interm_bytes=interm,
        ))

    return chains


def _common_scope(scopes: List[str]) -> str:
    if not scopes:
        return ""
    parts = [s.split("/") for s in scopes]
    common: List[str] = []
    for group in zip(*parts):
        if len(set(group)) == 1:
            common.append(group[0])
        else:
            break
    return "/".join(common)


def chain_summary(chains: List[Chain]) -> "OrderedDict[str, dict]":
    """Group chains by op signature -- the fused-kernel shopping list."""
    agg: Dict[str, dict] = {}
    for c in chains:
        pat = c.pattern
        e = agg.setdefault(c.signature, {
            "signature": c.signature,
            "kernel": pat.name if pat else None,
            "note": pat.note if pat else None,
            "count": 0,
            "nodes_folded": 0,
            "interm_bytes": 0,
            "example_scope": c.scope,
        })
        e["count"] += 1
        e["nodes_folded"] += len(c.node_indices) - 1
        e["interm_bytes"] += c.interm_bytes
    return OrderedDict(
        sorted(agg.items(), key=lambda kv: -kv[1]["interm_bytes"])
    )


# ---------------------------------------------------------------------------
# tensor arena planning
# ---------------------------------------------------------------------------


@dataclass
class ArenaPlan:
    arena_bytes: int              # greedy allocator result -- what we'd reserve
    peak_live_bytes: int          # theoretical floor (perfect packing)
    naive_bytes: int              # every activation allocated separately
    weight_bytes: int
    num_buffers: int
    offsets: Dict[str, Tuple[int, int]]   # tensor -> (offset, size)
    peak_step: int
    peak_tensors: List[Tuple[str, int]]

    @property
    def fragmentation(self) -> float:
        if self.peak_live_bytes == 0:
            return 0.0
        return self.arena_bytes / self.peak_live_bytes - 1.0

    def to_dict(self) -> dict:
        return {
            "arena_bytes": self.arena_bytes,
            "peak_live_bytes": self.peak_live_bytes,
            "naive_bytes": self.naive_bytes,
            "weight_bytes": self.weight_bytes,
            "num_buffers": self.num_buffers,
            "fragmentation": round(self.fragmentation, 4),
            "peak_step": self.peak_step,
            "peak_tensors": self.peak_tensors,
        }


def plan_arena(ir: GraphIR, align: int = 64) -> ArenaPlan:
    """Offline offset assignment for every activation tensor.

    Liveness is computed over the *runtime* execution order.  Buffers are placed
    greedily in decreasing-size order (the standard TFLite-Micro style planner):
    for each tensor, take the lowest offset that does not overlap an already
    placed tensor whose lifetime intersects.
    """
    runtime = ir.runtime_nodes
    step_of = {n.index: step for step, n in enumerate(runtime)}
    nsteps = len(runtime)

    first_def: Dict[str, int] = {}
    last_use: Dict[str, int] = {}

    graph_inputs = set(ir.inputs)
    graph_outputs = set(ir.outputs)

    for step, n in enumerate(runtime):
        for name in n.inputs:
            t = ir.tensors.get(name)
            if t is None or t.is_static or t.is_initializer:
                continue
            last_use[name] = step
            first_def.setdefault(name, 0 if name in graph_inputs else step)
        for name in n.outputs:
            t = ir.tensors.get(name)
            if t is None or t.is_static:
                continue
            first_def.setdefault(name, step)
            last_use.setdefault(name, step)

    for name in graph_outputs:
        if name in first_def:
            last_use[name] = nsteps - 1

    sizes: Dict[str, int] = {}
    for name in first_def:
        t = ir.tensors.get(name)
        nb = t.nbytes if t is not None else None
        if not nb:
            continue
        sizes[name] = _align(nb, align)

    # --- theoretical floor: max simultaneous live bytes ---------------------
    live_at = [0] * max(nsteps, 1)
    for name, size in sizes.items():
        for s in range(first_def[name], last_use[name] + 1):
            live_at[s] += size
    peak_live = max(live_at) if live_at else 0
    peak_step = live_at.index(peak_live) if live_at else 0
    peak_tensors = sorted(
        ((name, size) for name, size in sizes.items()
         if first_def[name] <= peak_step <= last_use[name]),
        key=lambda kv: -kv[1],
    )[:12]

    # --- greedy placement ---------------------------------------------------
    order = sorted(sizes, key=lambda n: (-sizes[n], first_def[n]))
    placed: List[Tuple[int, int, int, int]] = []   # (off, end, start_step, end_step)
    offsets: Dict[str, Tuple[int, int]] = {}
    arena = 0

    for name in order:
        size = sizes[name]
        s, e = first_def[name], last_use[name]
        blocked = sorted(
            (off, off_end) for off, off_end, ps, pe in placed
            if not (pe < s or ps > e)
        )
        offset = 0
        for b_start, b_end in blocked:
            if offset + size <= b_start:
                break
            offset = max(offset, b_end)
        offsets[name] = (offset, size)
        placed.append((offset, offset + size, s, e))
        arena = max(arena, offset + size)

    weight_bytes = sum(
        t.nbytes or 0 for t in ir.tensors.values() if t.is_initializer
    )

    return ArenaPlan(
        arena_bytes=arena,
        peak_live_bytes=peak_live,
        naive_bytes=sum(sizes.values()),
        weight_bytes=weight_bytes,
        num_buffers=len(sizes),
        offsets=offsets,
        peak_step=peak_step,
        peak_tensors=peak_tensors,
    )


def _align(n: int, align: int) -> int:
    return (n + align - 1) // align * align


# ---------------------------------------------------------------------------
# scope-level rollup
# ---------------------------------------------------------------------------


@dataclass
class ScopeStat:
    scope: str
    nodes: int
    runtime_nodes: int
    macs: int
    weight_bytes: int
    ops: "OrderedDict[str, int]"

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "nodes": self.nodes,
            "runtime_nodes": self.runtime_nodes,
            "macs": self.macs,
            "weight_bytes": self.weight_bytes,
            "ops": dict(self.ops),
        }


def rollup_scopes(ir: GraphIR, depth: int = 2) -> List[ScopeStat]:
    """Aggregate node stats by the first ``depth`` levels of the module path."""
    buckets: Dict[str, List[NodeInfo]] = {}
    for n in ir.nodes:
        key = "/".join(n.scope.split("/")[:depth]) if n.scope else "(root)"
        buckets.setdefault(key, []).append(n)

    stats = []
    for scope, group in buckets.items():
        stats.append(ScopeStat(
            scope=scope,
            nodes=len(group),
            runtime_nodes=sum(1 for n in group if not n.is_static),
            macs=sum(n.macs for n in group),
            weight_bytes=sum(n.weight_bytes for n in group),
            ops=OrderedDict(Counter(n.op_type for n in group).most_common()),
        ))
    stats.sort(key=lambda s: -s.macs)
    return stats
