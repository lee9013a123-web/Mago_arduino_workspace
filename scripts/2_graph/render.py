"""
Graph rendering for the CAM++ static IR.

Two levels, because 3180 nodes is not a picture:

  block level   scopes collapsed to N path segments (``xvector/block2``) --
                the dataflow skeleton, where the dense connections show up
  op level      one scope expanded node by node -- what a fused kernel would
                have to swallow

Emits Graphviz DOT (offline rendering) and Mermaid (renders natively in an
Artifact page, no toolchain needed).
"""

from __future__ import annotations

import html
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from graph_ir import GraphIR, NodeInfo


# ---------------------------------------------------------------------------
# block-level graph
# ---------------------------------------------------------------------------


@dataclass
class BlockNode:
    key: str
    nodes: int = 0
    runtime_nodes: int = 0
    macs: int = 0
    weight_bytes: int = 0
    ops: Dict[str, int] = field(default_factory=dict)


@dataclass
class BlockGraph:
    depth: int
    blocks: "OrderedDict[str, BlockNode]"
    edges: "OrderedDict[Tuple[str, str], int]"     # (src, dst) -> bytes moved


def _scope_key(node: NodeInfo, depth: int) -> str:
    if not node.scope:
        return "(root)"
    return "/".join(node.scope.split("/")[:depth])


def build_block_graph(
    ir: GraphIR,
    depth: int = 2,
    runtime_only: bool = True,
) -> BlockGraph:
    src_nodes = ir.runtime_nodes if runtime_only else ir.nodes
    keys: Dict[int, str] = {n.index: _scope_key(n, depth) for n in src_nodes}

    blocks: "OrderedDict[str, BlockNode]" = OrderedDict()
    for n in src_nodes:
        key = keys[n.index]
        b = blocks.get(key)
        if b is None:
            b = blocks[key] = BlockNode(key=key)
        b.nodes += 1
        if not n.is_static:
            b.runtime_nodes += 1
        b.macs += n.macs
        b.weight_bytes += n.weight_bytes
        b.ops[n.op_type] = b.ops.get(n.op_type, 0) + 1

    edges: "OrderedDict[Tuple[str, str], int]" = OrderedDict()
    for name, t in ir.tensors.items():
        if t.is_initializer or t.is_static or t.producer is None:
            continue
        src = keys.get(t.producer)
        if src is None:
            continue
        nbytes = t.nbytes or 0
        for c in t.consumers:
            dst = keys.get(c)
            if dst is None or dst == src:
                continue
            edges[(src, dst)] = edges.get((src, dst), 0) + nbytes

    # graph input edge so the picture has an entry point
    for gi in ir.inputs:
        t = ir.tensors.get(gi)
        if t is None:
            continue
        for c in t.consumers:
            dst = keys.get(c)
            if dst is not None:
                edges[("input", dst)] = edges.get(("input", dst), 0) + (t.nbytes or 0)
    for go in ir.outputs:
        t = ir.tensors.get(go)
        if t is None or t.producer is None:
            continue
        src = keys.get(t.producer)
        if src is not None:
            edges[(src, "output")] = edges.get((src, "output"), 0) + (t.nbytes or 0)

    return BlockGraph(depth=depth, blocks=blocks, edges=edges)


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def _fmt_bytes(n: int) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.0f} KB"
    return f"{n} B"


def _fmt_macs(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} G"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} M"
    if n >= 1_000:
        return f"{n / 1e3:.1f} K"
    return str(n)


def _fmt_shape(shape: Optional[List[int]]) -> str:
    if shape is None:
        return "?"
    return "x".join(str(d) for d in shape)


def _scope_nodes(
    ir: GraphIR,
    scope_prefix: str,
    runtime_only: bool,
) -> List[NodeInfo]:
    """Nodes inside a scope, matched on path segments.

    Plain ``startswith`` would pull ``tdnnd10..19`` into ``tdnnd1``.
    """
    prefix = scope_prefix.rstrip("/")
    src = ir.runtime_nodes if runtime_only else ir.nodes
    return [
        n for n in src
        if n.scope == prefix or n.scope.startswith(prefix + "/")
    ]


_KIND_COLOR = {
    "compute": "#3b7dd8",
    "move": "#8a8f98",
    "shape": "#c9a227",
    "other": "#a05ec4",
}


# ---------------------------------------------------------------------------
# DOT
# ---------------------------------------------------------------------------


def block_graph_to_dot(bg: BlockGraph, title: str = "CAM++") -> str:
    lines = [
        "digraph campp {",
        "  rankdir=TB;",
        '  graph [fontname="Helvetica", labelloc="t", '
        f'label="{html.escape(title)}"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fillcolor="#eef3fb", color="#3b7dd8"];',
        '  edge [fontname="Helvetica", fontsize=9, color="#666666"];',
        '  input [shape=ellipse, fillcolor="#e8f5e9", color="#2e7d32"];',
        '  output [shape=ellipse, fillcolor="#fdecea", color="#c62828"];',
    ]
    for key, b in bg.blocks.items():
        label = (
            f"{key}\\n{b.runtime_nodes} ops | {_fmt_macs(b.macs)} MAC\\n"
            f"w {_fmt_bytes(b.weight_bytes)}"
        )
        lines.append(f'  "{key}" [label="{label}"];')
    for (src, dst), nbytes in bg.edges.items():
        lines.append(
            f'  "{src}" -> "{dst}" [label="{_fmt_bytes(nbytes)}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def op_graph_to_dot(
    ir: GraphIR,
    scope_prefix: str,
    runtime_only: bool = True,
    title: Optional[str] = None,
) -> str:
    nodes = _scope_nodes(ir, scope_prefix, runtime_only)
    keep = {n.index for n in nodes}

    lines = [
        "digraph scope {",
        "  rankdir=TB;",
        '  graph [fontname="Helvetica", labelloc="t", '
        f'label="{html.escape(title or scope_prefix)}"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8, color="#666666"];',
    ]
    for n in nodes:
        color = _KIND_COLOR.get(n.kind, "#8a8f98")
        out_shape = _fmt_shape(
            ir.tensors[n.outputs[0]].shape if n.outputs else None
        )
        short = n.name.split("/")[-1]
        extra = f"\\n{_fmt_macs(n.macs)} MAC" if n.macs else ""
        lines.append(
            f'  n{n.index} [label="{n.op_type}\\n{short}\\n[{out_shape}]{extra}", '
            f'color="{color}", fillcolor="{color}22"];'
        )
    for n in nodes:
        for name in n.outputs:
            t = ir.tensors.get(name)
            if t is None:
                continue
            for c in t.consumers:
                if c in keep:
                    lines.append(
                        f'  n{n.index} -> n{c} [label="{_fmt_shape(t.shape)}"];'
                    )
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mermaid
# ---------------------------------------------------------------------------


def _mm_id(key: str) -> str:
    return "b_" + "".join(ch if ch.isalnum() else "_" for ch in key)


def block_graph_to_mermaid(bg: BlockGraph) -> str:
    lines = ["flowchart TB"]
    for key, b in bg.blocks.items():
        label = (
            f"{key}<br/>{b.runtime_nodes} ops · {_fmt_macs(b.macs)} MAC"
            f"<br/>w {_fmt_bytes(b.weight_bytes)}"
        )
        lines.append(f'  {_mm_id(key)}["{label}"]')
    lines.append('  b_input(["feature"])')
    lines.append('  b_output(["embedding"])')
    for (src, dst), nbytes in bg.edges.items():
        lines.append(
            f"  {_mm_id(src)} -->|{_fmt_bytes(nbytes)}| {_mm_id(dst)}"
        )
    return "\n".join(lines)


def op_graph_to_mermaid(
    ir: GraphIR,
    scope_prefix: str,
    runtime_only: bool = True,
) -> str:
    nodes = _scope_nodes(ir, scope_prefix, runtime_only)
    keep = {n.index for n in nodes}

    lines = ["flowchart TB"]
    for n in nodes:
        out_shape = _fmt_shape(
            ir.tensors[n.outputs[0]].shape if n.outputs else None
        )
        short = n.name.split("/")[-1]
        lines.append(f'  n{n.index}["{n.op_type}<br/>{short}<br/>{out_shape}"]')
    for n in nodes:
        for name in n.outputs:
            t = ir.tensors.get(name)
            if t is None:
                continue
            for c in t.consumers:
                if c in keep:
                    lines.append(f"  n{n.index} --> n{c}")
    for n in nodes:
        cls = n.kind
        lines.append(f"  class n{n.index} k_{cls};")
    lines.append("  classDef k_compute fill:#3b7dd822,stroke:#3b7dd8;")
    lines.append("  classDef k_move fill:#8a8f9822,stroke:#8a8f98;")
    lines.append("  classDef k_shape fill:#c9a22722,stroke:#c9a227;")
    lines.append("  classDef k_other fill:#a05ec422,stroke:#a05ec4;")
    return "\n".join(lines)
