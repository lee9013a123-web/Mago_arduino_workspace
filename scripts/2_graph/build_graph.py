#!/usr/bin/env python3
"""
Stage 0 of the CAM++ acceleration work: turn the dynamic-shape ONNX export into
a static, per-bucket execution graph and report what that buys us.

Usage
-----
    venv/bin/python campp_acceleration/graph/build_graph.py \
        --model models/campplus_int8_static_qop.onnx \
        --seconds 1 3 5 10 \
        --expand xvector/block2/tdnnd1 \
        --verify

Outputs (under --out-dir):
    ir_<sec>s.json          full static IR: nodes, tensors, shapes, arena offsets
    graph_report.md         human-readable summary across all buckets
    blocks_<sec>s.dot|mmd   block-level dataflow graph
    scope_<name>.dot|mmd    one expanded scope, op by op
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyze  # noqa: E402
import graph_ir as gir  # noqa: E402
import render  # noqa: E402

DEFAULT_MODEL = "models/campplus_int8_static_qop.onnx"
DEFAULT_BUCKETS = (1.0, 3.0, 5.0, 10.0)


def _mb(n: int) -> str:
    return f"{n / (1 << 20):.2f} MB"


def _build_one(
    model_path: Path,
    seconds: float,
    out_dir: Path,
    expand: List[str],
    do_verify: bool,
) -> dict:
    frames = gir.frames_for_seconds(seconds)
    ir = gir.build_ir(model_path, frames=frames, seconds=seconds)

    chains = analyze.find_chains(ir)
    summary = analyze.chain_summary(chains)
    arena = analyze.plan_arena(ir)
    scopes = analyze.rollup_scopes(ir, depth=2)

    payload = ir.to_dict()
    payload["arena"] = arena.to_dict()
    payload["arena"]["offsets"] = {
        k: {"offset": v[0], "size": v[1]} for k, v in arena.offsets.items()
    }
    payload["fusion_candidates"] = list(summary.values())
    payload["scopes"] = [s.to_dict() for s in scopes]

    tag = f"{seconds:g}s"
    (out_dir / f"ir_{tag}.json").write_text(json.dumps(payload, indent=1))

    bg = render.build_block_graph(ir, depth=2)
    (out_dir / f"blocks_{tag}.dot").write_text(
        render.block_graph_to_dot(bg, title=f"CAM++ static graph @ {tag} ({frames} frames)")
    )
    (out_dir / f"blocks_{tag}.mmd").write_text(render.block_graph_to_mermaid(bg))

    for scope in expand:
        safe = scope.replace("/", "_")
        (out_dir / f"scope_{safe}_{tag}.dot").write_text(
            render.op_graph_to_dot(ir, scope, title=f"{scope} @ {tag}")
        )
        (out_dir / f"scope_{safe}_{tag}.mmd").write_text(
            render.op_graph_to_mermaid(ir, scope)
        )

    verify_line = "not run"
    if do_verify:
        import verify as verify_mod

        checked, mismatched, bad = verify_mod.verify_shapes(ir)
        verify_line = f"{checked} tensors checked, {mismatched} mismatched"
        if bad:
            verify_line += " | " + "; ".join(bad[:3])

    return {
        "seconds": seconds,
        "frames": frames,
        "ir": ir,
        "chains": chains,
        "summary": summary,
        "arena": arena,
        "scopes": scopes,
        "verify": verify_line,
    }


def _write_report(
    results: List[dict],
    model_path: Path,
    out_dir: Path,
    ref_seconds: float,
) -> Path:
    # The bucket-by-bucket table covers everything; the detail sections
    # (histogram, hotspots, fusion, peak tensors) are reported for one
    # reference bucket so the numbers in them are directly comparable.
    ref_result = min(results, key=lambda r: abs(r["seconds"] - ref_seconds))
    ref = ref_result["ir"]
    lines: List[str] = []
    a = lines.append

    a("# CAM++ static graph report")
    a("")
    a(f"- model: `{model_path}`")
    a(f"- opset: {ref.opset}, input `{ref.inputs[0]}` -> output `{ref.outputs[0]}`")
    a(f"- raw graph: **{len(ref.nodes)} nodes**, "
      f"{sum(1 for t in ref.tensors.values() if t.is_initializer)} initializers")
    a("")

    # ---- bucket table -----------------------------------------------------
    a("## Length buckets")
    a("")
    a("| bucket | fbank frames | nodes | runtime nodes | folded away | MACs | "
      "arena | weights |")
    a("|---|---|---|---|---|---|---|---|")
    for r in results:
        ir = r["ir"]
        arena = r["arena"]
        folded = len(ir.static_nodes)
        pct = 100.0 * folded / max(len(ir.nodes), 1)
        a(f"| {r['seconds']:g} s | {r['frames']} | {len(ir.nodes)} | "
          f"{len(ir.runtime_nodes)} | {folded} ({pct:.0f}%) | "
          f"{ir.total_macs() / 1e6:.0f} M | {_mb(arena.arena_bytes)} | "
          f"{_mb(arena.weight_bytes)} |")
    a("")
    a("`folded away` = nodes whose inputs are all compile-time known once the "
      "input length is frozen (Shape/Gather/Concat shape arithmetic, Constants). "
      "A static engine never executes them.")
    a("")

    # ---- op histogram -----------------------------------------------------
    a("## Op histogram (raw vs static)")
    a("")
    raw_hist = ref.op_histogram()
    rt_hist = ref.op_histogram(runtime_only=True)
    a(f"At {ref.seconds:g} s / {ref.frames} frames.")
    a("")
    a("| op | raw | after static shape resolution | removed |")
    a("|---|---|---|---|")
    for op, n in raw_hist.items():
        left = rt_hist.get(op, 0)
        a(f"| {op} | {n} | {left} | {n - left} |")
    a(f"| **total** | **{len(ref.nodes)}** | **{len(ref.runtime_nodes)}** | "
      f"**{len(ref.static_nodes)}** |")
    a("")

    # ---- where the compute is --------------------------------------------
    a("## Where the compute is")
    a("")
    a(f"At {ref.seconds:g} s, total {ref.total_macs() / 1e6:.0f} M MAC.")
    a("")
    a("| scope | nodes | runtime | MACs | share | weights |")
    a("|---|---|---|---|---|---|")
    total_macs = max(ref.total_macs(), 1)
    for s in ref_result["scopes"]:
        if s.nodes < 2 and s.macs == 0:
            continue
        a(f"| `{s.scope}` | {s.nodes} | {s.runtime_nodes} | "
          f"{s.macs / 1e6:.1f} M | {100.0 * s.macs / total_macs:.1f}% | "
          f"{s.weight_bytes / 1024:.0f} KB |")
    a("")

    # ---- fusion candidates ------------------------------------------------
    a("## Fusion candidates")
    a("")
    a("Maximal straight-line runs with no fan-out. Every node after the first "
      "in a run is a kernel launch and a round trip to LPDDR that a fused "
      "kernel removes.")
    a("")
    chains = ref_result["chains"]
    summary = ref_result["summary"]
    folded_ops = sum(v["nodes_folded"] for v in summary.values())
    a(f"- {len(chains)} fusable runs covering "
      f"{sum(len(c.node_indices) for c in chains)} of "
      f"{len(ref.runtime_nodes)} runtime nodes")
    a(f"- collapsing them removes **{folded_ops} kernel launches** "
      f"({100.0 * folded_ops / max(len(ref.runtime_nodes), 1):.0f}% of runtime nodes)")
    a("")
    a("| x | intermediate traffic | kernel | pattern |")
    a("|---|---|---|---|")
    for v in list(summary.values())[:20]:
        sig = v["signature"]
        if len(sig) > 150:
            sig = sig[:147] + "..."
        kernel = f"`{v['kernel']}`" if v["kernel"] else "—"
        a(f"| {v['count']} | {v['interm_bytes'] / (1 << 20):.2f} MB | {kernel} | "
          f"`{sig}` |")
    a("")

    named = OrderedDict()
    for v in summary.values():
        if v["kernel"] and v["kernel"] not in named:
            named[v["kernel"]] = v["note"]
    if named:
        a("### What each kernel is allowed to do")
        a("")
        for name, note in named.items():
            a(f"**`{name}`** — {note}")
            a("")

    # ---- memory plan ------------------------------------------------------
    a("## Tensor arena")
    a("")
    a("| bucket | naive (one buffer per tensor) | arena (offline planner) | "
      "theoretical floor | overhead | buffers |")
    a("|---|---|---|---|---|---|")
    for r in results:
        p = r["arena"]
        a(f"| {r['seconds']:g} s | {_mb(p.naive_bytes)} | {_mb(p.arena_bytes)} | "
          f"{_mb(p.peak_live_bytes)} | {100.0 * p.fragmentation:.1f}% | "
          f"{p.num_buffers} |")
    a("")
    p0 = ref_result["arena"]
    a(f"Largest live tensors at the peak step of the {ref_result['seconds']:g} s "
      "bucket:")
    a("")
    for name, size in p0.peak_tensors[:8]:
        a(f"- `{name}` — {size / 1024:.0f} KB")
    a("")

    # ---- verification -----------------------------------------------------
    a("## Shape verification")
    a("")
    for r in results:
        a(f"- {r['seconds']:g} s: {r['verify']}")
    a("")

    path = out_dir / "graph_report.md"
    path.write_text("\n".join(lines))
    return path


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    ap.add_argument("--seconds", nargs="+", type=float, default=list(DEFAULT_BUCKETS),
                    help="input length buckets to specialize for")
    ap.add_argument("--out-dir", default="campp_acceleration/out/graph", type=Path)
    ap.add_argument("--expand", nargs="*", default=["xvector/block2/tdnnd1"],
                    help="scopes to render op by op")
    ap.add_argument("--ref-seconds", type=float, default=3.0,
                    help="bucket the detail sections of the report are based on")
    ap.add_argument("--verify", action="store_true",
                    help="check every inferred shape against a real ORT run")
    args = ap.parse_args(argv)

    model_path = args.model
    if not model_path.exists():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sec in args.seconds:
        print(f"[build_graph] {sec:g}s ...", flush=True)
        results.append(_build_one(model_path, sec, out_dir, args.expand, args.verify))

    report = _write_report(results, model_path, out_dir, args.ref_seconds)
    print(f"[build_graph] wrote {report}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
