#!/usr/bin/env python3
"""Phase 3의 두 번째 실행 스크립트다.

고정 feature 입력을 bucket별 정적 ONNX에 넣고 ONNX Runtime의 모든 중간 Tensor와
최종 embedding을 저장한다. C Runtime과 같은 입력 바이트를 쓰도록 feature는
raw float32로도 함께 기록한다.

원시 출력은 runs/runtime/ort_reference에 저장한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SOURCE = ROOT / "src" / "python"

# (고정 frame 수, graph IR 파일 접미사)
BUCKETS: tuple[tuple[int, str], ...] = (
    (98, "1s"),
    (298, "3s"),
    (498, "5s"),
    (998, "10s"),
)

FEATURE_DIM = 80
INPUT_NAME = "feature"


def build_feature(frames: int, seed: int) -> np.ndarray:
    """bucket마다 재현 가능한 고정 feature를 만든다.

    실제 fbank 통계에 가깝도록 평균 0 근처, 표준편차 1 정도의 값을 쓴다.
    seed를 bucket과 무관하게 고정하면 앞부분 frame이 bucket 간에 동일해져
    bucket별 차이를 관찰하기 쉽다.
    """

    generator = np.random.default_rng(seed)
    feature = generator.standard_normal((1, frames, FEATURE_DIM), dtype=np.float32)
    return np.ascontiguousarray(feature, dtype=np.float32)


def all_tensor_names(model) -> list[str]:
    """graph의 모든 중간 Tensor 이름을 초기값(initializer)을 빼고 모은다."""

    initializers = {entry.name for entry in model.graph.initializer}
    names: list[str] = []
    seen: set[str] = set()
    for node in model.graph.node:
        for name in node.output:
            if name and name not in initializers and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def expose_all_outputs(model, names: list[str]):
    """모든 중간 Tensor를 graph 출력으로 승격한 사본을 돌려준다."""

    import onnx

    patched = onnx.ModelProto()
    patched.CopyFrom(model)
    existing = {entry.name for entry in patched.graph.output}
    for name in names:
        if name in existing:
            continue
        patched.graph.output.append(onnx.ValueInfoProto(name=name))
    return patched


def run_bucket(
    frames: int,
    static_path: Path,
    output_dir: Path,
    seed: int,
    intermediates: bool,
) -> dict:
    import onnx
    import onnxruntime as ort

    feature = build_feature(frames, seed)
    raw_path = output_dir / f"feature_{frames}.f32"
    raw_path.write_bytes(feature.tobytes(order="C"))

    model = onnx.load(str(static_path))
    names = all_tensor_names(model) if intermediates else []
    runnable = expose_all_outputs(model, names) if intermediates else model

    options = ort.SessionOptions()
    # 중간 Tensor를 그대로 보려면 graph 최적화를 꺼야 한다.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        runnable.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    requested = [entry.name for entry in session.get_outputs()]
    values = session.run(requested, {INPUT_NAME: feature})

    tensors = {name: np.asarray(value) for name, value in zip(requested, values)}
    npz_path = output_dir / f"ort_{frames}.npz"
    np.savez(npz_path, **tensors)

    embedding = tensors["embedding"]
    index = {
        "bucket_frames": frames,
        "static_model": str(static_path.relative_to(ROOT)),
        "seed": seed,
        "feature_path": raw_path.name,
        "feature_shape": list(feature.shape),
        "tensor_count": len(tensors),
        "embedding_shape": list(embedding.shape),
        "tensors": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in tensors.items()
        },
    }
    (output_dir / f"ort_{frames}.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )

    print(
        f"  {frames:>4} frames  tensors={len(tensors):>5}  "
        f"embedding={embedding.shape}  "
        f"|emb|={float(np.linalg.norm(embedding)):.6f}"
    )
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=ROOT / "results" / "static",
        help="campp_static_{frames}.onnx가 있는 폴더",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "runtime" / "ort_reference",
        help="ORT 기준 출력을 저장할 폴더",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--no-intermediates",
        dest="intermediates",
        action="store_false",
        default=True,
        help="최종 embedding만 저장한다",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        nargs="*",
        default=[frames for frames, _ in BUCKETS],
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("ORT 기준 출력 생성")
    summary = []
    for frames, _tag in BUCKETS:
        if frames not in args.buckets:
            continue
        static_path = args.static_dir / f"campp_static_{frames}.onnx"
        if not static_path.is_file():
            print(f"정적 ONNX가 없다: {static_path}", file=sys.stderr)
            return 1
        summary.append(
            run_bucket(
                frames, static_path, args.output_dir, args.seed, args.intermediates
            )
        )

    (args.output_dir / "index.json").write_text(
        json.dumps({"seed": args.seed, "buckets": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"저장 완료: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
