#!/usr/bin/env python3
"""ORT 입력으로 C Kernel을 하나씩 독립 재생한다.

Phase 14의 ``ort_{frames}.npz``와 ``feature_{frames}.f32``를 Tensor ID 기반
reference store로 변환한다. C replay runner는 Operator를 실행하기 직전에 이
store에서 입력 Tensor를 복원하므로 앞 Operator의 C 오차가 다음 Operator의
판정에 전파되지 않는다.

기본 실행:

    python tests/runtime/operator_replay_tests/test_operator_replay.py

순서:

1. ORT Tensor를 ``reference_{frames}.rpl``로 pack
2. ``campp_operator_replay``로 1,438개 Kernel을 독립 실행
3. ``05_compare_runtime_outputs.py``로 ORT 출력과 replay 출력 비교
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PYTHON_SOURCE = ROOT / "src" / "python"

BUCKETS: tuple[tuple[int, str], ...] = (
    (98, "1s"),
    (298, "3s"),
    (498, "5s"),
    (998, "10s"),
)

REPLAY_MAGIC = b"CMPPRPL1"
REPLAY_VERSION = 1
REPLAY_HEADER = struct.Struct("<8sIIIIIIQQQQ")
REPLAY_ENTRY = struct.Struct("<IBBH4IQQQ")
REPLAY_ENTRY_PRESENT = 1
REPLAY_ALIGNMENT = 8
MISSING_PAYLOAD_OFFSET = (1 << 64) - 1

TENSOR_STORAGE_INPUT = 1
TENSOR_STORAGE_CONSTANT = 3

NP_DTYPE_BY_CODE: dict[int, np.dtype] = {
    1: np.dtype(np.float32),
    2: np.dtype(np.uint8),
    3: np.dtype(np.int8),
    4: np.dtype(np.int32),
    5: np.dtype(np.int64),
    6: np.dtype(np.bool_),
    7: np.dtype(np.float16),
}


class OperatorReplayError(RuntimeError):
    """Replay artifact 또는 실행 결과가 유효하지 않을 때 발생한다."""


def align_up(value: int, alignment: int = REPLAY_ALIGNMENT) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def load_runtime_graph(frames: int, tag: str):
    if str(PYTHON_SOURCE) not in sys.path:
        sys.path.insert(0, str(PYTHON_SOURCE))
    from runtime_bundle_exporter.reader.graph_ir_reader import read_graph_ir
    from runtime_bundle_exporter.reader.static_model_reader import (
        build_runtime_graph,
        read_static_model,
    )

    static = read_static_model(
        ROOT / "results" / "static" / f"campp_static_{frames}.onnx"
    )
    graph_ir = read_graph_ir(ROOT / "results" / "graph" / f"ir_{tag}.json")
    return build_runtime_graph(
        static, graph_ir, validate=True, name=f"campp_{frames}f"
    )


def _padded_dimensions(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(shape) > 4:
        raise OperatorReplayError(f"rank {len(shape)} exceeds replay capacity")
    return tuple(shape) + (1,) * (4 - len(shape))  # type: ignore[return-value]


def _validate_array(tensor, array: np.ndarray) -> np.ndarray:
    dtype_code = int(tensor.dtype)
    expected_dtype = NP_DTYPE_BY_CODE.get(dtype_code)
    if expected_dtype is None:
        raise OperatorReplayError(
            f"Tensor {tensor.tensor_id} uses unsupported dtype code {dtype_code}"
        )
    value = np.asarray(array)
    if value.dtype != expected_dtype:
        raise OperatorReplayError(
            f"Tensor {tensor.tensor_id} dtype mismatch: "
            f"ORT={value.dtype}, graph={expected_dtype}"
        )
    if tuple(value.shape) != tuple(tensor.shape):
        raise OperatorReplayError(
            f"Tensor {tensor.tensor_id} shape mismatch: "
            f"ORT={tuple(value.shape)}, graph={tuple(tensor.shape)}"
        )
    if value.nbytes != tensor.byte_size:
        raise OperatorReplayError(
            f"Tensor {tensor.tensor_id} byte size mismatch: "
            f"ORT={value.nbytes}, graph={tensor.byte_size}"
        )
    return np.ascontiguousarray(value)


def _feature_array(graph, feature_path: Path) -> dict[int, np.ndarray]:
    values: dict[int, np.ndarray] = {}
    input_ids = tuple(graph.input_tensor_ids)
    if len(input_ids) != 1:
        raise OperatorReplayError(
            f"operator replay currently expects one graph input, got {len(input_ids)}"
        )
    tensor = graph.tensors[input_ids[0]]
    dtype = NP_DTYPE_BY_CODE.get(int(tensor.dtype))
    if dtype is None:
        raise OperatorReplayError(f"unsupported graph input dtype {tensor.dtype}")
    raw = np.fromfile(feature_path, dtype=dtype)
    expected_elements = int(np.prod(tensor.shape, dtype=np.int64))
    if raw.size != expected_elements:
        raise OperatorReplayError(
            f"feature element count mismatch: file={raw.size}, "
            f"graph={expected_elements}"
        )
    values[tensor.tensor_id] = raw.reshape(tensor.shape)
    return values


def write_reference_store(
    graph,
    ort_npz_path: Path,
    feature_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """ORT Tensor를 C가 스트리밍할 수 있는 fixed-index store로 기록한다."""

    tensors = tuple(graph.tensors)
    if any(tensor.tensor_id != index for index, tensor in enumerate(tensors)):
        raise OperatorReplayError("RuntimeGraph Tensor IDs must be contiguous")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    feature_values = _feature_array(graph, feature_path)
    entries: list[tuple[int, int, int, int, tuple[int, int, int, int], int, int]] = []
    present_count = 0

    try:
        with np.load(ort_npz_path) as ort_tensors, temporary.open("w+b") as stream:
            table_offset = REPLAY_HEADER.size
            data_offset = align_up(
                table_offset + len(tensors) * REPLAY_ENTRY.size
            )
            stream.seek(data_offset)

            for tensor in tensors:
                storage_type = int(tensor.storage_type)
                array: np.ndarray | None
                if storage_type == TENSOR_STORAGE_CONSTANT:
                    array = None
                elif tensor.tensor_id in feature_values:
                    array = feature_values[tensor.tensor_id]
                elif tensor.name in ort_tensors.files:
                    array = np.asarray(ort_tensors[tensor.name])
                else:
                    raise OperatorReplayError(
                        f"ORT reference is missing non-constant Tensor "
                        f"{tensor.tensor_id} ({tensor.name})"
                    )

                dimensions = _padded_dimensions(tuple(tensor.shape))
                if array is None:
                    entries.append(
                        (
                            tensor.tensor_id,
                            int(tensor.dtype),
                            len(tensor.shape),
                            0,
                            dimensions,
                            MISSING_PAYLOAD_OFFSET,
                            0,
                        )
                    )
                    continue

                dense = _validate_array(tensor, array)
                payload_offset = stream.tell()
                payload = dense.tobytes(order="C")
                stream.write(payload)
                padding = align_up(stream.tell()) - stream.tell()
                if padding:
                    stream.write(b"\x00" * padding)
                entries.append(
                    (
                        tensor.tensor_id,
                        int(tensor.dtype),
                        len(tensor.shape),
                        REPLAY_ENTRY_PRESENT,
                        dimensions,
                        payload_offset,
                        len(payload),
                    )
                )
                present_count += 1

            file_size = stream.tell()
            stream.seek(0)
            stream.write(
                REPLAY_HEADER.pack(
                    REPLAY_MAGIC,
                    REPLAY_VERSION,
                    REPLAY_HEADER.size,
                    int(graph.bucket_frames),
                    len(tensors),
                    REPLAY_ENTRY.size,
                    0,
                    table_offset,
                    data_offset,
                    file_size,
                    0,
                )
            )
            for (
                tensor_id,
                dtype,
                rank,
                flags,
                dimensions,
                payload_offset,
                byte_size,
            ) in entries:
                stream.write(
                    REPLAY_ENTRY.pack(
                        tensor_id,
                        dtype,
                        rank,
                        flags,
                        *dimensions,
                        payload_offset,
                        byte_size,
                        0,
                    )
                )
            if stream.tell() > data_offset:
                raise OperatorReplayError("replay index overlaps payload")
            stream.write(b"\x00" * (data_offset - stream.tell()))
            stream.flush()
            os.fsync(stream.fileno())

        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    digest = hashlib.sha256()
    with output_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    metadata = {
        "format": "campp_operator_replay_store",
        "version": REPLAY_VERSION,
        "bucket_frames": int(graph.bucket_frames),
        "tensor_count": len(tensors),
        "present_tensor_count": present_count,
        "byte_size": output_path.stat().st_size,
        "sha256": digest.hexdigest(),
        "ort_reference": ort_npz_path.name,
        "feature": feature_path.name,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def _runner_default() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return ROOT / "build" / f"campp_operator_replay{suffix}"


def _store_is_stale(store: Path, sources: tuple[Path, ...]) -> bool:
    if not store.is_file():
        return True
    store_time = store.stat().st_mtime_ns
    return any(not source.is_file() or source.stat().st_mtime_ns > store_time for source in sources)


def run_bucket_replay(
    frames: int,
    tag: str,
    *,
    ort_dir: Path,
    bundle_dir: Path,
    replay_dir: Path,
    runner: Path,
    rebuild_store: bool,
) -> None:
    graph = load_runtime_graph(frames, tag)
    ort_npz = ort_dir / f"ort_{frames}.npz"
    feature = ort_dir / f"feature_{frames}.f32"
    store = replay_dir / f"reference_{frames}.rpl"
    plan = bundle_dir / "execution_plans" / f"plan_{frames}.bin"
    weights = bundle_dir / "weights.bin"
    sources = (
        ort_npz,
        feature,
        ROOT / "results" / "static" / f"campp_static_{frames}.onnx",
        ROOT / "results" / "graph" / f"ir_{tag}.json",
    )
    missing = [path for path in (*sources, plan, weights, runner) if not path.is_file()]
    if missing:
        raise OperatorReplayError(
            "required replay files are missing:\n  "
            + "\n  ".join(str(path) for path in missing)
        )

    if rebuild_store or _store_is_stale(store, sources):
        print(f"[{frames} frames] ORT input store 생성")
        metadata = write_reference_store(graph, ort_npz, feature, store)
        print(
            f"  tensors={metadata['present_tensor_count']} "
            f"bytes={metadata['byte_size']} sha256={str(metadata['sha256'])[:12]}…"
        )
    else:
        print(f"[{frames} frames] 기존 ORT input store 재사용: {store.name}")

    output_prefix = replay_dir / f"c_{frames}"
    command = [
        str(runner),
        str(plan),
        str(weights),
        str(store),
        str(output_prefix),
    ]
    print(f"[{frames} frames] 1,438개 Operator 독립 replay")
    subprocess.run(command, check=True)


def run_comparison(
    *, ort_dir: Path, replay_dir: Path, results_dir: Path, buckets: list[int]
) -> int:
    comparator = ROOT / "scripts" / "3_runtime" / "05_compare_runtime_outputs.py"
    command = [
        sys.executable,
        str(comparator),
        "--ort-dir",
        str(ort_dir),
        "--c-dir",
        str(replay_dir),
        "--results-dir",
        str(results_dir),
        "--buckets",
        *(str(frames) for frames in buckets),
    ]
    return subprocess.run(command, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ort-dir",
        type=Path,
        default=ROOT / "runs" / "runtime" / "ort_reference",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=ROOT / "models" / "compiled" / "reference",
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=ROOT / "runs" / "runtime" / "operator_replay",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "runtime" / "operator_replay",
    )
    parser.add_argument("--runner", type=Path, default=_runner_default())
    parser.add_argument(
        "--buckets", type=int, nargs="*", default=[frames for frames, _ in BUCKETS]
    )
    parser.add_argument(
        "--rebuild-store",
        action="store_true",
        help="cached reference_*.rpl을 무시하고 다시 생성",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="C replay 출력만 만들고 수치 비교는 생략",
    )
    args = parser.parse_args(argv)

    known = {frames for frames, _ in BUCKETS}
    unknown = sorted(set(args.buckets) - known)
    if unknown:
        parser.error(f"unsupported buckets: {unknown}")
    args.replay_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    try:
        for frames, tag in BUCKETS:
            if frames in args.buckets:
                run_bucket_replay(
                    frames,
                    tag,
                    ort_dir=args.ort_dir,
                    bundle_dir=args.bundle_dir,
                    replay_dir=args.replay_dir,
                    runner=args.runner,
                    rebuild_store=args.rebuild_store,
                )
    except (OperatorReplayError, subprocess.CalledProcessError) as exc:
        print(f"operator replay failed: {exc}", file=sys.stderr)
        return 1

    if args.skip_compare:
        return 0
    return run_comparison(
        ort_dir=args.ort_dir,
        replay_dir=args.replay_dir,
        results_dir=args.results_dir,
        buckets=args.buckets,
    )


class ReferenceStoreFormatTests(unittest.TestCase):
    def test_store_contains_dynamic_tensors_and_omits_constants(self) -> None:
        tensors = (
            SimpleNamespace(
                tensor_id=0,
                name="feature",
                dtype=1,
                shape=(1, 2),
                byte_size=8,
                storage_type=TENSOR_STORAGE_INPUT,
            ),
            SimpleNamespace(
                tensor_id=1,
                name="weight",
                dtype=1,
                shape=(2,),
                byte_size=8,
                storage_type=TENSOR_STORAGE_CONSTANT,
            ),
            SimpleNamespace(
                tensor_id=2,
                name="output",
                dtype=1,
                shape=(1, 2),
                byte_size=8,
                storage_type=4,
            ),
        )
        graph = SimpleNamespace(
            tensors=tensors, input_tensor_ids=(0,), bucket_frames=2
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.asarray([[1.0, 2.0]], dtype=np.float32).tofile(root / "feature.f32")
            np.savez(root / "ort.npz", output=np.asarray([[3.0, 4.0]], dtype=np.float32))
            metadata = write_reference_store(
                graph, root / "ort.npz", root / "feature.f32", root / "store.rpl"
            )
            raw = (root / "store.rpl").read_bytes()

        header = REPLAY_HEADER.unpack_from(raw)
        self.assertEqual(header[0], REPLAY_MAGIC)
        self.assertEqual(header[1], REPLAY_VERSION)
        self.assertEqual(header[3], 2)
        self.assertEqual(header[4], 3)
        self.assertEqual(metadata["present_tensor_count"], 2)

        entries = [
            REPLAY_ENTRY.unpack_from(raw, REPLAY_HEADER.size + i * REPLAY_ENTRY.size)
            for i in range(3)
        ]
        self.assertEqual(entries[0][3], REPLAY_ENTRY_PRESENT)
        self.assertEqual(entries[1][3], 0)
        self.assertEqual(entries[1][8], MISSING_PAYLOAD_OFFSET)
        self.assertEqual(entries[2][3], REPLAY_ENTRY_PRESENT)
        output_offset = entries[2][8]
        output_size = entries[2][9]
        np.testing.assert_array_equal(
            np.frombuffer(raw[output_offset : output_offset + output_size], dtype=np.float32),
            np.asarray([3.0, 4.0], dtype=np.float32),
        )

    def test_store_rejects_dtype_mismatch(self) -> None:
        tensors = (
            SimpleNamespace(
                tensor_id=0,
                name="feature",
                dtype=1,
                shape=(1,),
                byte_size=4,
                storage_type=TENSOR_STORAGE_INPUT,
            ),
            SimpleNamespace(
                tensor_id=1,
                name="output",
                dtype=1,
                shape=(1,),
                byte_size=4,
                storage_type=4,
            ),
        )
        graph = SimpleNamespace(
            tensors=tensors, input_tensor_ids=(0,), bucket_frames=1
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.asarray([1.0], dtype=np.float32).tofile(root / "feature.f32")
            np.savez(root / "ort.npz", output=np.asarray([1], dtype=np.int32))
            with self.assertRaises(OperatorReplayError):
                write_reference_store(
                    graph,
                    root / "ort.npz",
                    root / "feature.f32",
                    root / "store.rpl",
                )


def _load_comparator_module():
    path = ROOT / "scripts" / "3_runtime" / "05_compare_runtime_outputs.py"
    spec = importlib.util.spec_from_file_location("runtime_output_comparator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load comparator module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ComparisonMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.comparator = _load_comparator_module()

    def test_dtype_is_checked_before_values(self) -> None:
        result = self.comparator.compare_arrays(
            np.asarray([1], dtype=np.int32),
            np.asarray([1.0], dtype=np.float32),
            1,
        )
        self.assertEqual(result["status"], "dtype_mismatch")

    def test_float_metrics_include_mean_absolute_error(self) -> None:
        result = self.comparator.compare_arrays(
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([2.0, 2.0], dtype=np.float32),
            1,
            float_atol=0.0,
            float_rtol=0.0,
        )
        self.assertEqual(result["status"], "float_mismatch")
        self.assertEqual(result["max_abs_error"], 1.0)
        self.assertEqual(result["mean_abs_error"], 0.5)
        self.assertEqual(result["max_rel_error"], 1.0)
        self.assertIn("cosine_similarity", result)

    def test_integer_comparison_requires_exact_values(self) -> None:
        result = self.comparator.compare_arrays(
            np.asarray([3, 4], dtype=np.uint8),
            np.asarray([3, 5], dtype=np.uint8),
            2,
        )
        self.assertEqual(result["status"], "integer_mismatch")
        self.assertEqual(result["mismatched_elements"], 1)
        self.assertFalse(result["bitwise_identical"])


if __name__ == "__main__":
    if "unittest" in sys.argv:
        sys.argv.remove("unittest")
        unittest.main()
    else:
        raise SystemExit(main())
