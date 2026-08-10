# Runtime bundle exporter의 구조 검증 테스트 파일이다.
#
# 확인할 내용:
# - 정적 그래프의 runtime node와 operator table 수 일치
# - 모든 input/output Tensor ID 존재
# - 네 bucket의 topology 동일성과 shape 차이
# - weights와 plan manifest checksum 일치
#
# 마지막 항목은 weight_blob_writer / bundle_manifest_writer가 생기면 추가한다.

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    import onnx  # noqa: F401

    ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover - onnx 없는 환경
    ONNX_AVAILABLE = False

from runtime_bundle_exporter.binary_format_schema import (  # noqa: E402
    OPERATOR_INPUT_CAPACITY,
    TENSOR_MAX_RANK,
    TensorStorageType,
)
from runtime_bundle_exporter.graph_ir_reader import (  # noqa: E402
    GraphIRMismatchError,
    read_graph_ir,
)

if ONNX_AVAILABLE:
    from runtime_bundle_exporter.static_model_reader import (  # noqa: E402
        build_runtime_graph,
        read_static_model,
    )

# (frames, IR 파일 접미사). 3초 bucket이 Phase 3의 기준 모델이다.
BUCKETS = ((98, "1s"), (298, "3s"), (498, "5s"), (998, "10s"))
REFERENCE_FRAMES = 298
REFERENCE_RUNTIME_NODES = 1438
REFERENCE_INITIALIZERS = 2280

STATIC_DIR = ROOT / "results" / "static"
GRAPH_DIR = ROOT / "results" / "graph"


def static_path(frames: int) -> Path:
    return STATIC_DIR / f"campp_static_{frames}.onnx"


def ir_path(tag: str) -> Path:
    return GRAPH_DIR / f"ir_{tag}.json"


def artifacts_present() -> bool:
    return all(
        static_path(frames).exists() and ir_path(tag).exists()
        for frames, tag in BUCKETS
    )


@unittest.skipUnless(ONNX_AVAILABLE, "onnx가 설치되어 있지 않다")
@unittest.skipUnless(artifacts_present(), "정적 ONNX 또는 graph IR 산출물이 없다")
class ReferenceBucketTests(unittest.TestCase):
    """3초 bucket 한 개를 RuntimeGraph까지 변환하는 경로를 검증한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.static = read_static_model(static_path(REFERENCE_FRAMES))
        cls.ir = read_graph_ir(ir_path("3s"))
        cls.graph = build_runtime_graph(cls.static, cls.ir)

    def test_runtime_node_count_matches_operator_table(self) -> None:
        self.assertEqual(len(self.static.nodes), REFERENCE_RUNTIME_NODES)
        self.assertEqual(len(self.ir.runtime_nodes), REFERENCE_RUNTIME_NODES)
        self.assertEqual(len(self.graph.operators), REFERENCE_RUNTIME_NODES)

    def test_static_nodes_stay_out_of_the_execution_plan(self) -> None:
        static_node_names = {node.name for node in self.ir.static_nodes}
        operator_names = {operator.name for operator in self.graph.operators}
        self.assertTrue(static_node_names)
        self.assertFalse(static_node_names & operator_names)

    def test_every_operator_tensor_id_exists(self) -> None:
        for operator in self.graph.operators:
            self.assertGreaterEqual(len(operator.input_tensor_ids), 1)
            self.assertLessEqual(
                len(operator.input_tensor_ids), OPERATOR_INPUT_CAPACITY
            )
            for tensor_id in (
                *operator.input_tensor_ids,
                *operator.output_tensor_ids,
            ):
                self.graph.tensor(tensor_id)  # 없으면 KeyError

    def test_tensor_ids_are_dense_and_typed(self) -> None:
        self.assertEqual(
            [tensor.tensor_id for tensor in self.graph.tensors],
            list(range(len(self.graph.tensors))),
        )
        for tensor in self.graph.tensors:
            self.assertLessEqual(len(tensor.shape), TENSOR_MAX_RANK)
            self.assertEqual(len(tensor.strides), len(tensor.shape))

    def test_graph_io_matches_the_ir(self) -> None:
        inputs = [self.graph.tensor(i).name for i in self.graph.input_tensor_ids]
        outputs = [self.graph.tensor(i).name for i in self.graph.output_tensor_ids]
        self.assertEqual(tuple(inputs), self.ir.graph_inputs)
        self.assertEqual(tuple(outputs), self.ir.graph_outputs)

    def test_every_initializer_becomes_a_constant_tensor(self) -> None:
        self.assertEqual(len(self.static.initializers), REFERENCE_INITIALIZERS)
        self.assertEqual(len(self.graph.initializers), REFERENCE_INITIALIZERS)
        constants = {
            tensor.tensor_id
            for tensor in self.graph.tensors
            if tensor.storage_type == TensorStorageType.CONSTANT
        }
        self.assertEqual(
            {item.tensor_id for item in self.graph.initializers}, constants
        )
        for item in self.graph.initializers:
            tensor = self.graph.tensor(item.tensor_id)
            self.assertEqual(item.byte_size, tensor.byte_size)
            self.assertIsNone(tensor.producer)

    def test_qlinear_conv_carries_explicit_attributes(self) -> None:
        node = next(n for n in self.static.nodes if n.op_type == "QLinearConv")
        for key in ("kernel_shape", "strides", "pads", "dilations", "group"):
            self.assertIn(key, node.attributes)

        binding = self.static.qlinear_conv_binding(node)
        weight_scales = self.static.scale_values(binding.w.scale)
        output_channels = self.static.tensor(binding.w.tensor).shape[0]
        self.assertIn(len(weight_scales), (1, output_channels))
        self.assertEqual(
            len(self.static.zero_point_values(binding.w.zero_point)),
            len(weight_scales),
        )
        self.assertEqual(len(self.static.scale_values(binding.y.scale)), 1)

    def test_bucket_frames_come_from_the_ir(self) -> None:
        self.assertEqual(self.graph.bucket_frames, REFERENCE_FRAMES)


@unittest.skipUnless(ONNX_AVAILABLE, "onnx가 설치되어 있지 않다")
@unittest.skipUnless(artifacts_present(), "정적 ONNX 또는 graph IR 산출물이 없다")
class BucketConsistencyTests(unittest.TestCase):
    """네 bucket이 같은 topology를 공유하고 shape만 달라지는지 확인한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graphs = {
            frames: build_runtime_graph(
                read_static_model(static_path(frames)), read_graph_ir(ir_path(tag))
            )
            for frames, tag in BUCKETS
        }

    def test_topology_is_identical_across_buckets(self) -> None:
        reference = self.graphs[REFERENCE_FRAMES]
        expected_operators = [
            (operator.name, operator.opcode, operator.input_tensor_ids)
            for operator in reference.operators
        ]
        expected_tensors = [tensor.name for tensor in reference.tensors]
        for frames, graph in self.graphs.items():
            with self.subTest(frames=frames):
                self.assertEqual(
                    [
                        (operator.name, operator.opcode, operator.input_tensor_ids)
                        for operator in graph.operators
                    ],
                    expected_operators,
                )
                self.assertEqual(
                    [tensor.name for tensor in graph.tensors], expected_tensors
                )

    def test_activation_shapes_track_the_bucket(self) -> None:
        name = self.graphs[REFERENCE_FRAMES].operators[0].name
        widths = {}
        for frames, graph in self.graphs.items():
            operator = next(o for o in graph.operators if o.name == name)
            widths[frames] = graph.tensor(operator.output_tensor_ids[0]).shape
        self.assertEqual(len(set(widths.values())), len(self.graphs))
        for frames, shape in widths.items():
            self.assertIn(frames, shape)

    def test_mismatched_bucket_stops_the_conversion(self) -> None:
        static = read_static_model(static_path(REFERENCE_FRAMES))
        for frames, tag in BUCKETS:
            if frames == REFERENCE_FRAMES:
                continue
            with self.subTest(tag=tag):
                with self.assertRaises(GraphIRMismatchError):
                    build_runtime_graph(static, read_graph_ir(ir_path(tag)))


if __name__ == "__main__":
    unittest.main()
