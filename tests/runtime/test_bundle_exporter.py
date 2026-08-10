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

import dataclasses
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    import onnx  # noqa: F401

    ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover - onnx 없는 환경
    ONNX_AVAILABLE = False

from runtime_bundle_exporter.binary_format_schema import (  # noqa: E402
    INVALID_DATA_OFFSET,
    INVALID_TENSOR_ID,
    OPERATOR_DESCRIPTOR_SIZE,
    OPERATOR_INPUT_CAPACITY,
    PLAN_FORMAT_VERSION,
    PLAN_HEADER_SIZE,
    PLAN_MAGIC,
    PLAN_SECTION_ALIGNMENT,
    TENSOR_DESCRIPTOR_SIZE,
    TENSOR_MAX_RANK,
    OperatorCode,
    TensorDType,
    TensorFlags,
    TensorStorageType,
)
from runtime_bundle_exporter.bundle_manifest_writer import (  # noqa: E402
    MANIFEST_FILE_NAME,
    BundleManifestError,
    verify_bundle_manifest,
    write_bundle_manifest,
)
from runtime_bundle_exporter.execution_plan_writer import (  # noqa: E402
    EXECUTION_PLAN_DIR_NAME,
    ExecutionPlanError,
    plan_file_name,
    read_execution_plan,
    verify_plan,
    write_execution_plan,
)
from runtime_bundle_exporter.weight_blob_writer import (  # noqa: E402
    WEIGHT_BLOB_FILE_NAME,
    WeightBlobError,
    build_weight_blob,
    read_weight_blob,
    verify_weight_blob,
    write_weight_blob,
)
from runtime_bundle_exporter.graph_ir_reader import DTYPE_BYTE_SIZE  # noqa: E402
from runtime_bundle_exporter.operator_table_builder import (  # noqa: E402
    OperatorTableError,
    build_operator_table,
    validate_execution_order,
)
from runtime_bundle_exporter.tensor_table_builder import (  # noqa: E402
    WEIGHT_BLOB_ALIGNMENT,
    build_tensor_table,
    plan_weight_blob,
)
from runtime_bundle_exporter.graph_ir_reader import (  # noqa: E402
    GraphIRMismatchError,
    read_graph_ir,
)
from runtime_bundle_exporter.runtime_ir import (  # noqa: E402
    InitializerScope,
    RuntimeBundle,
    RuntimeIRError,
    RuntimeOperator,
    RuntimeTensor,
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
# 원본 ONNX가 들고 있던 학습 parameter와, export_static이 shape 도메인에서 접은 상수.
REFERENCE_SHARED_INITIALIZERS = 1963
REFERENCE_BUCKET_LOCAL_INITIALIZERS = 317

STATIC_DIR = ROOT / "results" / "static"
GRAPH_DIR = ROOT / "results" / "graph"
MODEL_FILE_NAME = "campplus_int8_static_qop.onnx"


def static_path(frames: int) -> Path:
    return STATIC_DIR / f"campp_static_{frames}.onnx"


def ir_path(tag: str) -> Path:
    return GRAPH_DIR / f"ir_{tag}.json"


def artifacts_present() -> bool:
    return all(
        static_path(frames).exists() and ir_path(tag).exists()
        for frames, tag in BUCKETS
    )


_BUNDLE_CACHE: list = []


def reference_bundle():
    """네 bucket을 한 번만 변환해 재사용한다 (모델 하나가 8MB라 로딩이 느리다)."""

    if not _BUNDLE_CACHE:
        _BUNDLE_CACHE.append(
            RuntimeBundle(
                graphs=tuple(
                    build_runtime_graph(
                        read_static_model(static_path(frames)),
                        read_graph_ir(ir_path(tag)),
                    )
                    for frames, tag in BUCKETS
                )
            )
        )
    return _BUNDLE_CACHE[0]


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

    def test_initializer_scope_follows_the_ir_origin(self) -> None:
        shared = [
            item
            for item in self.graph.initializers
            if item.scope is InitializerScope.SHARED
        ]
        bucket_local = [
            item
            for item in self.graph.initializers
            if item.scope is InitializerScope.BUCKET_LOCAL
        ]
        self.assertEqual(len(shared), REFERENCE_SHARED_INITIALIZERS)
        self.assertEqual(len(bucket_local), REFERENCE_BUCKET_LOCAL_INITIALIZERS)

        # SHARED는 원본 ONNX의 initializer, BUCKET_LOCAL은 접힌 static node 출력이다.
        for item in shared:
            tensor = self.ir.tensor(item.name)
            self.assertTrue(tensor.is_initializer, item.name)
            self.assertIsNone(tensor.producer, item.name)
        for item in bucket_local:
            tensor = self.ir.tensor(item.name)
            self.assertFalse(tensor.is_initializer, item.name)
            self.assertIsNotNone(tensor.producer, item.name)
            self.assertTrue(self.ir.node(tensor.producer).is_static, item.name)


@unittest.skipUnless(ONNX_AVAILABLE, "onnx가 설치되어 있지 않다")
@unittest.skipUnless(artifacts_present(), "정적 ONNX 또는 graph IR 산출물이 없다")
class BucketConsistencyTests(unittest.TestCase):
    """네 bucket이 같은 topology를 공유하고 shape만 달라지는지 확인한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graphs = {
            graph.bucket_frames: graph for graph in reference_bundle().graphs
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

    def test_every_diverging_initializer_is_bucket_local(self) -> None:
        """bucket마다 값이 달라지는 상수가 하나도 SHARED로 새지 않아야 한다."""

        reference = {
            item.name: item
            for item in self.graphs[REFERENCE_FRAMES].initializers
        }
        diverging = {
            item.name
            for graph in self.graphs.values()
            for item in graph.initializers
            if (item.raw_data, item.shape)
            != (reference[item.name].raw_data, reference[item.name].shape)
        }
        self.assertTrue(diverging)
        for name in diverging:
            self.assertIs(
                reference[name].scope, InitializerScope.BUCKET_LOCAL, name
            )


@unittest.skipUnless(ONNX_AVAILABLE, "onnx가 설치되어 있지 않다")
@unittest.skipUnless(artifacts_present(), "정적 ONNX 또는 graph IR 산출물이 없다")
class BundleScopeTests(unittest.TestCase):
    """네 bucket을 한 weight blob으로 묶는 규칙을 검증한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graphs = reference_bundle().graphs

    def replace_initializer(self, position: int, **changes: object):
        """graph 하나의 initializer 한 개만 바꾼 사본을 만든다."""

        graph = self.graphs[position]
        initializers = list(graph.initializers)
        index, original = next(
            (index, item)
            for index, item in enumerate(initializers)
            if item.scope is changes.pop("scope_filter")
        )
        initializers[index] = dataclasses.replace(original, **changes)
        return dataclasses.replace(graph, initializers=tuple(initializers))

    def test_bundle_accepts_the_four_buckets(self) -> None:
        bundle = RuntimeBundle(graphs=self.graphs)
        shared = bundle.shared_initializers
        self.assertEqual(len(shared), REFERENCE_SHARED_INITIALIZERS)
        self.assertTrue(
            all(item.scope is InitializerScope.SHARED for item in shared)
        )
        for frames, _ in BUCKETS:
            with self.subTest(frames=frames):
                local = bundle.bucket_initializers(frames)
                self.assertEqual(
                    len(local), REFERENCE_BUCKET_LOCAL_INITIALIZERS
                )
                self.assertTrue(
                    all(
                        item.scope is InitializerScope.BUCKET_LOCAL
                        for item in local
                    )
                )

    def test_bucket_local_values_are_preserved_per_graph(self) -> None:
        bundle = RuntimeBundle(graphs=self.graphs)
        by_name = [
            {item.name: item.raw_data for item in bundle.bucket_initializers(frames)}
            for frames, _ in BUCKETS
        ]
        changed = {
            name
            for name, data in by_name[0].items()
            if any(other[name] != data for other in by_name[1:])
        }
        self.assertTrue(changed, "bucket별 값 차이가 보존되지 않았다")

    def test_bundle_rejects_a_shared_initializer_that_diverges(self) -> None:
        original = self.graphs[1].initializers
        target = next(
            item for item in original if item.scope is InitializerScope.SHARED
        )
        tampered = bytes([target.raw_data[0] ^ 0xFF]) + target.raw_data[1:]
        graph = self.replace_initializer(
            1, scope_filter=InitializerScope.SHARED, raw_data=tampered
        )
        graphs = (self.graphs[0], graph, *self.graphs[2:])
        with self.assertRaises(RuntimeIRError) as caught:
            RuntimeBundle(graphs=graphs)
        self.assertIn("SHARED", str(caught.exception))

    def test_bundle_rejects_scope_disagreement(self) -> None:
        graph = self.replace_initializer(
            1,
            scope_filter=InitializerScope.SHARED,
            scope=InitializerScope.BUCKET_LOCAL,
        )
        graphs = (self.graphs[0], graph, *self.graphs[2:])
        with self.assertRaises(RuntimeIRError) as caught:
            RuntimeBundle(graphs=graphs)
        message = str(caught.exception)
        self.assertIn(InitializerScope.SHARED.value, message)
        self.assertIn(InitializerScope.BUCKET_LOCAL.value, message)


@unittest.skipUnless(ONNX_AVAILABLE, "onnx가 설치되어 있지 않다")
@unittest.skipUnless(artifacts_present(), "정적 ONNX 또는 graph IR 산출물이 없다")
class TensorTableTests(unittest.TestCase):
    """Tensor 이름을 정수 ID로 바꾼 표가 C에서 그대로 쓸 수 있는지 확인한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = RuntimeBundle(
            graphs=tuple(
                build_runtime_graph(
                    read_static_model(static_path(frames)),
                    read_graph_ir(ir_path(tag)),
                )
                for frames, tag in BUCKETS
            )
        )
        cls.layout = plan_weight_blob(cls.bundle)
        cls.graph = cls.bundle.graph_for_bucket(REFERENCE_FRAMES)
        cls.table = build_tensor_table(cls.graph, cls.layout)

    def test_every_tensor_id_is_unique_and_dense(self) -> None:
        ids = [descriptor.tensor_id for descriptor in self.table.descriptors]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(ids, list(range(len(self.table))))
        self.assertEqual(len(self.table.ids), len(self.table))

    def test_names_resolve_to_ids_both_ways(self) -> None:
        for tensor_id, name in enumerate(self.table.names):
            self.assertEqual(self.table.id_of(name), tensor_id)
            self.assertEqual(self.table.name_of(tensor_id), name)

    def test_byte_size_matches_shape_times_dtype(self) -> None:
        for descriptor in self.table.descriptors:
            with self.subTest(tensor_id=descriptor.tensor_id):
                elements = 1
                for axis in range(descriptor.rank):
                    elements *= descriptor.dimensions[axis]
                expected = elements * DTYPE_BYTE_SIZE[TensorDType(descriptor.dtype)]
                self.assertEqual(descriptor.logical_byte_size, expected)
                self.assertEqual(descriptor.storage_span_bytes, expected)

    def test_unused_dimensions_and_strides_are_normalized(self) -> None:
        for descriptor in self.table.descriptors:
            for axis in range(descriptor.rank, TENSOR_MAX_RANK):
                self.assertEqual(descriptor.dimensions[axis], 1)
                self.assertEqual(descriptor.byte_strides[axis], 0)

    def test_feature_and_embedding_are_pinned(self) -> None:
        feature = self.table.descriptors[self.table.id_of("feature")]
        embedding = self.table.descriptors[self.table.id_of("embedding")]
        self.assertEqual(feature.storage_type, TensorStorageType.INPUT)
        self.assertEqual(embedding.storage_type, TensorStorageType.OUTPUT)
        self.assertEqual(feature.dimensions[:3], (1, REFERENCE_FRAMES, 80))
        self.assertEqual(embedding.dimensions[:2], (1, 192))
        # 입력은 외부 pointer를 연결하므로 offset이 없다.
        self.assertEqual(feature.data_offset, INVALID_DATA_OFFSET)

    def test_only_constants_carry_a_data_offset(self) -> None:
        for descriptor in self.table.descriptors:
            with self.subTest(tensor_id=descriptor.tensor_id):
                if descriptor.storage_type == TensorStorageType.CONSTANT:
                    self.assertNotEqual(descriptor.data_offset, INVALID_DATA_OFFSET)
                    self.assertTrue(descriptor.flags & TensorFlags.READ_ONLY)
                    self.assertTrue(descriptor.flags & TensorFlags.EXTERNAL)
                else:
                    # Phase 3는 Arena를 아직 계획하지 않는다.
                    self.assertEqual(descriptor.data_offset, INVALID_DATA_OFFSET)

    def test_lifetimes_are_ordered(self) -> None:
        for descriptor in self.table.descriptors:
            if descriptor.storage_type == TensorStorageType.CONSTANT:
                continue
            with self.subTest(tensor_id=descriptor.tensor_id):
                self.assertLessEqual(descriptor.first_use, descriptor.last_use)

    def test_packed_table_has_the_fixed_stride(self) -> None:
        packed = self.table.pack()
        self.assertEqual(len(packed), len(self.table) * TENSOR_DESCRIPTOR_SIZE)
        self.assertEqual(len(packed), self.table.byte_size)

    def test_weight_blob_places_every_constant(self) -> None:
        for frames, _ in BUCKETS:
            graph = self.bundle.graph_for_bucket(frames)
            with self.subTest(frames=frames):
                for item in graph.initializers:
                    offset = self.layout.offset_of(frames, item.tensor_id)
                    self.assertEqual(offset % WEIGHT_BLOB_ALIGNMENT, 0)
                    self.assertLessEqual(
                        offset + item.byte_size, self.layout.total_bytes
                    )

    def test_shared_weights_keep_one_offset_across_buckets(self) -> None:
        graphs = {frames: self.bundle.graph_for_bucket(frames) for frames, _ in BUCKETS}
        shared = self.bundle.shared_initializers
        for item in shared[:200]:
            offsets = {
                self.layout.offset_of(frames, graph.ids_by_name[item.name])
                if hasattr(graph, "ids_by_name")
                else self.layout.offset_of(
                    frames,
                    next(
                        other.tensor_id
                        for other in graph.initializers
                        if other.name == item.name
                    ),
                )
                for frames, graph in graphs.items()
            }
            self.assertEqual(len(offsets), 1, item.name)

    def test_bucket_local_constants_get_their_own_offsets(self) -> None:
        name = self.bundle.bucket_initializers(REFERENCE_FRAMES)[0].name
        offsets = set()
        for frames, _ in BUCKETS:
            graph = self.bundle.graph_for_bucket(frames)
            tensor_id = next(
                item.tensor_id for item in graph.initializers if item.name == name
            )
            offsets.add(self.layout.offset_of(frames, tensor_id))
        self.assertEqual(len(offsets), len(BUCKETS))


@unittest.skipUnless(ONNX_AVAILABLE, "onnx가 설치되어 있지 않다")
@unittest.skipUnless(artifacts_present(), "정적 ONNX 또는 graph IR 산출물이 없다")
class OperatorTableTests(unittest.TestCase):
    """1,438개 node가 실행 가능한 명령 순서가 되었는지 확인한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = build_runtime_graph(
            read_static_model(static_path(REFERENCE_FRAMES)),
            read_graph_ir(ir_path("3s")),
        )
        cls.table = build_operator_table(cls.graph)

    def test_one_command_per_runtime_node(self) -> None:
        self.assertEqual(len(self.table), REFERENCE_RUNTIME_NODES)
        self.assertEqual(
            [descriptor.operator_id for descriptor in self.table.descriptors],
            list(range(REFERENCE_RUNTIME_NODES)),
        )

    def test_every_opcode_is_one_of_the_twenty_kernels(self) -> None:
        opcodes = {OperatorCode(d.opcode) for d in self.table.descriptors}
        self.assertEqual(len(opcodes), 20)
        self.assertNotIn(OperatorCode.INVALID, opcodes)

    def test_inputs_are_only_read_after_they_are_produced(self) -> None:
        """완료 조건: 표를 위에서 아래로 읽어도 실행이 성립한다."""

        ready = {
            tensor.tensor_id
            for tensor in self.graph.tensors
            if tensor.storage_type
            in (TensorStorageType.INPUT, TensorStorageType.CONSTANT)
        }
        for descriptor in self.table.descriptors:
            used = descriptor.input_tensor_ids[: descriptor.input_count]
            for tensor_id in used:
                self.assertIn(tensor_id, ready, descriptor.operator_id)
            ready.update(descriptor.output_tensor_ids)

    def test_unused_input_slots_are_invalid(self) -> None:
        for descriptor in self.table.descriptors:
            with self.subTest(operator_id=descriptor.operator_id):
                used = descriptor.input_tensor_ids[: descriptor.input_count]
                unused = descriptor.input_tensor_ids[descriptor.input_count :]
                self.assertNotIn(INVALID_TENSOR_ID, used)
                self.assertTrue(all(item == INVALID_TENSOR_ID for item in unused))

    def test_attributes_round_trip_through_the_section(self) -> None:
        by_name = {name: index for index, name in enumerate(self.table.names)}
        transpose = self.table.attributes_of(by_name["/Transpose"])
        self.assertEqual(transpose["perm"], (0, 2, 1))

        conv_id = next(
            d.operator_id
            for d in self.table.descriptors
            if d.opcode == OperatorCode.QLINEAR_CONV
        )
        conv = self.table.attributes_of(conv_id)
        self.assertEqual(conv["kernel_shape"], (3, 3))
        self.assertEqual(len(conv["pads"]), 4)
        self.assertIn("group", conv)

        relu_id = next(
            d.operator_id
            for d in self.table.descriptors
            if d.opcode == OperatorCode.RELU
        )
        self.assertEqual(self.table.attributes_of(relu_id), {})
        self.assertEqual(self.table.descriptors[relu_id].attribute_offset, 0)
        self.assertEqual(self.table.descriptors[relu_id].attribute_size, 0)

    def test_identical_attributes_share_one_block(self) -> None:
        with_attributes = [
            d for d in self.table.descriptors if d.attribute_size > 0
        ]
        offsets = {d.attribute_offset for d in with_attributes}
        self.assertLess(len(offsets), len(with_attributes))
        self.assertEqual(len(offsets), self.table.attribute_blocks)
        for descriptor in with_attributes:
            self.assertEqual(
                descriptor.attribute_offset % PLAN_SECTION_ALIGNMENT, 0
            )

    def test_packed_table_has_the_fixed_stride(self) -> None:
        packed = self.table.pack()
        self.assertEqual(len(packed), len(self.table) * OPERATOR_DESCRIPTOR_SIZE)

    def test_out_of_order_operators_are_rejected(self) -> None:
        operators = list(self.graph.operators)
        swapped = dataclasses.replace(
            self.graph, operators=(operators[1], operators[0], *operators[2:])
        )
        with self.assertRaises(OperatorTableError):
            validate_execution_order(swapped)


@unittest.skipUnless(ONNX_AVAILABLE, "onnx가 설치되어 있지 않다")
@unittest.skipUnless(artifacts_present(), "정적 ONNX 또는 graph IR 산출물이 없다")
class TensorTableTests(unittest.TestCase):
    """Tensor 이름을 정수 ID로 굳힌 표를 검증한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = reference_bundle()
        cls.layout = plan_weight_blob(cls.bundle)
        cls.graph = cls.bundle.graph_for_bucket(REFERENCE_FRAMES)
        cls.table = build_tensor_table(cls.graph, cls.layout)

    def test_every_tensor_id_is_unique_and_dense(self) -> None:
        ids = [descriptor.tensor_id for descriptor in self.table.descriptors]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(ids, list(range(len(self.table))))
        self.assertEqual(len(self.table.ids), len(self.table))

    def test_names_resolve_to_ids_and_back(self) -> None:
        for tensor in self.graph.tensors:
            self.assertEqual(self.table.id_of(tensor.name), tensor.tensor_id)
            self.assertEqual(self.table.name_of(tensor.tensor_id), tensor.name)

    def test_byte_size_matches_shape_times_dtype(self) -> None:
        for descriptor in self.table.descriptors:
            dimensions = descriptor.dimensions[: descriptor.rank]
            expected = DTYPE_BYTE_SIZE[TensorDType(descriptor.dtype)]
            for dimension in dimensions:
                expected *= dimension
            with self.subTest(tensor_id=descriptor.tensor_id):
                self.assertEqual(descriptor.logical_byte_size, expected)
                self.assertEqual(descriptor.storage_span_bytes, expected)

    def test_unused_rank_slots_are_normalized(self) -> None:
        for descriptor in self.table.descriptors:
            for axis in range(descriptor.rank, TENSOR_MAX_RANK):
                self.assertEqual(descriptor.dimensions[axis], 1)
                self.assertEqual(descriptor.byte_strides[axis], 0)

    def test_feature_and_embedding_are_pinned(self) -> None:
        feature = self.table.descriptors[self.table.id_of("feature")]
        embedding = self.table.descriptors[self.table.id_of("embedding")]
        self.assertEqual(feature.storage_type, TensorStorageType.INPUT)
        self.assertEqual(embedding.storage_type, TensorStorageType.OUTPUT)
        self.assertEqual(
            feature.dimensions[: feature.rank], (1, REFERENCE_FRAMES, 80)
        )
        self.assertEqual(embedding.dimensions[: embedding.rank], (1, 192))
        # 입력 버퍼는 외부에서 연결하므로 plan이 위치를 정하지 않는다.
        self.assertEqual(feature.data_offset, INVALID_DATA_OFFSET)

    def test_only_constants_carry_a_weights_offset(self) -> None:
        for descriptor in self.table.descriptors:
            with self.subTest(tensor_id=descriptor.tensor_id):
                if descriptor.storage_type == TensorStorageType.CONSTANT:
                    self.assertNotEqual(descriptor.data_offset, INVALID_DATA_OFFSET)
                    self.assertTrue(descriptor.flags & TensorFlags.READ_ONLY)
                    self.assertTrue(descriptor.flags & TensorFlags.EXTERNAL)
                else:
                    # Phase 3은 Arena를 아직 계획하지 않는다.
                    self.assertEqual(descriptor.data_offset, INVALID_DATA_OFFSET)

    def test_lifetime_covers_producer_and_consumers(self) -> None:
        for tensor in self.graph.tensors:
            descriptor = self.table.descriptors[tensor.tensor_id]
            with self.subTest(tensor=tensor.name):
                self.assertLessEqual(descriptor.first_use, descriptor.last_use)
                if tensor.producer is not None:
                    self.assertEqual(descriptor.first_use, tensor.producer)
                if tensor.consumers:
                    self.assertEqual(descriptor.last_use, tensor.consumers[-1])

    def test_packed_table_has_the_fixed_record_size(self) -> None:
        packed = self.table.pack()
        self.assertEqual(len(packed), len(self.table) * TENSOR_DESCRIPTOR_SIZE)
        self.assertEqual(len(packed), self.table.byte_size)

    def test_weight_blob_entries_do_not_overlap(self) -> None:
        spans = sorted(
            (entry.offset, entry.offset + entry.byte_size)
            for entry in self.layout.entries
        )
        for (_, end), (start, _) in zip(spans, spans[1:]):
            self.assertLessEqual(end, start)
        self.assertLessEqual(spans[-1][1], self.layout.total_bytes)
        for entry in self.layout.entries:
            self.assertEqual(entry.offset % WEIGHT_BLOB_ALIGNMENT, 0)

    def test_shared_weights_are_stored_once_and_bucket_locals_per_bucket(self) -> None:
        shared = [
            entry
            for entry in self.layout.entries
            if entry.scope is InitializerScope.SHARED
        ]
        bucket_local = [
            entry
            for entry in self.layout.entries
            if entry.scope is InitializerScope.BUCKET_LOCAL
        ]
        self.assertEqual(len(shared), REFERENCE_SHARED_INITIALIZERS)
        self.assertEqual(
            len(bucket_local),
            REFERENCE_BUCKET_LOCAL_INITIALIZERS * len(BUCKETS),
        )
        self.assertEqual(len({entry.name for entry in shared}), len(shared))

    def test_same_tensor_id_points_at_a_different_offset_per_bucket(self) -> None:
        """plan_98의 Tensor N과 plan_298의 Tensor N은 다른 weight를 가리킨다."""

        tables = {
            graph.bucket_frames: build_tensor_table(graph, self.layout)
            for graph in self.bundle.graphs
        }
        reference = tables[REFERENCE_FRAMES]
        bucket_local_ids = [
            item.tensor_id
            for item in self.graph.initializers
            if item.scope is InitializerScope.BUCKET_LOCAL
        ]
        moved = [
            tensor_id
            for tensor_id in bucket_local_ids
            if len(
                {
                    table.descriptors[tensor_id].data_offset
                    for table in tables.values()
                }
            )
            == len(tables)
        ]
        self.assertEqual(len(moved), len(bucket_local_ids))

        shared_id = next(
            item.tensor_id
            for item in self.graph.initializers
            if item.scope is InitializerScope.SHARED
        )
        offsets = {
            table.descriptors[shared_id].data_offset for table in tables.values()
        }
        self.assertEqual(len(offsets), 1)
        self.assertEqual(
            reference.descriptors[shared_id].data_offset, offsets.pop()
        )


@unittest.skipUnless(ONNX_AVAILABLE, "onnx가 설치되어 있지 않다")
@unittest.skipUnless(artifacts_present(), "정적 ONNX 또는 graph IR 산출물이 없다")
class OperatorTableTests(unittest.TestCase):
    """1,438개 node를 실행 명령 표로 굳힌 결과를 검증한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = reference_bundle().graph_for_bucket(REFERENCE_FRAMES)
        cls.table = build_operator_table(cls.graph)

    def test_one_command_per_runtime_node(self) -> None:
        self.assertEqual(len(self.table), REFERENCE_RUNTIME_NODES)
        ids = [descriptor.operator_id for descriptor in self.table.descriptors]
        self.assertEqual(ids, list(range(REFERENCE_RUNTIME_NODES)))

    def test_inputs_are_only_read_after_they_are_produced(self) -> None:
        """단계 5의 완료 조건. 표를 위에서 아래로 읽어 독립적으로 확인한다."""

        ready = {
            tensor.tensor_id
            for tensor in self.graph.tensors
            if tensor.storage_type
            in (TensorStorageType.INPUT, TensorStorageType.CONSTANT)
        }
        for descriptor in self.table.descriptors:
            used = descriptor.input_tensor_ids[: descriptor.input_count]
            for tensor_id in used:
                with self.subTest(operator_id=descriptor.operator_id):
                    self.assertIn(tensor_id, ready)
            ready.update(descriptor.output_tensor_ids)

    def test_unused_input_slots_are_invalid(self) -> None:
        for descriptor in self.table.descriptors:
            slots = descriptor.input_tensor_ids
            self.assertEqual(len(slots), OPERATOR_INPUT_CAPACITY)
            for index in range(descriptor.input_count, OPERATOR_INPUT_CAPACITY):
                self.assertEqual(slots[index], INVALID_TENSOR_ID)

    def test_every_opcode_is_a_frozen_runtime_opcode(self) -> None:
        opcodes = {
            OperatorCode(descriptor.opcode) for descriptor in self.table.descriptors
        }
        self.assertEqual(len(opcodes), 20)
        self.assertNotIn(OperatorCode.INVALID, opcodes)
        self.assertIn(OperatorCode.QLINEAR_CONV, opcodes)

    def test_repeated_opcodes_reuse_one_kernel_slot(self) -> None:
        """225개 QLinearConv는 225개 C 함수가 아니라 같은 kernel 호출 225번이다."""

        convs = [
            descriptor
            for descriptor in self.table.descriptors
            if descriptor.opcode == OperatorCode.QLINEAR_CONV
        ]
        self.assertEqual(len(convs), 225)
        self.assertEqual({descriptor.kernel_id for descriptor in convs}, {0})
        self.assertEqual({descriptor.backend_id for descriptor in convs}, {1})

    def test_attributes_survive_the_round_trip(self) -> None:
        transpose = self.table.id_of("/Transpose")
        self.assertEqual(
            self.table.descriptors[transpose].opcode, OperatorCode.TRANSPOSE
        )
        self.assertEqual(self.table.attributes_of(transpose), {"perm": (0, 2, 1)})

        conv_id = next(
            descriptor.operator_id
            for descriptor in self.table.descriptors
            if descriptor.opcode == OperatorCode.QLINEAR_CONV
        )
        attributes = self.table.attributes_of(conv_id)
        self.assertEqual(
            set(attributes),
            {"kernel_shape", "pads", "strides", "dilations", "group"},
        )
        self.assertEqual(len(attributes["pads"]), 2 * len(attributes["kernel_shape"]))

    def test_operators_without_attributes_carry_an_empty_block(self) -> None:
        relu = next(
            descriptor
            for descriptor in self.table.descriptors
            if descriptor.opcode == OperatorCode.RELU
        )
        self.assertEqual(relu.attribute_size, 0)
        self.assertEqual(relu.attribute_offset, 0)
        self.assertEqual(self.table.attributes_of(relu.operator_id), {})

    def test_identical_attributes_share_one_block(self) -> None:
        with_attributes = [
            descriptor
            for descriptor in self.table.descriptors
            if descriptor.attribute_size > 0
        ]
        offsets = {descriptor.attribute_offset for descriptor in with_attributes}
        self.assertLess(len(offsets), len(with_attributes))
        self.assertEqual(len(offsets), self.table.attribute_blocks)
        for offset in offsets:
            self.assertEqual(offset % PLAN_SECTION_ALIGNMENT, 0)

    def test_attribute_offsets_stay_inside_the_section(self) -> None:
        for descriptor in self.table.descriptors:
            if descriptor.attribute_size == 0:
                continue
            end = descriptor.attribute_offset + descriptor.attribute_size
            self.assertLessEqual(end, self.table.attribute_section_size)

    def test_packed_table_has_the_fixed_record_size(self) -> None:
        packed = self.table.pack()
        self.assertEqual(len(packed), len(self.table) * OPERATOR_DESCRIPTOR_SIZE)
        self.assertEqual(len(packed), self.table.byte_size)

    def test_shuffled_operators_cannot_even_be_constructed(self) -> None:
        """RuntimeGraph가 먼저 막는다. 표를 만들 기회조차 없어야 한다."""

        operators = list(self.graph.operators)
        operators[10], operators[20] = operators[20], operators[10]
        with self.assertRaises(RuntimeIRError):
            dataclasses.replace(self.graph, operators=tuple(operators))


class ExecutionOrderValidatorTests(unittest.TestCase):
    """검증기 자체를 손으로 만든 최소 그래프로 시험한다.

    RuntimeGraph는 잘못된 순서를 애초에 거부하므로, 검증기가 정말 순서를 보는지
    확인하려면 그 불변식 밖에서 만든 그래프가 필요하다.
    """

    class FakeGraph:
        def __init__(self, tensors, operators) -> None:
            self.tensors = tensors
            self.operators = operators

        def tensor(self, tensor_id: int):
            return self.tensors[tensor_id]

    @staticmethod
    def make_tensor(
        tensor_id: int,
        storage_type: TensorStorageType,
        producer: int | None,
        consumers: tuple[int, ...],
    ):
        return RuntimeTensor(
            name=f"t{tensor_id}",
            tensor_id=tensor_id,
            dtype=TensorDType.FLOAT32,
            shape=(2,),
            strides=(4,),
            byte_size=8,
            storage_type=storage_type,
            producer=producer,
            consumers=consumers,
        )

    @staticmethod
    def make_operator(operator_id: int, inputs: tuple[int, ...], output: int):
        return RuntimeOperator(
            name=f"op{operator_id}",
            operator_id=operator_id,
            opcode=OperatorCode.RELU,
            input_tensor_ids=inputs,
            output_tensor_ids=(output,),
        )

    def test_accepts_a_correctly_ordered_graph(self) -> None:
        graph = self.FakeGraph(
            tensors=[
                self.make_tensor(0, TensorStorageType.INPUT, None, (0,)),
                self.make_tensor(1, TensorStorageType.ACTIVATION, 0, (1,)),
                self.make_tensor(2, TensorStorageType.OUTPUT, 1, ()),
            ],
            operators=[
                self.make_operator(0, (0,), 1),
                self.make_operator(1, (1,), 2),
            ],
        )
        validate_execution_order(graph)  # 예외가 없으면 통과다

    def test_rejects_use_before_produce(self) -> None:
        graph = self.FakeGraph(
            tensors=[
                self.make_tensor(0, TensorStorageType.INPUT, None, (0,)),
                self.make_tensor(1, TensorStorageType.ACTIVATION, 1, (0,)),
                self.make_tensor(2, TensorStorageType.OUTPUT, 0, ()),
            ],
            operators=[
                self.make_operator(0, (0, 1), 2),
                self.make_operator(1, (0,), 1),
            ],
        )
        with self.assertRaises(OperatorTableError) as caught:
            validate_execution_order(graph)
        self.assertIn("생산되지 않은", str(caught.exception))

    def test_rejects_a_tensor_no_operator_produces(self) -> None:
        graph = self.FakeGraph(
            tensors=[
                self.make_tensor(0, TensorStorageType.INPUT, None, (0,)),
                self.make_tensor(1, TensorStorageType.OUTPUT, 0, ()),
                self.make_tensor(2, TensorStorageType.ACTIVATION, None, ()),
            ],
            operators=[self.make_operator(0, (0,), 1)],
        )
        with self.assertRaises(OperatorTableError):
            validate_execution_order(graph)


@unittest.skipUnless(ONNX_AVAILABLE, "onnx가 설치되어 있지 않다")
@unittest.skipUnless(artifacts_present(), "정적 ONNX 또는 graph IR 산출물이 없다")
class BundleSerializationTests(unittest.TestCase):
    """단계 6의 완료 조건: 쓴 파일을 되읽으면 원본 RuntimeGraph와 일치한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = reference_bundle()
        cls.layout = plan_weight_blob(cls.bundle)
        cls._directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._directory.name)

        cls.weights = write_weight_blob(
            cls.bundle, cls.layout, cls.root / WEIGHT_BLOB_FILE_NAME
        )
        cls.plans = tuple(
            write_execution_plan(
                graph,
                cls.layout,
                cls.root
                / EXECUTION_PLAN_DIR_NAME
                / plan_file_name(graph.bucket_frames),
            )
            for graph in cls.bundle.graphs
        )
        cls.manifest = write_bundle_manifest(
            cls.root / MANIFEST_FILE_NAME,
            bundle=cls.bundle,
            layout=cls.layout,
            weights=cls.weights,
            plans=cls.plans,
            canonical_model=ROOT / "models" / "source" / MODEL_FILE_NAME,
            graph_manifest=GRAPH_DIR / "campp_graph_manifest.json",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    # -- weights.bin -------------------------------------------------------

    def test_weight_blob_round_trips_every_initializer(self) -> None:
        blob = read_weight_blob(self.root / WEIGHT_BLOB_FILE_NAME)
        self.assertEqual(len(blob), self.layout.total_bytes)
        verify_weight_blob(blob, self.bundle, self.layout, records=self.weights.records)

        for graph in self.bundle.graphs:
            for item in graph.initializers:
                offset = self.layout.offset_of(graph.bucket_frames, item.tensor_id)
                with self.subTest(bucket=graph.bucket_frames, name=item.name):
                    self.assertEqual(
                        blob[offset : offset + item.byte_size], item.raw_data
                    )

    def test_weight_blob_is_deterministic(self) -> None:
        first, _ = build_weight_blob(self.bundle, self.layout)
        second, _ = build_weight_blob(self.bundle, self.layout)
        self.assertEqual(first, second)
        self.assertEqual(
            hashlib.sha256(first).hexdigest(), self.weights.sha256
        )

    def test_weight_records_describe_every_entry(self) -> None:
        self.assertEqual(len(self.weights.records), len(self.layout.entries))
        blob = read_weight_blob(self.root / WEIGHT_BLOB_FILE_NAME)
        for record in self.weights.records:
            with self.subTest(name=record.name, bucket=record.bucket_frames):
                stored = blob[record.offset : record.offset + record.byte_size]
                self.assertEqual(
                    hashlib.sha256(stored).hexdigest(), record.sha256
                )
                expected = DTYPE_BYTE_SIZE[record.dtype]
                for dimension in record.shape:
                    expected *= dimension
                self.assertEqual(record.byte_size, expected)

    def test_padding_between_entries_is_zero(self) -> None:
        blob = read_weight_blob(self.root / WEIGHT_BLOB_FILE_NAME)
        spans = sorted(
            (entry.offset, entry.offset + entry.byte_size)
            for entry in self.layout.entries
        )
        for (_, end), (start, _) in zip(spans, spans[1:]):
            if end != start:
                self.assertEqual(blob[end:start], b"\x00" * (start - end))

    def test_corrupted_weight_bytes_are_detected(self) -> None:
        blob = bytearray(read_weight_blob(self.root / WEIGHT_BLOB_FILE_NAME))
        target = self.weights.records[0]
        blob[target.offset] ^= 0xFF
        with self.assertRaises(WeightBlobError):
            verify_weight_blob(bytes(blob), self.bundle, self.layout)

    # -- plan_*.bin --------------------------------------------------------

    def test_plan_round_trips_into_the_original_graph(self) -> None:
        for graph in self.bundle.graphs:
            with self.subTest(bucket=graph.bucket_frames):
                loaded = read_execution_plan(
                    self.root
                    / EXECUTION_PLAN_DIR_NAME
                    / plan_file_name(graph.bucket_frames)
                )
                verify_plan(loaded, graph, self.layout)

    def test_plan_header_describes_the_sections(self) -> None:
        graph = self.bundle.graph_for_bucket(REFERENCE_FRAMES)
        loaded = read_execution_plan(
            self.root / EXECUTION_PLAN_DIR_NAME / plan_file_name(REFERENCE_FRAMES)
        )
        header = loaded.header
        self.assertEqual(header.magic, PLAN_MAGIC)
        self.assertEqual(header.format_version, PLAN_FORMAT_VERSION)
        self.assertEqual(header.bucket_frames, REFERENCE_FRAMES)
        self.assertEqual(header.tensor_count, len(graph.tensors))
        self.assertEqual(header.operator_count, len(graph.operators))
        self.assertEqual(header.tensor_table_offset, PLAN_HEADER_SIZE)
        self.assertEqual(
            header.operator_table_offset,
            header.tensor_table_offset + header.tensor_count * TENSOR_DESCRIPTOR_SIZE,
        )
        self.assertEqual(
            header.attribute_section_offset,
            header.operator_table_offset
            + header.operator_count * OPERATOR_DESCRIPTOR_SIZE,
        )
        for offset in (
            header.tensor_table_offset,
            header.operator_table_offset,
            header.attribute_section_offset,
        ):
            self.assertEqual(offset % PLAN_SECTION_ALIGNMENT, 0)

    def test_plans_differ_per_bucket_but_share_the_shape_of_the_table(self) -> None:
        digests = {plan.sha256 for plan in self.plans}
        self.assertEqual(len(digests), len(self.plans))
        self.assertEqual({plan.byte_size for plan in self.plans}, {390800})
        self.assertEqual({plan.tensor_count for plan in self.plans}, {3719})
        self.assertEqual(
            {plan.operator_count for plan in self.plans},
            {REFERENCE_RUNTIME_NODES},
        )

    def test_corrupted_plan_payload_fails_the_checksum(self) -> None:
        path = self.root / EXECUTION_PLAN_DIR_NAME / "tampered.bin"
        data = bytearray(
            (
                self.root
                / EXECUTION_PLAN_DIR_NAME
                / plan_file_name(REFERENCE_FRAMES)
            ).read_bytes()
        )
        data[PLAN_HEADER_SIZE + 16] ^= 0xFF
        path.write_bytes(bytes(data))
        with self.assertRaises(ExecutionPlanError):
            read_execution_plan(path)

    def test_plan_mismatch_against_another_bucket_is_reported(self) -> None:
        loaded = read_execution_plan(
            self.root / EXECUTION_PLAN_DIR_NAME / plan_file_name(REFERENCE_FRAMES)
        )
        other = self.bundle.graph_for_bucket(998)
        with self.assertRaises(ExecutionPlanError):
            verify_plan(loaded, other, self.layout)

    # -- manifest.json -----------------------------------------------------

    def test_manifest_records_the_whole_bundle(self) -> None:
        document = self.manifest.document
        self.assertEqual(document["format_version"], PLAN_FORMAT_VERSION)
        self.assertEqual(document["weights"]["sha256"], self.weights.sha256)
        self.assertEqual(document["weights"]["entry_count"], len(self.layout.entries))
        self.assertEqual(
            [entry["bucket_frames"] for entry in document["plans"]],
            sorted(frames for frames, _ in BUCKETS),
        )
        for entry, plan in zip(
            document["plans"], sorted(self.plans, key=lambda item: item.bucket_frames)
        ):
            self.assertEqual(entry["sha256"], plan.sha256)
            self.assertEqual(entry["operator_count"], REFERENCE_RUNTIME_NODES)
        self.assertEqual(len(document["canonical_model"]["sha256"]), 64)
        self.assertEqual(len(document["graph_manifest"]["sha256"]), 64)

    def test_manifest_paths_are_relative(self) -> None:
        document = self.manifest.document
        self.assertEqual(document["weights"]["path"], WEIGHT_BLOB_FILE_NAME)
        for entry in document["plans"]:
            self.assertFalse(Path(entry["path"]).is_absolute())
        self.assertFalse(Path(document["canonical_model"]["path"]).is_absolute())

    def test_manifest_verifies_the_files_it_points_at(self) -> None:
        verify_bundle_manifest(self.root / MANIFEST_FILE_NAME)

    def test_manifest_detects_a_changed_file(self) -> None:
        target = self.root / EXECUTION_PLAN_DIR_NAME / plan_file_name(998)
        original = target.read_bytes()
        try:
            target.write_bytes(original[:-1])
            with self.assertRaises(BundleManifestError):
                verify_bundle_manifest(self.root / MANIFEST_FILE_NAME)
        finally:
            target.write_bytes(original)

    def test_weight_index_can_be_left_out(self) -> None:
        lean = write_bundle_manifest(
            self.root / "manifest_lean.json",
            bundle=self.bundle,
            layout=self.layout,
            weights=self.weights,
            plans=self.plans,
            include_weight_index=False,
        )
        self.assertNotIn("weight_index", lean.document)
        self.assertIn("weight_index", self.manifest.document)
        self.assertLess(lean.byte_size, self.manifest.byte_size // 100)


if __name__ == "__main__":
    unittest.main()
