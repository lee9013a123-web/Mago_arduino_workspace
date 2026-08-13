from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from runtime_bundle_exporter.builder.planner.tensor_arena_planner import (  # noqa: E402
    TensorArenaPlanningError,
    plan_tensor_arena,
    plan_tensor_arena_from_descriptors,
)
from runtime_bundle_exporter.builder.tensor_table_builder import (  # noqa: E402
    build_tensor_table,
    plan_weight_blob,
)
from runtime_bundle_exporter.format.binary_format_schema import (  # noqa: E402
    INVALID_DATA_OFFSET,
    TENSOR_DESCRIPTOR_SIZE,
    OperatorCode,
    TensorDType,
    TensorFlags,
    TensorStorageType,
)
from runtime_bundle_exporter.runtime_ir import (  # noqa: E402
    RuntimeBundle,
    RuntimeGraph,
    RuntimeOperator,
    RuntimeTensor,
)
from runtime_bundle_exporter.writer.execution_plan_writer import (  # noqa: E402
    ExecutionPlanError,
    read_execution_plan,
    verify_plan,
    write_execution_plan,
)
from runtime_bundle_exporter.writer.bundle_manifest_writer import (  # noqa: E402
    write_bundle_manifest,
)
from runtime_bundle_exporter.writer.weight_blob_writer import (  # noqa: E402
    write_weight_blob,
)


def _tensor(
    tensor_id: int,
    *,
    storage_type: TensorStorageType,
    producer: int | None,
    consumers: tuple[int, ...],
) -> RuntimeTensor:
    return RuntimeTensor(
        name=f"tensor_{tensor_id}",
        tensor_id=tensor_id,
        dtype=TensorDType.FLOAT32,
        shape=(16,),
        strides=(4,),
        byte_size=64,
        storage_type=storage_type,
        producer=producer,
        consumers=consumers,
    )


def make_arena_graph() -> RuntimeGraph:
    # Tensor 1은 Operator 1과 3에서 사용된다. 중간에 쉬더라도 마지막 소비자인
    # Operator 3까지 살아 있어야 한다.
    tensors = (
        _tensor(
            0,
            storage_type=TensorStorageType.INPUT,
            producer=None,
            consumers=(0,),
        ),
        _tensor(
            1,
            storage_type=TensorStorageType.ACTIVATION,
            producer=0,
            consumers=(1, 3),
        ),
        _tensor(
            2,
            storage_type=TensorStorageType.ACTIVATION,
            producer=1,
            consumers=(2,),
        ),
        _tensor(
            3,
            storage_type=TensorStorageType.ACTIVATION,
            producer=2,
            consumers=(3,),
        ),
        _tensor(
            4,
            storage_type=TensorStorageType.ACTIVATION,
            producer=3,
            consumers=(4,),
        ),
        _tensor(
            5,
            storage_type=TensorStorageType.OUTPUT,
            producer=4,
            consumers=(),
        ),
    )
    operators = (
        RuntimeOperator(
            name="relu_0",
            operator_id=0,
            opcode=OperatorCode.RELU,
            input_tensor_ids=(0,),
            output_tensor_ids=(1,),
        ),
        RuntimeOperator(
            name="relu_1",
            operator_id=1,
            opcode=OperatorCode.RELU,
            input_tensor_ids=(1,),
            output_tensor_ids=(2,),
        ),
        RuntimeOperator(
            name="relu_2",
            operator_id=2,
            opcode=OperatorCode.RELU,
            input_tensor_ids=(2,),
            output_tensor_ids=(3,),
        ),
        RuntimeOperator(
            name="add_3",
            operator_id=3,
            opcode=OperatorCode.ADD,
            input_tensor_ids=(1, 3),
            output_tensor_ids=(4,),
        ),
        RuntimeOperator(
            name="relu_4",
            operator_id=4,
            opcode=OperatorCode.RELU,
            input_tensor_ids=(4,),
            output_tensor_ids=(5,),
        ),
    )
    return RuntimeGraph(
        name="arena_test",
        bucket_frames=16,
        tensors=tensors,
        operators=operators,
        initializers=(),
    )


class TensorArenaPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = make_arena_graph()
        self.weight_layout = plan_weight_blob(RuntimeBundle((self.graph,)))

    def test_offline_layout_reuses_only_expired_tensor_regions(self) -> None:
        layout = plan_tensor_arena(self.graph, alignment=64)

        self.assertEqual(layout.offset_of(1), 0)
        self.assertEqual(layout.offset_of(2), 64)
        # Tensor 2는 Operator 2에서 입력이고 Tensor 3은 같은 Operator의
        # 출력이다. last_use == first_use이므로 같은 주소를 쓰면 안 된다.
        self.assertEqual(layout.offset_of(3), 128)
        # Tensor 2는 Operator 2 이후 죽으므로 Operator 3 출력이 재사용한다.
        self.assertEqual(layout.offset_of(4), 64)
        # 마지막 Operator에서 Tensor 1/3이 모두 죽었으므로 output이 0을 쓴다.
        self.assertEqual(layout.offset_of(5), 0)
        self.assertEqual(layout.arena_size, 192)
        self.assertEqual(layout.theoretical_peak_bytes, 192)
        self.assertEqual(layout.naive_activation_bytes, 320)
        self.assertEqual(layout.saved_bytes, 128)
        self.assertEqual(layout.packing_efficiency, 1.0)

    def test_tensor_with_a_consumer_gap_stays_live_until_last_consumer(self) -> None:
        allocation = plan_tensor_arena(self.graph).allocation(1)
        self.assertEqual((allocation.first_use, allocation.last_use), (0, 3))

    def test_layout_is_deterministic(self) -> None:
        first = plan_tensor_arena(self.graph)
        second = plan_tensor_arena(self.graph)
        self.assertEqual(first, second)

    def test_budget_smaller_than_safe_layout_is_rejected(self) -> None:
        with self.assertRaisesRegex(TensorArenaPlanningError, "예산"):
            plan_tensor_arena(self.graph, max_arena_bytes=191)

    def test_alignment_must_be_a_power_of_two(self) -> None:
        with self.assertRaisesRegex(TensorArenaPlanningError, "2의 거듭제곱"):
            plan_tensor_arena(self.graph, alignment=48)

    def test_existing_descriptor_gets_offset_without_changing_its_size(self) -> None:
        arena = plan_tensor_arena(self.graph)
        reference = build_tensor_table(self.graph, self.weight_layout)
        planned = build_tensor_table(
            self.graph, self.weight_layout, arena_layout=arena
        )

        for tensor_id in range(len(self.graph.tensors)):
            before = reference.descriptors[tensor_id]
            after = planned.descriptors[tensor_id]
            self.assertEqual(len(after.pack()), TENSOR_DESCRIPTOR_SIZE)
            self.assertEqual(TENSOR_DESCRIPTOR_SIZE, 80)
            if after.storage_type in (
                TensorStorageType.ACTIVATION,
                TensorStorageType.OUTPUT,
            ):
                self.assertEqual(after.data_offset, arena.offset_of(tensor_id))
                self.assertTrue(after.flags & TensorFlags.DENSE_SLAB)
                self.assertEqual(before.data_offset, INVALID_DATA_OFFSET)
                self.assertFalse(before.flags & TensorFlags.DENSE_SLAB)
            else:
                self.assertEqual(after.data_offset, before.data_offset)

    def test_reference_descriptors_produce_the_same_layout(self) -> None:
        reference = build_tensor_table(self.graph, self.weight_layout)
        from_graph = plan_tensor_arena(self.graph)
        from_descriptors = plan_tensor_arena_from_descriptors(
            reference.descriptors,
            bucket_frames=self.graph.bucket_frames,
            operator_count=len(self.graph.operators),
        )
        self.assertEqual(from_descriptors, from_graph)

    def test_arena_plan_round_trips_and_requires_the_same_layout_to_verify(self) -> None:
        arena = plan_tensor_arena(self.graph)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan_16.bin"
            result = write_execution_plan(
                self.graph,
                self.weight_layout,
                path,
                arena_layout=arena,
            )
            loaded = read_execution_plan(path)

        self.assertEqual(result.arena_size, arena.arena_size)
        self.assertEqual(result.arena_alignment, arena.alignment)
        verify_plan(
            loaded,
            self.graph,
            self.weight_layout,
            arena_layout=arena,
        )
        with self.assertRaises(ExecutionPlanError):
            verify_plan(loaded, self.graph, self.weight_layout)

    def test_only_arena_manifest_records_the_memory_layout(self) -> None:
        arena = plan_tensor_arena(self.graph)
        bundle = RuntimeBundle((self.graph,))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = write_weight_blob(
                bundle, self.weight_layout, root / "weights.bin"
            )
            reference_plan = write_execution_plan(
                self.graph, self.weight_layout, root / "plan_reference.bin"
            )
            arena_plan = write_execution_plan(
                self.graph,
                self.weight_layout,
                root / "plan_arena.bin",
                arena_layout=arena,
            )
            reference_manifest = write_bundle_manifest(
                root / "reference_manifest.json",
                bundle=bundle,
                layout=self.weight_layout,
                weights=weights,
                plans=(reference_plan,),
            ).document
            arena_manifest = write_bundle_manifest(
                root / "arena_manifest.json",
                bundle=bundle,
                layout=self.weight_layout,
                weights=weights,
                plans=(arena_plan,),
            ).document

        self.assertNotIn("memory_layout", reference_manifest)
        self.assertNotIn("arena_size_bytes", reference_manifest["plans"][0])
        self.assertEqual(arena_manifest["memory_layout"], "tensor_arena")
        self.assertEqual(
            arena_manifest["plans"][0]["arena_size_bytes"], arena.arena_size
        )
        self.assertEqual(
            arena_manifest["plans"][0]["arena_alignment"], arena.alignment
        )


if __name__ == "__main__":
    unittest.main()
