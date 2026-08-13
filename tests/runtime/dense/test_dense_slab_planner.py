from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from runtime_bundle_exporter.builder.planner.dense_slab_planner import (  # noqa: E402
    inspect_compiled_dense_concats,
    rewrite_dense_concats_as_slabs,
)
from runtime_bundle_exporter.builder.planner.tensor_arena_planner import (  # noqa: E402
    plan_tensor_arena,
    plan_tensor_arena_from_descriptors,
)
from runtime_bundle_exporter.builder.tensor_table_builder import (  # noqa: E402
    build_tensor_table,
    plan_weight_blob,
)
from runtime_bundle_exporter.format.binary_format_schema import (  # noqa: E402
    OperatorCode,
    TensorDType,
    TensorStorageType,
)
from runtime_bundle_exporter.runtime_ir import (  # noqa: E402
    RuntimeBundle,
    RuntimeGraph,
    RuntimeOperator,
    RuntimeTensor,
)
from runtime_bundle_exporter.writer.execution_plan_writer import (  # noqa: E402
    read_execution_plan,
    verify_plan,
    write_execution_plan,
)


def _tensor(
    tensor_id: int,
    name: str,
    shape: tuple[int, ...],
    storage_type: TensorStorageType,
    producer: int | None,
    consumers: tuple[int, ...],
) -> RuntimeTensor:
    strides = (shape[1] * shape[2] * 4, shape[2] * 4, 4)
    return RuntimeTensor(
        name=name,
        tensor_id=tensor_id,
        dtype=TensorDType.FLOAT32,
        shape=shape,
        strides=strides,
        byte_size=shape[0] * shape[1] * shape[2] * 4,
        storage_type=storage_type,
        producer=producer,
        consumers=consumers,
    )


def make_dense_graph() -> RuntimeGraph:
    tensors = (
        _tensor(0, "input", (1, 2, 3), TensorStorageType.INPUT, None, (0,)),
        _tensor(1, "base", (1, 2, 3), TensorStorageType.ACTIVATION, 0, (1, 2)),
        _tensor(2, "feature_1", (1, 1, 3), TensorStorageType.ACTIVATION, 1, (2,)),
        _tensor(3, "prefix_1", (1, 3, 3), TensorStorageType.ACTIVATION, 2, (3, 4)),
        _tensor(4, "feature_2", (1, 1, 3), TensorStorageType.ACTIVATION, 3, (4,)),
        _tensor(5, "prefix_2", (1, 4, 3), TensorStorageType.ACTIVATION, 4, (5,)),
        _tensor(6, "output", (1, 4, 3), TensorStorageType.OUTPUT, 5, ()),
    )
    operators = (
        RuntimeOperator("base_relu", 0, OperatorCode.RELU, (0,), (1,)),
        RuntimeOperator("feature_1_relu", 1, OperatorCode.RELU, (1,), (2,)),
        RuntimeOperator(
            "dense_concat_1", 2, OperatorCode.CONCAT, (1, 2), (3,), {"axis": 1}
        ),
        RuntimeOperator("feature_2_relu", 3, OperatorCode.RELU, (3,), (4,)),
        RuntimeOperator(
            "dense_concat_2", 4, OperatorCode.CONCAT, (3, 4), (5,), {"axis": 1}
        ),
        RuntimeOperator("output_relu", 5, OperatorCode.RELU, (5,), (6,)),
    )
    return RuntimeGraph(
        tensors=tensors,
        operators=operators,
        initializers=(),
        name="dense_test",
        bucket_frames=3,
    )


class DenseSlabPlannerTests(unittest.TestCase):
    def test_existing_plan_exposes_three_dense_chains_without_names(self) -> None:
        path = (
            ROOT
            / "runs"
            / "runtime"
            / "tensor_arena"
            / "bundle"
            / "execution_plans"
            / "plan_998.bin"
        )
        if not path.is_file():
            self.skipTest("existing Tensor Arena plan is not present")
        blocks = inspect_compiled_dense_concats(read_execution_plan(path))
        self.assertEqual([len(block.concat_operator_ids) for block in blocks], [12, 24, 16])
        self.assertEqual([block.slab_shape for block in blocks], [
            (1, 512, 499),
            (1, 1024, 499),
            (1, 1024, 499),
        ])

    def test_rewrite_removes_concat_and_preserves_tensor_ids(self) -> None:
        result = rewrite_dense_concats_as_slabs(
            make_dense_graph(), expected_block_count=1, expected_concat_count=2
        )
        graph = result.graph

        self.assertEqual(result.removed_concat_count, 2)
        self.assertEqual(len(graph.operators), 4)
        self.assertFalse(any(op.opcode is OperatorCode.CONCAT for op in graph.operators))
        self.assertEqual([op.operator_id for op in graph.operators], list(range(4)))
        self.assertEqual([tensor.tensor_id for tensor in graph.tensors], list(range(7)))

    def test_feature_slices_and_prefixes_alias_one_backing_slab(self) -> None:
        graph = rewrite_dense_concats_as_slabs(make_dense_graph()).graph
        base = graph.tensor(1)
        feature_1 = graph.tensor(2)
        prefix_1 = graph.tensor(3)
        feature_2 = graph.tensor(4)
        prefix_2 = graph.tensor(5)

        self.assertEqual(base.storage_span_bytes, 48)
        self.assertEqual(feature_1.storage_type, TensorStorageType.VIEW)
        self.assertEqual((feature_1.alias_of_tensor_id, feature_1.view_byte_offset), (1, 24))
        self.assertEqual((prefix_1.alias_of_tensor_id, prefix_1.view_byte_offset), (1, 0))
        self.assertEqual((feature_2.alias_of_tensor_id, feature_2.view_byte_offset), (1, 36))
        self.assertEqual((prefix_2.alias_of_tensor_id, prefix_2.view_byte_offset), (1, 0))

    def test_arena_allocates_backing_but_not_views(self) -> None:
        graph = rewrite_dense_concats_as_slabs(make_dense_graph()).graph
        arena = plan_tensor_arena(graph)

        self.assertEqual(arena.allocation(1).byte_size, 48)
        self.assertEqual((arena.allocation(1).first_use, arena.allocation(1).last_use), (0, 3))
        allocated = {item.tensor_id for item in arena.allocations}
        self.assertNotIn(2, allocated)
        self.assertNotIn(3, allocated)
        self.assertNotIn(4, allocated)
        self.assertNotIn(5, allocated)

    def test_tensor_descriptors_encode_alias_target_and_offset(self) -> None:
        graph = rewrite_dense_concats_as_slabs(make_dense_graph()).graph
        bundle = RuntimeBundle((graph,))
        weights = plan_weight_blob(bundle)
        arena = plan_tensor_arena(graph)
        table = build_tensor_table(graph, weights, arena_layout=arena)

        self.assertEqual(table.descriptors[1].storage_span_bytes, 48)
        self.assertEqual(table.descriptors[2].storage_type, TensorStorageType.VIEW)
        self.assertEqual(table.descriptors[2].alias_of_tensor_id, 1)
        self.assertEqual(table.descriptors[2].data_offset, 24)
        self.assertEqual(table.descriptors[3].data_offset, 0)
        rebuilt = plan_tensor_arena_from_descriptors(
            table.descriptors,
            bucket_frames=graph.bucket_frames,
            operator_count=len(graph.operators),
        )
        self.assertEqual(rebuilt, arena)

    def test_rewritten_execution_plan_round_trips(self) -> None:
        graph = rewrite_dense_concats_as_slabs(make_dense_graph()).graph
        bundle = RuntimeBundle((graph,))
        weights = plan_weight_blob(bundle)
        arena = plan_tensor_arena(graph)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan_3.bin"
            write_execution_plan(graph, weights, path, arena_layout=arena)
            loaded = read_execution_plan(path)

        verify_plan(loaded, graph, weights, arena_layout=arena)
        self.assertEqual(loaded.header.operator_count, 4)
        self.assertEqual(
            sum(
                descriptor.storage_type == TensorStorageType.VIEW
                for descriptor in loaded.tensors
            ),
            4,
        )


if __name__ == "__main__":
    unittest.main()
