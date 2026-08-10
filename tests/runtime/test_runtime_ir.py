from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import struct
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from runtime_bundle_exporter.binary_format_schema import (  # noqa: E402
    OperatorCode,
    TensorDType,
    TensorStorageType,
)
from runtime_bundle_exporter.runtime_ir import (  # noqa: E402
    RuntimeBundle,
    RuntimeGraph,
    RuntimeInitializer,
    RuntimeIRError,
    RuntimeOperator,
    RuntimeTensor,
)


def make_graph(*, bucket_frames: int = 298) -> RuntimeGraph:
    tensors = (
        RuntimeTensor(
            name="feature",
            tensor_id=0,
            dtype=TensorDType.FLOAT32,
            shape=(1, 2),
            strides=(8, 4),
            byte_size=8,
            storage_type=TensorStorageType.INPUT,
            producer=None,
            consumers=(0,),
        ),
        RuntimeTensor(
            name="bias",
            tensor_id=1,
            dtype=TensorDType.FLOAT32,
            shape=(2,),
            strides=(4,),
            byte_size=8,
            storage_type=TensorStorageType.CONSTANT,
            producer=None,
            consumers=(0,),
        ),
        RuntimeTensor(
            name="embedding",
            tensor_id=2,
            dtype=TensorDType.FLOAT32,
            shape=(1, 2),
            strides=(8, 4),
            byte_size=8,
            storage_type=TensorStorageType.OUTPUT,
            producer=0,
            consumers=(),
        ),
    )
    operator = RuntimeOperator(
        name="add_bias",
        operator_id=0,
        opcode=OperatorCode.ADD,
        input_tensor_ids=(0, 1),
        output_tensor_ids=(2,),
        attributes={"axis": -1, "broadcast_shape": [1, 2]},
    )
    initializer = RuntimeInitializer(
        name="bias",
        tensor_id=1,
        dtype=TensorDType.FLOAT32,
        shape=(2,),
        raw_data=struct.pack("<2f", 0.25, -0.5),
    )
    return RuntimeGraph(
        name="campp_3s",
        bucket_frames=bucket_frames,
        tensors=tensors,
        operators=(operator,),
        initializers=(initializer,),
    )


class RuntimeValueTests(unittest.TestCase):
    def test_tensor_rejects_inconsistent_logical_size(self) -> None:
        with self.assertRaisesRegex(RuntimeIRError, "byte_size"):
            RuntimeTensor(
                name="feature",
                tensor_id=0,
                dtype=TensorDType.FLOAT32,
                shape=(1, 2),
                strides=(8, 4),
                byte_size=7,
                storage_type=TensorStorageType.INPUT,
                producer=None,
                consumers=(0,),
            )

    def test_operator_attributes_are_normalized_and_immutable(self) -> None:
        operator = make_graph().operator(0)
        self.assertEqual(operator.attributes["broadcast_shape"], (1, 2))
        with self.assertRaises(TypeError):
            operator.attributes["axis"] = 0  # type: ignore[index]

    def test_operator_enforces_binary_descriptor_capacity(self) -> None:
        with self.assertRaisesRegex(RuntimeIRError, "9-input capacity"):
            RuntimeOperator(
                name="too_many_inputs",
                operator_id=0,
                opcode=OperatorCode.ADD,
                input_tensor_ids=tuple(range(10)),
                output_tensor_ids=(10,),
            )

    def test_initializer_copies_a_mutable_byte_buffer(self) -> None:
        source = bytearray(struct.pack("<2f", 1.0, 2.0))
        initializer = RuntimeInitializer(
            name="bias",
            tensor_id=1,
            dtype=TensorDType.FLOAT32,
            shape=(2,),
            raw_data=source,
        )
        source[0] ^= 0xFF
        self.assertEqual(initializer.raw_data, struct.pack("<2f", 1.0, 2.0))


class RuntimeGraphTests(unittest.TestCase):
    def test_builds_graph_and_indexes_stable_ids(self) -> None:
        graph = make_graph()
        self.assertEqual(graph.input_tensor_ids, (0,))
        self.assertEqual(graph.output_tensor_ids, (2,))
        self.assertEqual(graph.tensor(2).producer, 0)
        self.assertEqual(graph.operator(0).input_tensor_ids, (0, 1))
        self.assertEqual(graph.initializer(1).name, "bias")

    def test_rejects_consumer_link_that_disagrees_with_operators(self) -> None:
        graph = make_graph()
        invalid_feature = replace(graph.tensor(0), consumers=())
        with self.assertRaisesRegex(RuntimeIRError, "consumers"):
            RuntimeGraph(
                tensors=(invalid_feature,) + graph.tensors[1:],
                operators=graph.operators,
                initializers=graph.initializers,
                name=graph.name,
                bucket_frames=graph.bucket_frames,
            )

    def test_rejects_operator_that_consumes_a_future_tensor(self) -> None:
        graph = make_graph()
        future_tensor = RuntimeTensor(
            name="future",
            tensor_id=3,
            dtype=TensorDType.FLOAT32,
            shape=(1, 2),
            strides=(8, 4),
            byte_size=8,
            storage_type=TensorStorageType.ACTIVATION,
            producer=1,
            consumers=(0,),
        )
        early_operator = replace(
            graph.operator(0), input_tensor_ids=(0, 1, 3)
        )
        later_operator = RuntimeOperator(
            name="later",
            operator_id=1,
            opcode=OperatorCode.RELU,
            input_tensor_ids=(2,),
            output_tensor_ids=(3,),
        )
        with self.assertRaisesRegex(RuntimeIRError, "before it is produced"):
            RuntimeGraph(
                tensors=graph.tensors + (future_tensor,),
                operators=(early_operator, later_operator),
                initializers=graph.initializers,
                name=graph.name,
                bucket_frames=graph.bucket_frames,
            )

    def test_rejects_constant_without_initializer(self) -> None:
        graph = make_graph()
        with self.assertRaisesRegex(RuntimeIRError, "CONSTANT"):
            RuntimeGraph(
                tensors=graph.tensors,
                operators=graph.operators,
                initializers=(),
                name=graph.name,
                bucket_frames=graph.bucket_frames,
            )


class RuntimeBundleTests(unittest.TestCase):
    def test_multiple_buckets_share_initializer_bytes(self) -> None:
        first = make_graph(bucket_frames=98)
        second = replace(first, name="campp_3s", bucket_frames=298)
        bundle = RuntimeBundle((first, second))
        self.assertEqual(bundle.graph_for_bucket(298), second)
        self.assertEqual(bundle.initializers, first.initializers)

    def test_rejects_different_initializer_data_between_buckets(self) -> None:
        first = make_graph(bucket_frames=98)
        changed = replace(
            first.initializers[0], raw_data=struct.pack("<2f", 9.0, 10.0)
        )
        second = RuntimeGraph(
            tensors=first.tensors,
            operators=first.operators,
            initializers=(changed,),
            name="campp_3s",
            bucket_frames=298,
        )
        with self.assertRaisesRegex(RuntimeIRError, "identical initializers"):
            RuntimeBundle((first, second))


if __name__ == "__main__":
    unittest.main()
