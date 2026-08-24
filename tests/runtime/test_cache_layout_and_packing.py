from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from runtime_bundle_exporter.planner.cache_layout_planner import (  # noqa: E402
    rewrite_cache_friendly_activations,
)
from runtime_bundle_exporter.planner.tensor_arena_planner import (  # noqa: E402
    plan_tensor_arena,
)
from runtime_bundle_exporter.planner.weight_packing_planner import (  # noqa: E402
    pack_qconv_o4i4,
    unpack_qconv_o4i4,
)
from runtime_bundle_exporter.format.binary_format_schema import (  # noqa: E402
    OperatorCode,
    TensorDType,
    TensorStorageType,
)
from runtime_bundle_exporter.runtime_ir import (  # noqa: E402
    RuntimeGraph,
    RuntimeOperator,
    RuntimeTensor,
)


class CacheLayoutPlannerTests(unittest.TestCase):
    def test_input_transpose_becomes_view_and_output_is_channel_padded(self) -> None:
        graph = RuntimeGraph(
            tensors=(
                RuntimeTensor(
                    "input", 0, TensorDType.FLOAT32, (1, 5, 3),
                    (60, 12, 4), 60, TensorStorageType.INPUT, None, (0,)
                ),
                RuntimeTensor(
                    "transpose", 1, TensorDType.FLOAT32, (1, 3, 5),
                    (60, 20, 4), 60, TensorStorageType.ACTIVATION, 0, (1,)
                ),
                RuntimeTensor(
                    "output", 2, TensorDType.FLOAT32, (1, 3, 5),
                    (60, 20, 4), 60, TensorStorageType.OUTPUT, 1, ()
                ),
            ),
            operators=(
                RuntimeOperator(
                    "transpose", 0, OperatorCode.TRANSPOSE, (0,), (1,),
                    {"perm": (0, 2, 1)},
                ),
                RuntimeOperator("relu", 1, OperatorCode.RELU, (1,), (2,)),
            ),
            initializers=(),
            bucket_frames=5,
        )

        result = rewrite_cache_friendly_activations(graph)
        self.assertEqual(result.removed_transpose_operator_ids, (0,))
        self.assertEqual(len(result.graph.operators), 1)
        transpose = result.graph.tensor(1)
        self.assertEqual(transpose.storage_type, TensorStorageType.VIEW)
        self.assertEqual(transpose.alias_of_tensor_id, 0)
        self.assertEqual(transpose.strides, (60, 4, 12))
        output = result.graph.tensor(2)
        self.assertEqual(output.strides, (80, 4, 16))
        self.assertEqual(output.storage_span_bytes, 80)
        self.assertEqual(result.padded_tensor_count, 1)
        self.assertEqual(plan_tensor_arena(result.graph).arena_size, 128)


class WeightPackingTests(unittest.TestCase):
    def test_o4i4_round_trip_with_channel_tails(self) -> None:
        shape = (5, 3, 1)
        raw = bytes(range(15))
        packed = pack_qconv_o4i4(
            raw, shape, 1, (0, 1, 2, 3, 4), TensorDType.INT8
        )
        self.assertEqual(len(packed), 32)
        self.assertEqual(unpack_qconv_o4i4(packed, shape, 1), raw)

    def test_o4i4_round_trip_for_groups(self) -> None:
        shape = (8, 3, 3)
        raw = bytes((index * 7) & 0x7F for index in range(72))
        packed = pack_qconv_o4i4(
            raw, shape, 2, (0,), TensorDType.UINT8
        )
        self.assertEqual(unpack_qconv_o4i4(packed, shape, 2), raw)


if __name__ == "__main__":
    unittest.main()
