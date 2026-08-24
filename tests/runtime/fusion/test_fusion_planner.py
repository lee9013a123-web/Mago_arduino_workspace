from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "python"))

from runtime_bundle_exporter.format.binary_format_schema import (  # noqa: E402
    OperatorCode,
    TensorDType,
    TensorStorageType,
)
from runtime_bundle_exporter.planner.fusion_planner import (  # noqa: E402
    FUSION_QDQ_ELEMENTWISE_KERNEL_ID,
    FusionConfig,
    rewrite_operator_fusions,
)
from runtime_bundle_exporter.planner.tensor_arena_planner import (  # noqa: E402
    plan_tensor_arena,
)
from runtime_bundle_exporter.runtime_ir import (  # noqa: E402
    RuntimeGraph,
    RuntimeInitializer,
    RuntimeOperator,
    RuntimeTensor,
)
from runtime_bundle_exporter.writer.execution_plan_writer import (  # noqa: E402
    read_execution_plan,
)


def _tensor(
    tensor_id: int,
    name: str,
    dtype: TensorDType,
    shape: tuple[int, ...],
    storage: TensorStorageType,
    producer: int | None,
    consumers: tuple[int, ...],
) -> RuntimeTensor:
    element_size = {TensorDType.FLOAT32: 4, TensorDType.UINT8: 1}[dtype]
    strides = []
    running = element_size
    for dimension in reversed(shape):
        strides.append(running)
        running *= dimension
    return RuntimeTensor(
        name=name,
        tensor_id=tensor_id,
        dtype=dtype,
        shape=shape,
        strides=tuple(reversed(strides)),
        byte_size=running,
        storage_type=storage,
        producer=producer,
        consumers=consumers,
    )


def _qdq_graph() -> RuntimeGraph:
    tensors = (
        _tensor(0, "q", TensorDType.UINT8, (1, 2), TensorStorageType.INPUT, None, (0,)),
        _tensor(1, "old_scale", TensorDType.FLOAT32, (), TensorStorageType.CONSTANT, None, (0,)),
        _tensor(2, "old_zero", TensorDType.UINT8, (), TensorStorageType.CONSTANT, None, (0,)),
        _tensor(3, "other", TensorDType.FLOAT32, (1, 2), TensorStorageType.INPUT, None, (1,)),
        _tensor(4, "dq", TensorDType.FLOAT32, (1, 2), TensorStorageType.ACTIVATION, 0, (1,)),
        _tensor(5, "sum", TensorDType.FLOAT32, (1, 2), TensorStorageType.ACTIVATION, 1, (2,)),
        _tensor(6, "new_scale", TensorDType.FLOAT32, (), TensorStorageType.CONSTANT, None, (2,)),
        _tensor(7, "new_zero", TensorDType.UINT8, (), TensorStorageType.CONSTANT, None, (2,)),
        _tensor(8, "output", TensorDType.UINT8, (1, 2), TensorStorageType.OUTPUT, 2, ()),
    )
    operators = (
        RuntimeOperator("dq", 0, OperatorCode.DEQUANTIZE_LINEAR, (0, 1, 2), (4,)),
        RuntimeOperator("add", 1, OperatorCode.ADD, (4, 3), (5,)),
        RuntimeOperator("quant", 2, OperatorCode.QUANTIZE_LINEAR, (5, 6, 7), (8,)),
    )
    initializers = (
        RuntimeInitializer("old_scale", 1, TensorDType.FLOAT32, (), struct.pack("<f", 0.5)),
        RuntimeInitializer("old_zero", 2, TensorDType.UINT8, (), bytes((128,))),
        RuntimeInitializer("new_scale", 6, TensorDType.FLOAT32, (), struct.pack("<f", 0.25)),
        RuntimeInitializer("new_zero", 7, TensorDType.UINT8, (), bytes((127,))),
    )
    return RuntimeGraph(
        tensors=tensors,
        operators=operators,
        initializers=initializers,
        input_tensor_ids=(0, 3),
        output_tensor_ids=(8,),
        name="qdq-test",
        bucket_frames=98,
    )


class FusionPlannerTests(unittest.TestCase):
    def test_qdq_elementwise_is_independently_fused(self) -> None:
        result = rewrite_operator_fusions(
            _qdq_graph(),
            config=FusionConfig(
                conv_bias_act=False,
                bn_relu_quant=False,
                pool_cam=False,
                qdq_elementwise=True,
                stats_pooling=False,
            ),
            default_kernel_id=1,
        )
        self.assertEqual(len(result.graph.operators), 1)
        self.assertEqual(result.graph.operators[0].opcode, OperatorCode.ADD)
        self.assertEqual(result.kernel_ids[0], FUSION_QDQ_ELEMENTWISE_KERNEL_ID)
        self.assertEqual(result.eliminated_tensor_count, 2)

    def test_disabled_pass_is_identity(self) -> None:
        graph = _qdq_graph()
        result = rewrite_operator_fusions(
            graph,
            config=FusionConfig(False, False, False, False, False),
            default_kernel_id=1,
        )
        self.assertEqual(len(result.graph.operators), len(graph.operators))
        self.assertEqual(len(result.graph.tensors), len(graph.tensors))
        self.assertEqual(set(result.kernel_ids.values()), {1})


class CanonicalE7PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cache_bundle = (
            ROOT / "runs/runtime/kernel_optimization/cache_packed/bundle"
        )
        cls.e7_bundle = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
        if not (cls.cache_bundle / "manifest.json").is_file():
            raise unittest.SkipTest("cache-packed bundle is not present")

    def _source_graph(self) -> RuntimeGraph:
        script = ROOT / "scripts/3_runtime/12_export_cache_packed_bundle.py"
        spec = importlib.util.spec_from_file_location("cache_export_test", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = json.loads(
            (self.cache_bundle / "manifest.json").read_text(encoding="utf-8")
        )
        loaded = read_execution_plan(
            self.cache_bundle / "execution_plans/plan_98.bin"
        )
        return module._runtime_graph_from_plan(
            loaded,
            (self.cache_bundle / "weights.bin").read_bytes(),
            module._weight_records_by_offset(manifest),
        )

    def test_all_fusions_rebuild_canonical_memory_plan(self) -> None:
        result = rewrite_operator_fusions(self._source_graph())
        arena = plan_tensor_arena(result.graph)
        self.assertEqual((result.original_operator_count, len(result.graph.operators)), (1278, 832))
        self.assertEqual((result.original_tensor_count, len(result.graph.tensors)), (3719, 3272))
        self.assertEqual(arena.arena_size, 1_505_280)
        self.assertEqual(result.maximum_scratch_bytes, 125_440)
        self.assertEqual(
            result.to_dict()["family_counts"],
            {
                "FUSION_CONV_BIAS_ACT": 167,
                "FUSION_BN_RELU_QUANT": 55,
                "FUSION_POOL_CAM": 52,
                "FUSION_STATS_POOLING": 1,
            },
        )

    def test_serialized_e7_plan_uses_mixed_kernel_ids(self) -> None:
        plan_path = self.e7_bundle / "execution_plans/plan_98.bin"
        if not plan_path.is_file():
            self.skipTest("E7 bundle has not been exported")
        loaded = read_execution_plan(plan_path)
        self.assertEqual(loaded.header.operator_count, 832)
        self.assertEqual(
            {item.kernel_id for item in loaded.operators},
            {1, 2, 3, 4, 5},
        )


if __name__ == "__main__":
    unittest.main()
