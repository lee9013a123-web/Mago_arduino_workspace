from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "src/python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from runtime_bundle_exporter.format.binary_format_schema import (  # noqa: E402
    INVALID_DATA_OFFSET,
    INVALID_OPERATOR_INDEX,
    INVALID_QUANTIZATION_INDEX,
    INVALID_TENSOR_ID,
    OPERATOR_INPUT_CAPACITY,
    BackendId,
    OperatorCode,
    OperatorDescriptor,
    PlanHeader,
    TensorDescriptor,
    TensorDType,
    TensorFlags,
    TensorStorageType,
)
from runtime_bundle_exporter.planner.weight_residency_planner import (  # noqa: E402
    build_static_weight_plan,
)
from runtime_bundle_exporter.writer.execution_plan_writer import (  # noqa: E402
    LoadedPlan,
    build_loaded_plan_bytes,
)


def _tensor(
    tensor_id: int, storage: TensorStorageType, byte_size: int,
    *, offset: int = INVALID_DATA_OFFSET,
) -> TensorDescriptor:
    flags = int(TensorFlags.CONTIGUOUS)
    if storage is TensorStorageType.CONSTANT:
        flags |= int(TensorFlags.READ_ONLY | TensorFlags.EXTERNAL)
    return TensorDescriptor(
        tensor_id=tensor_id,
        dtype=TensorDType.UINT8,
        rank=1,
        storage_type=storage,
        flags=flags,
        dimensions=(byte_size, 1, 1, 1),
        byte_strides=(1, 0, 0, 0),
        data_offset=offset,
        logical_byte_size=byte_size,
        storage_span_bytes=byte_size,
        alias_of_tensor_id=INVALID_TENSOR_ID,
        quantization_index=INVALID_QUANTIZATION_INDEX,
        first_use=INVALID_OPERATOR_INDEX,
        last_use=INVALID_OPERATOR_INDEX,
    )


def _operator(operator_id: int, inputs: tuple[int, ...], output: int) -> OperatorDescriptor:
    padded = inputs + (INVALID_TENSOR_ID,) * (
        OPERATOR_INPUT_CAPACITY - len(inputs)
    )
    return OperatorDescriptor(
        operator_id=operator_id,
        opcode=OperatorCode.ADD,
        input_count=len(inputs),
        output_count=1,
        input_tensor_ids=padded,
        output_tensor_ids=(output,),
        attribute_offset=0,
        attribute_size=0,
        backend_id=BackendId.AUTO,
        kernel_id=0,
    )


class WeightResidencyPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        tensors = (
            _tensor(0, TensorStorageType.INPUT, 4),
            _tensor(1, TensorStorageType.CONSTANT, 64, offset=128),
            _tensor(2, TensorStorageType.CONSTANT, 4, offset=16),
            _tensor(3, TensorStorageType.CONSTANT, 8, offset=240),
            _tensor(4, TensorStorageType.ACTIVATION, 4),
            _tensor(5, TensorStorageType.OUTPUT, 4),
        )
        operators = (
            _operator(0, (0, 2), 4),
            _operator(1, (4, 1), 5),
        )
        header = PlanHeader.create(
            bucket_frames=98,
            tensor_count=len(tensors),
            operator_count=len(operators),
            tensor_table_offset=80,
            operator_table_offset=80 + len(tensors) * 80,
            attribute_section_offset=(
                80 + len(tensors) * 80 + len(operators) * 64
            ),
            payload=b"",
        )
        self.loaded = LoadedPlan(
            header=header,
            tensors=tensors,
            operators=operators,
            attribute_section=b"",
        )
        self.source_plan = build_loaded_plan_bytes(self.loaded)
        source_weights = bytearray(256)
        source_weights[16:20] = b"scal"
        source_weights[128:192] = bytes(range(64))
        source_weights[240:248] = b"unused!!"
        self.source_weights = bytes(source_weights)

    def test_first_use_order_is_static_and_direct(self) -> None:
        result = build_static_weight_plan(
            self.loaded, self.source_plan, self.source_weights
        )
        by_id = {entry.tensor_id: entry for entry in result.entries}
        self.assertEqual([entry.tensor_id for entry in result.entries], [2, 1, 3])
        self.assertEqual(by_id[2].destination_offset, 0)
        self.assertEqual(by_id[1].destination_offset, 64)
        self.assertEqual(by_id[3].destination_offset, 128)
        self.assertEqual(result.weight_bytes[0:4], b"scal")
        self.assertEqual(result.weight_bytes[64:128], bytes(range(64)))
        self.assertEqual(result.weight_bytes[128:136], b"unused!!")
        self.assertEqual(result.used_constant_count, 2)
        self.assertEqual(
            result.to_dict()["runtime_contract"]["inference_weight_moves"], 0
        )

    def test_only_constant_offsets_and_plan_checksum_change(self) -> None:
        result = build_static_weight_plan(
            self.loaded, self.source_plan, self.source_weights
        )
        from runtime_bundle_exporter.writer.execution_plan_writer import (  # noqa: E402
            read_execution_plan,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.bin"
            path.write_bytes(result.plan_bytes)
            output = read_execution_plan(path)
        self.assertEqual(output.operators, self.loaded.operators)
        self.assertEqual(output.attribute_section, self.loaded.attribute_section)
        for source, remapped in zip(self.loaded.tensors, output.tensors):
            if source.storage_type == TensorStorageType.CONSTANT:
                self.assertEqual(source, replace(
                    remapped, data_offset=source.data_offset
                ))
            else:
                self.assertEqual(source, remapped)
        self.assertNotEqual(
            hashlib.sha256(result.plan_bytes).digest(),
            hashlib.sha256(self.source_plan).digest(),
        )


if __name__ == "__main__":
    unittest.main()
