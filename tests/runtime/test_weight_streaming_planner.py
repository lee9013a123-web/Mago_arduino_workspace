from __future__ import annotations

import struct
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "src/python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from runtime_bundle_exporter.planner.weight_streaming_planner import (  # noqa: E402
    WEIGHT_SCHEDULE_HEADER_SIZE,
    WEIGHT_SCHEDULE_MAGIC,
    WEIGHT_SCHEDULE_RECORD_SIZE,
    build_windowed_weight_plan,
)
from tests.runtime import test_weight_residency_planner as residency_fixture  # noqa: E402


class WeightStreamingPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = residency_fixture.WeightResidencyPlannerTests(
            methodName="runTest"
        )
        fixture.setUp()
        self.loaded = fixture.loaded
        self.source_plan = fixture.source_plan
        self.source_weights = fixture.source_weights

    def test_blocks_are_page_aligned_and_payloads_are_identical(self) -> None:
        result = build_windowed_weight_plan(
            self.loaded,
            self.source_plan,
            self.source_weights,
            page_size=64,
            target_block_bytes=64,
        )
        self.assertEqual(len(result.blocks), 2)
        for block in result.blocks:
            self.assertEqual(block.file_offset % 64, 0)
            self.assertEqual(block.byte_size % 64, 0)
        by_id = {entry.tensor_id: entry for entry in result.entries}
        for tensor_id in (1, 2, 3):
            entry = by_id[tensor_id]
            expected = self.source_weights[
                entry.source_offset:entry.source_offset + entry.byte_size
            ]
            actual = result.weight_bytes[
                entry.destination_offset:
                entry.destination_offset + entry.byte_size
            ]
            self.assertEqual(actual, expected)
        self.assertEqual(result.blocks[0].first_operator, 0)
        self.assertEqual(result.blocks[0].last_operator, 0)
        self.assertEqual(result.blocks[1].first_operator, 1)

    def test_schedule_binary_contract(self) -> None:
        result = build_windowed_weight_plan(
            self.loaded,
            self.source_plan,
            self.source_weights,
            page_size=64,
            target_block_bytes=64,
        )
        header = struct.unpack(
            "<8s6I4Q", result.schedule_bytes[:WEIGHT_SCHEDULE_HEADER_SIZE]
        )
        self.assertEqual(header[0], WEIGHT_SCHEDULE_MAGIC)
        self.assertEqual(header[3], 98)
        self.assertEqual(header[4], 64)
        self.assertEqual(header[5], len(result.blocks))
        self.assertEqual(
            len(result.schedule_bytes),
            WEIGHT_SCHEDULE_HEADER_SIZE
            + len(result.blocks) * WEIGHT_SCHEDULE_RECORD_SIZE,
        )

    def test_block_lifetimes_cover_every_used_constant_once(self) -> None:
        result = build_windowed_weight_plan(
            self.loaded,
            self.source_plan,
            self.source_weights,
            page_size=64,
            target_block_bytes=64,
        )
        used_entries = {entry.tensor_id: entry for entry in result.entries
                        if entry.used}
        scheduled_ids = [
            tensor_id
            for block in result.blocks
            for tensor_id in block.tensor_ids
        ]
        self.assertEqual(sorted(scheduled_ids), sorted(used_entries))
        self.assertEqual(len(scheduled_ids), len(set(scheduled_ids)))
        for block_id, block in enumerate(result.blocks):
            entries = [used_entries[tensor_id] for tensor_id in block.tensor_ids]
            self.assertEqual(
                block.first_operator,
                min(entry.first_use_operator for entry in entries),
            )
            self.assertEqual(
                block.last_operator,
                max(entry.last_use_operator for entry in entries),
            )
            expected_prefetch = (
                0 if block_id == 0
                else result.blocks[block_id - 1].first_operator
            )
            self.assertEqual(block.prefetch_operator, expected_prefetch)


if __name__ == "__main__":
    unittest.main()
