from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
PYTHON_SRC = ROOT / "src/python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from runtime_model_package.format import (  # noqa: E402
    ModelPackageError,
    ModelPackageSection,
    ModelPackageSectionType,
    build_model_package,
    canonical_json_bytes,
    read_model_package,
)
from runtime_model_package.packager import build_final_98_package  # noqa: E402


def sections() -> list[ModelPackageSection]:
    return [
        ModelPackageSection(ModelPackageSectionType.EXECUTION_PLAN, b"plan"),
        ModelPackageSection(ModelPackageSectionType.PACKED_WEIGHTS, b"weights"),
        ModelPackageSection(
            ModelPackageSectionType.MODEL_METADATA_JSON,
            canonical_json_bytes({"schema_version": 1}),
        ),
        ModelPackageSection(
            ModelPackageSectionType.FRONTEND_CONTRACT_JSON,
            canonical_json_bytes({"schema_version": 1}),
        ),
        ModelPackageSection(
            ModelPackageSectionType.POSTPROCESS_CONTRACT_JSON,
            canonical_json_bytes({"schema_version": 1}),
        ),
    ]


class ModelPackageTests(unittest.TestCase):
    def test_round_trip_and_json_contracts(self) -> None:
        payload = build_model_package(bucket_frames=98, sections=sections())
        package = read_model_package(payload)
        self.assertEqual(package.bucket_frames, 98)
        self.assertEqual(
            package.sections[ModelPackageSectionType.PACKED_WEIGHTS], b"weights"
        )
        self.assertEqual(
            package.json_section(ModelPackageSectionType.MODEL_METADATA_JSON),
            {"schema_version": 1},
        )

    def test_corruption_is_rejected(self) -> None:
        payload = bytearray(build_model_package(bucket_frames=98, sections=sections()))
        payload[-1] ^= 0x80
        with self.assertRaises(ModelPackageError):
            read_model_package(bytes(payload))

    def test_missing_required_section_is_rejected(self) -> None:
        with self.assertRaises(ModelPackageError):
            build_model_package(bucket_frames=98, sections=sections()[:-1])

    def test_empty_section_is_rejected(self) -> None:
        values = sections()
        values[0] = ModelPackageSection(
            ModelPackageSectionType.EXECUTION_PLAN, b""
        )
        with self.assertRaises(ModelPackageError):
            build_model_package(bucket_frames=98, sections=values)

    def test_final_98_bundle_contract(self) -> None:
        bundle = ROOT / "runs/runtime/kernel_optimization/e7/bundle"
        required = (
            bundle / "execution_plans/plan_98.bin",
            bundle / "weights.bin",
            bundle / "manifest.json",
        )
        if not all(path.is_file() for path in required):
            self.skipTest("generated Final-98 bundle is not available")
        payload, report = build_final_98_package(
            plan_path=bundle / "execution_plans/plan_98.bin",
            weights_path=bundle / "weights.bin",
            source_manifest_path=bundle / "manifest.json",
        )
        package = read_model_package(payload)
        metadata = package.json_section(
            ModelPackageSectionType.MODEL_METADATA_JSON
        )
        postprocess = package.json_section(
            ModelPackageSectionType.POSTPROCESS_CONTRACT_JSON
        )
        self.assertEqual(metadata["input"]["shape"], [1, 98, 80])
        self.assertEqual(metadata["output"]["shape"], [1, 192])
        self.assertIsNone(postprocess["decision_threshold"])
        self.assertFalse(report["contracts"]["threshold_calibrated"])


if __name__ == "__main__":
    unittest.main()
