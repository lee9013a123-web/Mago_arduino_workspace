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
from runtime_model_package.packager import (  # noqa: E402
    SUPPORTED_BUCKET_FRAMES,
    _source_manifest_hash,
    build_final_98_package,
    build_final_bucket_package,
)


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
    def test_static_weight_manifest_hashes_are_bucket_scoped(self) -> None:
        manifest = {
            "bucket_frames": 98,
            "strategy": "offline_first_use_static_blob",
            "output": {
                "plan_sha256": "plan-hash",
                "weights_sha256": "weights-hash",
            },
        }
        self.assertEqual(
            _source_manifest_hash(manifest, "plan", 98), "plan-hash"
        )
        self.assertEqual(
            _source_manifest_hash(manifest, "weights", 98), "weights-hash"
        )
        self.assertIsNone(_source_manifest_hash(manifest, "plan", 298))

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

    def test_final_98_static_weight_plan_contract(self) -> None:
        root = (
            ROOT
            / "runs/models/campplus/final_v3/weight_residency/98"
        )
        required = (
            root / "plan_98.bin",
            root / "weights_98.bin",
            root / "weight_plan_98.json",
        )
        if not all(path.is_file() for path in required):
            self.skipTest("generated Final-98 static weight plan is not available")
        payload, report = build_final_98_package(
            plan_path=required[0],
            weights_path=required[1],
            source_manifest_path=required[2],
        )
        package = read_model_package(payload)
        metadata = package.json_section(
            ModelPackageSectionType.MODEL_METADATA_JSON
        )
        self.assertEqual(
            metadata["source"]["source_manifest_kind"],
            "bucket_static_weight_plan",
        )
        self.assertEqual(
            report["source_manifest_kind"], "bucket_static_weight_plan"
        )

    def test_all_static_bucket_contracts(self) -> None:
        base = ROOT / "runs/models/campplus/final_v3/weight_residency"
        required = [
            base / str(bucket) / name.format(bucket=bucket)
            for bucket in SUPPORTED_BUCKET_FRAMES
            for name in (
                "plan_{bucket}.bin",
                "weights_{bucket}.bin",
                "weight_plan_{bucket}.json",
            )
        ]
        if not all(path.is_file() for path in required):
            self.skipTest("generated multibucket weight plans are unavailable")
        for bucket in SUPPORTED_BUCKET_FRAMES:
            with self.subTest(bucket=bucket):
                root = base / str(bucket)
                payload, report = build_final_bucket_package(
                    bucket_frames=bucket,
                    plan_path=root / f"plan_{bucket}.bin",
                    weights_path=root / f"weights_{bucket}.bin",
                    source_manifest_path=root / f"weight_plan_{bucket}.json",
                )
                package = read_model_package(payload)
                metadata = package.json_section(
                    ModelPackageSectionType.MODEL_METADATA_JSON
                )
                self.assertEqual(package.bucket_frames, bucket)
                self.assertEqual(metadata["input"]["shape"], [1, bucket, 80])
                self.assertEqual(report["bucket_frames"], bucket)


if __name__ == "__main__":
    unittest.main()
