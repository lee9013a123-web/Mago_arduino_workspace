from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
PYTHON_SRC = ROOT / "src/python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

SCRIPT = ROOT / "scripts/5_model/12_run_bucket_model.py"
SPEC = importlib.util.spec_from_file_location("bucket_model_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

from runtime_model_package.format import (  # noqa: E402
    ModelPackageError,
    ModelPackageSection,
    ModelPackageSectionType,
    build_model_package,
    canonical_json_bytes,
)


def _package(bucket: int) -> bytes:
    return build_model_package(
        bucket_frames=bucket,
        sections=[
            ModelPackageSection(
                ModelPackageSectionType.EXECUTION_PLAN, b"plan"
            ),
            ModelPackageSection(
                ModelPackageSectionType.PACKED_WEIGHTS, b"weights"
            ),
            ModelPackageSection(
                ModelPackageSectionType.MODEL_METADATA_JSON,
                canonical_json_bytes({
                    "bucket_frames": bucket,
                    "input": {"shape": [1, bucket, 80]},
                    "output": {"shape": [1, 192]},
                }),
            ),
            ModelPackageSection(
                ModelPackageSectionType.FRONTEND_CONTRACT_JSON,
                canonical_json_bytes({"model_input_frames": bucket}),
            ),
            ModelPackageSection(
                ModelPackageSectionType.POSTPROCESS_CONTRACT_JSON,
                canonical_json_bytes({"embedding_dimension": 192}),
            ),
        ],
    )


class BucketModelSelectorTests(unittest.TestCase):
    def _fixture(self, directory: Path, bucket: int = 298) -> tuple[Path, Path]:
        package = _package(bucket)
        model = directory / f"campp_sv_{bucket}.camppmodel"
        feature = directory / f"feature_{bucket}.f32"
        manifest = directory / "campp_sv_multibucket.json"
        model.write_bytes(package)
        feature.write_bytes(bytes(bucket * 80 * 4))
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "format": "campp-fixed-bucket-model-set-v1",
            "supported_buckets": [bucket],
            "buckets": {
                str(bucket): {
                    "audio_seconds": 3.0,
                    "model": model.name,
                    "sha256": hashlib.sha256(package).hexdigest(),
                }
            },
        }), encoding="utf-8")
        return manifest, feature

    def test_selects_exact_bucket_from_feature_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, feature = self._fixture(root)
            bucket, model, seconds, _ = RUNNER.select_bucket_model(
                manifest_path=manifest, feature_path=feature
            )
            self.assertEqual(bucket, 298)
            self.assertEqual(model.name, "campp_sv_298.camppmodel")
            self.assertEqual(seconds, 3.0)

    def test_rejects_requested_bucket_that_differs_from_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, feature = self._fixture(root)
            with self.assertRaises(ModelPackageError):
                RUNNER.select_bucket_model(
                    manifest_path=manifest,
                    feature_path=feature,
                    requested_bucket=98,
                )

    def test_rejects_non_fbank_payload_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            feature = Path(temporary) / "bad.f32"
            feature.write_bytes(b"not-a-feature")
            with self.assertRaises(ModelPackageError):
                RUNNER.infer_feature_frames(feature)


if __name__ == "__main__":
    unittest.main()
