from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "3_runtime" / "10_export_dense_slab_bundle.py"
SPEC = importlib.util.spec_from_file_location("export_dense_slab", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DenseSlabBundleExportTests(unittest.TestCase):
    def test_existing_arena_bundle_is_rewritten_without_onnx(self) -> None:
        source = ROOT / "runs" / "runtime" / "tensor_arena" / "bundle"
        if not (source / "manifest.json").is_file():
            self.skipTest("existing Tensor Arena bundle is not present")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            MODULE.export_dense_slab_bundle(
                argparse.Namespace(
                    source_bundle=source,
                    output_dir=output,
                    arena_alignment=64,
                    force=False,
                )
            )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (
                    output
                    / "dense_slab_plans"
                    / "dense_slab_summary.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(
            [entry["operator_count"] for entry in manifest["plans"]],
            [1386, 1386, 1386, 1386],
        )
        self.assertEqual(
            manifest["optimization"]["removed_dense_concat_count_per_bucket"],
            52,
        )
        self.assertEqual(len(summary["bucket_results"]), 4)
        self.assertTrue(
            all(
                result["view_tensor_count"] == 104
                for result in summary["bucket_results"]
            )
        )


if __name__ == "__main__":
    unittest.main()
