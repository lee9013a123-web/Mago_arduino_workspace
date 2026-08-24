from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "3_runtime"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from runtime_pipeline_config import (  # noqa: E402
    RuntimePipelineConfigError,
    load_runtime_pipeline_config,
)


class RuntimePipelineConfigTests(unittest.TestCase):
    def _load_document(self, name: str) -> dict:
        path = ROOT / "configs" / "runtime" / name
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_modified(self, document: dict):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return load_runtime_pipeline_config(path, repository_root=ROOT)

    def test_reference_profile_selects_all_buckets(self) -> None:
        config = load_runtime_pipeline_config(
            ROOT / "configs" / "runtime" / "reference.json",
            repository_root=ROOT,
        )
        self.assertEqual(config.buckets, (98, 298, 498, 998))
        self.assertEqual(config.diagnostics.dump_tensor_ids, "all")
        self.assertTrue(config.include_weight_index)
        self.assertNotEqual(
            config.paths.workspace_root,
            (ROOT / "models" / "compiled" / "reference").resolve(),
        )

    def test_qrb2210_profile_uses_a_separate_workspace(self) -> None:
        config = load_runtime_pipeline_config(
            ROOT / "configs" / "runtime" / "qrb2210.json",
            repository_root=ROOT,
        )
        self.assertEqual(config.profile, "qrb2210")
        self.assertFalse(config.include_weight_index)
        self.assertIn("qrb2210", str(config.paths.workspace_root))

    def test_canonical_bundle_cannot_be_used_as_workspace(self) -> None:
        document = self._load_document("reference.json")
        document["paths"]["workspace_root"] = "models/compiled/reference"
        with self.assertRaisesRegex(
            RuntimePipelineConfigError, "workspace_root"
        ):
            self._load_modified(document)

    def test_selective_dump_is_rejected_until_comparator_supports_it(self) -> None:
        document = self._load_document("reference.json")
        document["diagnostics"]["dump_tensor_ids"] = [3718]
        with self.assertRaisesRegex(
            RuntimePipelineConfigError, "dump_tensor_ids='all'"
        ):
            self._load_modified(document)


if __name__ == "__main__":
    unittest.main()
