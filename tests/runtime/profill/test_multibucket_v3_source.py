from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts/4_profill/optimization"
    / "17_generate_multibucket_v3_source.py"
)
SPEC = importlib.util.spec_from_file_location("multibucket_v3_source", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATOR
SPEC.loader.exec_module(GENERATOR)


def measured_plan(bucket: int, offset: int) -> dict:
    return {
        "bucket_frames": bucket,
        "path": Path(f"plan_{bucket}.json"),
        "qconv": {
            "mac_fixed": [3 + offset],
            "v4": [6 + offset],
            "v5": [9 + offset],
        },
        "fused": {
            "combined_fixed": [12 + offset],
            "combined_hybrid": [15 + offset],
            "combined_v5": [18 + offset],
        },
    }


class MultibucketV3SourceTests(unittest.TestCase):
    def test_renders_independent_bucket_tables_and_safe_fallback(self) -> None:
        source = GENERATOR.render_source(
            [measured_plan(98, 0), measured_plan(298, 100)]
        )
        self.assertIn("CAMPP_QCONV_V5_IDS_98", source)
        self.assertIn("CAMPP_QCONV_V5_IDS_298", source)
        self.assertIn("        98u,", source)
        self.assertIn("        298u,", source)
        self.assertIn("if (plan == NULL)", source)
        self.assertIn("campp_conv_layer_hybrid_bucket_plan_count", source)
        self.assertIn("CAMPP_QCONV_CANDIDATE_V4", source)
        self.assertIn(
            "CAMPP_FUSED_QCONV_CANDIDATE_COMBINED_HYBRID", source
        )

    def test_rejects_duplicate_buckets(self) -> None:
        with self.assertRaises(GENERATOR.MultibucketPlanError):
            GENERATOR.render_source(
                [measured_plan(298, 0), measured_plan(298, 100)]
            )


if __name__ == "__main__":
    unittest.main()
