from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts" / "4_profill" / "optimization"
    / "13_build_conv_hybrid_plan.py"
)
SPEC = importlib.util.spec_from_file_location("conv_hybrid_plan", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PLAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAN
SPEC.loader.exec_module(PLAN)


def comparison(operator_id: int, shape: list[int], mode: str, ms: float) -> dict:
    return {
        "schema_version": 1,
        "all_output_hashes_bitwise_identical": True,
        "cases": [
            {
                "case_name": f"op_{operator_id}",
                "operator_id": operator_id,
                "kernel_name": "qlinear_conv_o4i4_neon",
                "weight_shape": shape,
                "bitwise_output_hashes": True,
                "candidate_mean_ms": ms,
                "candidate_p50_ms": ms,
                "candidate_p95_ms": ms,
            }
        ],
    }


class ConvHybridPlanTests(unittest.TestCase):
    def test_selects_fastest_only_when_margin_exceeds_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qconv = root / "qconv"
            fused = root / "fused"
            qconv.mkdir()
            fused.mkdir()
            for mode, ms in {
                "mac_fixed": 10.0,
                "v4": 8.0,
                "v5": 7.5,
            }.items():
                (qconv / f"{mode}.json").write_text(
                    json.dumps({"configuration": {"bucket_frames": 298}}),
                    encoding="utf-8",
                )
                (qconv / f"{mode}_comparison.json").write_text(
                    json.dumps(comparison(2, [32, 32, 3, 3], mode, ms)),
                    encoding="utf-8",
                )
            for mode, ms in {
                "combined_fixed": 10.0,
                "combined_hybrid": 8.0,
                "combined_v5": 7.96,
            }.items():
                (fused / f"{mode}.json").write_text(
                    json.dumps({"configuration": {"bucket_frames": 298}}),
                    encoding="utf-8",
                )
                (fused / f"{mode}_comparison.json").write_text(
                    json.dumps(comparison(10, [64, 128, 1], mode, ms)),
                    encoding="utf-8",
                )
            plan = PLAN.build_plan(
                qconv_dir=qconv,
                fused_dir=fused,
                min_margin_pct=1.0,
                bucket_frames=298,
            )
            self.assertEqual(plan["bucket_frames"], 298)
            qconv_row = plan["families"][0]["operators"][0]
            fused_row = plan["families"][1]["operators"][0]
            self.assertEqual(qconv_row["selected"], "v5")
            self.assertEqual(fused_row["selected"], "combined_hybrid")
            self.assertEqual(
                fused_row["selection_reason"], "incumbent_within_margin"
            )


if __name__ == "__main__":
    unittest.main()
