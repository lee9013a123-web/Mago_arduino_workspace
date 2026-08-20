from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT / "results/profiling/e7_98/optimization/conv_hybrid_plan.json"
SOURCE = (
    ROOT / "src/c/profill/optimization/candidates/conv_layer_hybrid"
    / "conv_layer_hybrid_plan.c"
)


def c_ids(source: str, symbol: str) -> set[int]:
    match = re.search(
        rf"{re.escape(symbol)}\[\]\s*=\s*\{{(.*?)\}};",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing C table: {symbol}")
    return {int(value) for value in re.findall(r"(\d+)u", match.group(1))}


class ConvLayerHybridSourceTests(unittest.TestCase):
    def test_embedded_tables_match_measured_hybrid_plan(self) -> None:
        document = json.loads(PLAN.read_text(encoding="utf-8"))
        families = {item["family"]: item for item in document["families"]}
        source = SOURCE.read_text(encoding="utf-8")
        qconv = families["qlinear_conv"]["operators"]
        fused = families["fused_quant_qconv"]["operators"]
        self.assertEqual(
            c_ids(source, "CAMPP_QCONV_MAC_FIXED_IDS"),
            {item["operator_id"] for item in qconv if item["selected"] == "mac_fixed"},
        )
        self.assertEqual(
            c_ids(source, "CAMPP_QCONV_V5_IDS"),
            {item["operator_id"] for item in qconv if item["selected"] == "v5"},
        )
        self.assertEqual(
            c_ids(source, "CAMPP_FUSED_QCONV_FIXED_IDS"),
            {item["operator_id"] for item in fused if item["selected"] == "combined_fixed"},
        )
        self.assertEqual(
            c_ids(source, "CAMPP_FUSED_QCONV_V5_IDS"),
            {item["operator_id"] for item in fused if item["selected"] == "combined_v5"},
        )


if __name__ == "__main__":
    unittest.main()
