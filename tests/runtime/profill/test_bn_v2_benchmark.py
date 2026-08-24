import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/4_profill/optimization/11_benchmark_bn_v2.py"
SPEC = importlib.util.spec_from_file_location("bn_v2_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def _operator(operator_id: int, channels: int, mean_ms: float) -> dict:
    return {
        "operator_id": operator_id,
        "kernel_id": 2,
        "kernel_name": "fused_bn_relu_quant_combined",
        "operator_type": "BATCH_NORMALIZATION",
        "mean_ms": mean_ms,
        "end_to_end_share_pct": 1.0,
        "output_tensors": [{"shape": [1, channels, 49]}],
    }


class BnV2BenchmarkTests(unittest.TestCase):
    def test_selects_small_middle_and_large_channel_cases(self) -> None:
        cases = BENCHMARK.select_cases(
            [
                _operator(1, 128, 1.0),
                _operator(2, 256, 2.0),
                _operator(3, 512, 3.0),
                _operator(4, 1024, 4.0),
            ]
        )
        self.assertEqual(
            [case["operator_id"] for case in cases], [1, 3, 4]
        )

    def test_all_operator_scope_keeps_every_case(self) -> None:
        cases = BENCHMARK.select_cases(
            [_operator(1, 128, 1.0), _operator(2, 256, 2.0)],
            all_operators=True,
        )
        self.assertEqual(len(cases), 2)

    def test_command_forwards_bn_candidate_mode(self) -> None:
        command = BENCHMARK._command(
            Path("microbench"),
            plan=Path("plan.bin"),
            weights=Path("weights.bin"),
            feature=Path("input.f32"),
            operator_id=824,
            warmup=5,
            repeat=20,
            mode="v2_exact16",
        )
        index = command.index("--bn-candidate")
        self.assertEqual(command[index + 1], "v2_exact16")

    def test_gate_requires_exact_modes_to_be_bitwise_and_faster(self) -> None:
        case = {
            "modes": {
                "combined": {"mean_ms": 10.0},
                "v2_exact16": {"mean_ms": 5.0},
                "v2_spatial2": {"mean_ms": 4.0},
                "v2_prescaled": {"mean_ms": 2.0},
            },
            "bitwise_by_mode": {
                "v2_exact16": True,
                "v2_spatial2": True,
                "v2_prescaled": False,
            },
        }
        result = BENCHMARK.build_result(
            [case],
            warmup=5,
            repeat=20,
            all_operators=False,
            elapsed_seconds=1.0,
        )
        self.assertTrue(result["candidate_microbench_gate_passed"])
        self.assertEqual(case["winner"], "v2_spatial2")
        self.assertEqual(case["winner_speedup_ratio"], 2.5)


if __name__ == "__main__":
    unittest.main()
