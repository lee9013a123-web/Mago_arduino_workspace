import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "scripts/4_profill/optimization/10_benchmark_fused_dequant_relu_quant.py"
)
SPEC = importlib.util.spec_from_file_location("fused_dqrq_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def _operator(operator_id: int, shape: list[int], mean_ms: float) -> dict:
    return {
        "operator_id": operator_id,
        "kernel_id": 4,
        "kernel_name": "fused_dequant_relu_quant",
        "operator_type": "DEQUANTIZE_LINEAR",
        "mean_ms": mean_ms,
        "end_to_end_share_pct": 1.0,
        "input_tensors": [{"shape": shape}],
    }


class FusedDqRqBenchmarkTests(unittest.TestCase):
    def test_selects_slowest_representative_for_each_shape(self) -> None:
        cases = BENCHMARK.select_cases(
            [
                _operator(1, [1, 32, 40, 98], 4.0),
                _operator(2, [1, 32, 40, 98], 7.0),
                _operator(3, [1, 64, 49], 2.0),
            ]
        )
        self.assertEqual([case["operator_id"] for case in cases], [2, 3])

    def test_all_operator_scope_keeps_every_case(self) -> None:
        cases = BENCHMARK.select_cases(
            [
                _operator(1, [1, 32, 40, 98], 4.0),
                _operator(2, [1, 32, 40, 98], 7.0),
            ],
            all_operators=True,
        )
        self.assertEqual(len(cases), 2)

    def test_command_forwards_candidate_mode(self) -> None:
        command = BENCHMARK._command(
            Path("microbench"), plan=Path("plan.bin"),
            weights=Path("weights.bin"), feature=Path("input.f32"),
            operator_id=50, warmup=5, repeat=20, mode="neon",
        )
        index = command.index("--fused-dqrq-candidate")
        self.assertEqual(command[index + 1], "neon")

    def test_gate_requires_bitwise_and_speedup(self) -> None:
        case = {
            "modes": {
                "baseline": {"mean_ms": 10.0},
                "scalar": {"mean_ms": 4.0},
                "neon": {"mean_ms": 1.0},
            },
            "bitwise_by_mode": {"scalar": True, "neon": True},
        }
        result = BENCHMARK.build_result(
            [case], warmup=5, repeat=20, all_operators=False,
            elapsed_seconds=1.0,
        )
        self.assertTrue(result["candidate_microbench_gate_passed"])
        self.assertEqual(case["winner"], "neon")
        self.assertEqual(case["winner_speedup_ratio"], 10.0)


if __name__ == "__main__":
    unittest.main()
