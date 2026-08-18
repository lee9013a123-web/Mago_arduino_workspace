import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/4_profill/optimization/07_benchmark_remaining_ops.py"
SPEC = importlib.util.spec_from_file_location("remaining_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


class RemainingBenchmarkTests(unittest.TestCase):
    def test_selects_slowest_representative_per_kernel(self) -> None:
        operators = [
            {
                "operator_id": 1,
                "kernel_id": 1,
                "kernel_name": "add_stride",
                "operator_type": "ADD",
                "mean_ms": 1.0,
                "end_to_end_share_pct": 0.1,
                "input_tensors": [],
                "output_tensors": [],
            },
            {
                "operator_id": 2,
                "kernel_id": 1,
                "kernel_name": "add_stride",
                "operator_type": "ADD",
                "mean_ms": 3.0,
                "end_to_end_share_pct": 0.2,
                "input_tensors": [],
                "output_tensors": [],
            },
        ]
        selected = BENCHMARK.select_representatives(
            operators, ("add_stride",)
        )
        self.assertEqual(selected[0]["operator_id"], 2)

    def test_missing_kernel_is_rejected(self) -> None:
        with self.assertRaises(BENCHMARK.RemainingBenchmarkError):
            BENCHMARK.select_representatives([], ("relu_stride",))

    def test_sample_summary_uses_nanoseconds(self) -> None:
        summary = BENCHMARK.summarize_samples([1_000_000, 2_000_000, 3_000_000])
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["mean_ms"], 2.0)
        self.assertEqual(summary["p50_ms"], 2.0)

    def test_command_forwards_remaining_mode(self) -> None:
        command = BENCHMARK._command(
            Path("microbench"),
            plan=Path("plan.bin"),
            weights=Path("weights.bin"),
            feature=Path("input.f32"),
            operator_id=8,
            warmup=5,
            repeat=20,
            mode="optimized",
        )
        self.assertEqual(
            command[command.index("--remaining-candidate") + 1],
            "optimized",
        )

    def test_gate_requires_bitwise_and_no_regression(self) -> None:
        case = {
            "bitwise_output_hashes": True,
            "baseline": {"mean_ms": 10.0},
            "optimized": {"mean_ms": 5.0},
        }
        result = BENCHMARK.build_result(
            [case], warmup=5, repeat=20, elapsed_seconds=1.0
        )
        self.assertTrue(result["candidate_gate_passed"])
        case["optimized"] = {"mean_ms": 11.0}
        result = BENCHMARK.build_result(
            [case], warmup=5, repeat=20, elapsed_seconds=1.0
        )
        self.assertFalse(result["candidate_gate_passed"])


if __name__ == "__main__":
    unittest.main()
