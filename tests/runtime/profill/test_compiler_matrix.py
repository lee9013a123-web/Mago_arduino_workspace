from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "4_profill" / "compiler" / "02_run_matrix.py"
SPEC = importlib.util.spec_from_file_location("compiler_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
SPEC.loader.exec_module(MATRIX)


SELECTION = {
    "minimum_p50_improvement_pct": 2.0,
    "maximum_p95_regression_pct": 1.0,
    "maximum_cv_pct": 3.0,
    "maximum_peak_rss_increase_bytes": 1024,
}


def result(
    variant: str,
    *,
    p50_ms: float,
    p95_ms: float,
    cv_pct: float = 1.0,
    peak_rss_bytes: int = 10_000,
    binary_size_bytes: int = 100_000,
    hashes_match: bool = True,
    status: str = "passed",
) -> dict:
    return {
        "variant": variant,
        "status": status,
        "all_output_hashes_match_reference": hashes_match,
        "all_retained_tensor_hashes_match_reference": hashes_match,
        "aggregate": {
            "latency": {
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
                "cv_pct": cv_pct,
            },
            "peak_rss_bytes": peak_rss_bytes,
        },
        "binary": {"size_bytes": binary_size_bytes},
    }


class CompilerMatrixTests(unittest.TestCase):
    def test_quick_config_uses_six_reduced_builds(self) -> None:
        config = MATRIX._load_matrix_config(
            ROOT / "configs" / "runtime" / "compiler_qrb2210_quick.json"
        )
        benchmark = MATRIX.BENCHMARK.load_config(
            config["_benchmark_config_path"], repository_root=ROOT
        )

        self.assertEqual(config["build_scope"], "compiler_quick")
        self.assertEqual(sum(len(stage["candidates"]) for stage in config["stages"]), 6)
        self.assertEqual(len(benchmark.input_ids), 3)
        self.assertEqual(benchmark.warmup, 0)
        self.assertEqual(benchmark.repeat, 1)
        self.assertEqual(benchmark.cold_runs, 0)

    def test_strict_config_targets_final_suite_in_stages(self) -> None:
        config = MATRIX._load_matrix_config(
            ROOT / "configs" / "runtime" / "compiler_qrb2210_strict.json"
        )

        self.assertEqual(config["expected_suite"], "final")
        self.assertEqual(config["stages"][0]["name"], "optimization_level")
        self.assertEqual(config["stages"][-1]["name"], "packaging")
        self.assertEqual(config["build_scope"], "compiler_matrix")
        self.assertNotIn("-ffast-math", config["base_cflags"])

    def test_merge_flags_rejects_non_option_tokens(self) -> None:
        with self.assertRaises(MATRIX.MatrixError):
            MATRIX._merge_flags("-O3", "source.c")

    def test_fast_math_is_kept_in_a_separate_matrix(self) -> None:
        config = MATRIX._load_matrix_config(
            ROOT / "configs" / "runtime" / "compiler_qrb2210_fast_math.json"
        )

        self.assertTrue(config["requires_base_decision"])
        self.assertEqual([stage["name"] for stage in config["stages"]], ["fast_math"])

    def test_latency_stage_selects_material_improvement(self) -> None:
        decision = MATRIX._choose_stage_winner(
            [
                result("control", p50_ms=100.0, p95_ms=105.0),
                result("candidate", p50_ms=95.0, p95_ms=104.0),
            ],
            SELECTION,
            "latency",
        )

        self.assertEqual(decision["winner"], "candidate")

    def test_latency_stage_keeps_control_for_small_change(self) -> None:
        decision = MATRIX._choose_stage_winner(
            [
                result("control", p50_ms=100.0, p95_ms=105.0),
                result("candidate", p50_ms=99.0, p95_ms=104.0),
            ],
            SELECTION,
            "latency",
        )

        self.assertEqual(decision["winner"], "control")

    def test_hash_and_p95_gates_reject_faster_candidates(self) -> None:
        decision = MATRIX._choose_stage_winner(
            [
                result("control", p50_ms=100.0, p95_ms=100.0),
                result(
                    "hash_mismatch",
                    p50_ms=80.0,
                    p95_ms=90.0,
                    hashes_match=False,
                ),
                result("unstable_tail", p50_ms=85.0, p95_ms=102.0),
            ],
            SELECTION,
            "latency",
        )

        self.assertEqual(decision["winner"], "control")
        evaluations = {item["variant"]: item for item in decision["evaluations"]}
        self.assertIn("output_hash_mismatch", evaluations["hash_mismatch"]["reasons"])
        self.assertIn(
            "retained_tensor_hash_mismatch",
            evaluations["hash_mismatch"]["reasons"],
        )
        self.assertIn("p95_regression", evaluations["unstable_tail"]["reasons"])

    def test_size_stage_uses_smallest_eligible_binary(self) -> None:
        decision = MATRIX._choose_stage_winner(
            [
                result(
                    "control", p50_ms=100.0, p95_ms=100.0,
                    binary_size_bytes=100_000,
                ),
                result(
                    "stripped", p50_ms=100.0, p95_ms=100.0,
                    binary_size_bytes=60_000,
                ),
            ],
            SELECTION,
            "size",
        )

        self.assertEqual(decision["winner"], "stripped")


if __name__ == "__main__":
    unittest.main()
