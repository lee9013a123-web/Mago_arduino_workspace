from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
PROFILL_DIR = ROOT / "scripts/4_profill"
if str(PROFILL_DIR) not in sys.path:
    sys.path.insert(0, str(PROFILL_DIR))

from bucket_profile_identity import (  # noqa: E402
    BucketProfileIdentityError,
    validate_profile_against_plan,
)


class BucketProfileIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = (
            ROOT / "results/profiling/e7_98/final_v3_hybrid/"
            "operator_profile.json"
        )
        cls.plan = (
            ROOT / "runs/runtime/kernel_optimization/e7/bundle/"
            "execution_plans/plan_98.bin"
        )

    def test_accepts_exact_profile_and_execution_plan(self) -> None:
        result = validate_profile_against_plan(self.profile, self.plan, 98)
        self.assertTrue(result["ready"])
        self.assertEqual(result["operator_count"], 832)
        self.assertEqual(result["qlinear_conv_count"], 225)

    def test_rejects_cross_bucket_profile_before_measurement(self) -> None:
        with self.assertRaises(BucketProfileIdentityError):
            validate_profile_against_plan(self.profile, self.plan, 298)


if __name__ == "__main__":
    unittest.main()
