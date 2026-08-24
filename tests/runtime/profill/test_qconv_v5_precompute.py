import unittest

from src.python.runtime_bundle_exporter.planner.qconv_precompute import (
    all_weight_zero_points_are_zero,
    corrected_bias,
    make_v5_pack_plan,
    weight_sums_by_output,
)


class QConvV5PrecomputeTest(unittest.TestCase):
    def test_zero_points(self) -> None:
        self.assertTrue(all_weight_zero_points_are_zero([0, 0, 0]))
        self.assertFalse(all_weight_zero_points_are_zero([0, -1, 0]))

    def test_weight_sums_and_corrected_bias(self) -> None:
        sums = weight_sums_by_output([1, -2, 3, 4, -5, 6], 2, 3)
        self.assertEqual(sums, [2, 5])
        self.assertEqual(corrected_bias([10, -3], 7, sums), [-4, -38])

    def test_pack_plan_validation(self) -> None:
        plan = make_v5_pack_plan(
            output_channels=32,
            input_channels=32,
            kernel_elements=9,
            symmetric_weight_zero_point=True,
        )
        self.assertTrue(plan.symmetric_weight_zero_point)
        with self.assertRaises(ValueError):
            make_v5_pack_plan(
                output_channels=0,
                input_channels=32,
                kernel_elements=9,
                symmetric_weight_zero_point=True,
            )


if __name__ == "__main__":
    unittest.main()
