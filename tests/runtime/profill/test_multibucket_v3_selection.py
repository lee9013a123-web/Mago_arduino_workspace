from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts/4_profill/optimization/16_select_multibucket_v3.py"
)
SPEC = importlib.util.spec_from_file_location("multibucket_v3_selection", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SELECTION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SELECTION
SPEC.loader.exec_module(SELECTION)


class MultibucketV3SelectionTests(unittest.TestCase):
    def test_feature_names_are_bucket_specific(self) -> None:
        paths = SELECTION._features(
            Path("features"), 498, ("0000", "0005", "0006")
        )
        self.assertEqual(
            [path.name for path in paths],
            [
                "multi__speaker_0000__498.f32",
                "multi__speaker_0005__498.f32",
                "multi__speaker_0006__498.f32",
            ],
        )

    def test_family_command_uses_batch_and_baseline_check_only(self) -> None:
        command = SELECTION._benchmark_command(
            script=Path("family.py"), bucket=298,
            profile=Path("profile.json"), binary=Path("bench"),
            plan=Path("plan.bin"), weights=Path("weights.bin"),
            features=[Path("feature.f32")],
            results_dir=Path("results"), runs_dir=Path("runs"),
            modes=("baseline", "v4", "v5"), warmup=2, repeat=10,
            preflight_only=True, force=False,
        )
        self.assertIn("--bucket-frames", command)
        self.assertIn("298", command)
        self.assertIn("--baseline-check-only", command)
        self.assertIn("--preflight-only", command)

    def test_source_generation_always_preserves_98_plan(self) -> None:
        command = SELECTION._generate_source_command(
            (298, 498, 998), Path("plan_98.json"),
            force=True,
        )
        plan_specs = [
            command[index + 1]
            for index, value in enumerate(command[:-1]) if value == "--plan"
        ]
        self.assertEqual(len(plan_specs), 4)
        self.assertTrue(plan_specs[0].startswith("98="))
        self.assertTrue(any(spec.startswith("998=") for spec in plan_specs))


if __name__ == "__main__":
    unittest.main()
