"""Archive identity and command-line checks for reproducible experiments."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from mixed_weak_pcrb.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_manifest_files_and_hashes(self):
        manifest = json.loads((ROOT / "metadata/manifest.json").read_text(encoding="utf-8"))
        for name, record in manifest["models"].items():
            for field in ("checkpoint", "resolved_config", "training_cost"):
                if field not in record:
                    continue
                path = Path(record[field])
                self.assertFalse(path.is_absolute(), (name, field))
                self.assertNotIn("..", path.parts)
                full = ROOT / path
                self.assertTrue(full.is_file(), (name, field))
                if field + "_sha256" in record:
                    self.assertEqual(hashlib.sha256(full.read_bytes()).hexdigest(), record[field + "_sha256"], (name, field))

    def test_original_A_is_distinct_from_seed42_repeat(self):
        original = ROOT / "pretrained/local_load_plate/model.weights.h5"
        repeat = ROOT / "pretrained/seeds/seed42_normalized/model.weights.h5"
        self.assertNotEqual(hashlib.sha256(original.read_bytes()).digest(), hashlib.sha256(repeat.read_bytes()).digest())

    def test_canonical_reference_metrics_match_result_records(self):
        for case, name in (("sinusoidal_plate", "sinusoidal"), ("local_load_plate", "local"), ("heterogeneous_roof", "heterogeneous")):
            reference = json.loads((ROOT / "pretrained" / case / "reference_metrics.json").read_text(encoding="utf-8"))
            record = json.loads((ROOT / "results/raw/canonical" / (name + ".json")).read_text(encoding="utf-8"))
            self.assertEqual(reference["field_errors"], record["field_errors"], case)
            self.assertEqual(reference["peak_response"], record["peak_response"], case)

    def test_zero_epochs_rejected_before_training(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "train.py"), "--config", "configs/sinusoidal_plate.json", "--epochs", "0"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--epochs must be positive", result.stderr)

    def test_final_matched_configs(self):
        for formulation, entrypoint in (("weak", "local_weak"), ("strong", "local_strong")):
            for seed in (42, 7, 2026):
                config = load_config(ROOT / f"configs/final/{entrypoint}_seed{seed}.json")
                self.assertEqual(config["seed"], seed)
                self.assertIsNone(config["output_scale"])
                self.assertEqual(config["physics"]["output_scale_margin"], 1.5)
                self.assertEqual(config["network"]["output_activation"], "linear")
                self.assertFalse(config["training"]["scale_calibration"]["enabled"])
                self.assertTrue(config["weak_form"]["residual_normalization"]["enabled"])
                if formulation == "strong":
                    self.assertEqual(config["formulation"], "strong")
                    self.assertEqual(config["loss_form"], "first_order_eight_field_strong")

    def test_final_multiseed_residual_record(self):
        with (ROOT / "results/raw/strong_control/multiseed_prior_linear_chi1p5.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            import csv

            rows = list(csv.DictReader(stream))
        self.assertEqual(
            {(row["formulation"], int(row["seed"])) for row in rows},
            {(method, seed) for method in ("strong", "weak") for seed in (42, 7, 2026)},
        )
        for row in rows:
            self.assertTrue(all(float(row[group]) >= 0 for group in ("kinematic", "constitutive", "moment", "equilibrium")))


if __name__ == "__main__":
    unittest.main()
