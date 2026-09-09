"""Unit tests for Probability Calibration module."""

import unittest
import numpy as np

from tuned.calibration import PlattCalibrator, IsotonicCalibrator, compute_calibration_curve


class TestCalibration(unittest.TestCase):

    def test_platt_calibrator_bounds_and_nan(self):
        np.random.seed(42)
        y_prob = np.random.uniform(0.01, 0.99, size=100)
        y_true = (y_prob > 0.5).astype(int)

        cal = PlattCalibrator(random_state=42)
        cal.fit(y_prob, y_true)

        cal_prob = cal.predict_proba(y_prob)

        self.assertEqual(len(cal_prob), 100)
        self.assertTrue(np.all(cal_prob >= 0.0) and np.all(cal_prob <= 1.0))
        self.assertFalse(np.isnan(cal_prob).any())
        self.assertFalse(np.isinf(cal_prob).any())

    def test_isotonic_calibrator_bounds_and_nan(self):
        np.random.seed(42)
        y_prob = np.random.uniform(0.01, 0.99, size=100)
        y_true = (y_prob > 0.5).astype(int)

        cal = IsotonicCalibrator()
        cal.fit(y_prob, y_true)

        cal_prob = cal.predict_proba(y_prob)

        self.assertEqual(len(cal_prob), 100)
        self.assertTrue(np.all(cal_prob >= 0.0) and np.all(cal_prob <= 1.0))
        self.assertFalse(np.isnan(cal_prob).any())
        self.assertFalse(np.isinf(cal_prob).any())

    def test_calibration_curve_generation(self):
        np.random.seed(42)
        y_prob = np.random.uniform(0.0, 1.0, size=200)
        y_true = np.random.binomial(1, y_prob)

        df_curve, ece = compute_calibration_curve(y_true, y_prob, n_bins=10)

        self.assertEqual(len(df_curve), 10)
        self.assertIn("mean_predicted_probability", df_curve.columns)
        self.assertIn("observed_failure_rate", df_curve.columns)
        self.assertIn("sample_count", df_curve.columns)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)
        self.assertEqual(df_curve["sample_count"].sum(), 200)


if __name__ == "__main__":
    unittest.main()
