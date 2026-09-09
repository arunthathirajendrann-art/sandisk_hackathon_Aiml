"""Unit tests for Multi-Resolution Interpretability module."""

import unittest
import numpy as np
import pandas as pd

from tuned.pipeline import Fusion, LOG_SHAPE, HAZARD_SHAPE
from tuned.interpretability import (
    decompose_logit,
    explain_parametric,
    explain_spatial,
    explain_block,
    explain_die,
)


class TestInterpretability(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 50
        self.x = np.random.randn(n, 500).astype(np.float32)
        self.readings = np.random.randn(n, 2000).astype(np.float32)
        
        # Add a synthetic spike to die 0 to test anomaly peak localization
        self.readings[0, 500:520] += 5.0

        self.frame = pd.DataFrame({
            "wafer_id": ["W1"] * 25 + ["W2"] * 25,
            "label": np.random.randint(0, 2, size=n, dtype=np.int8),
            "old_label": np.zeros(n, dtype=np.int8),
            "die_x": np.random.randint(0, 10, size=n),
            "die_y": np.random.randint(0, 10, size=n),
            HAZARD_SHAPE: np.ones(n, dtype=np.float64),
            LOG_SHAPE: np.zeros(n, dtype=np.float64),
            "haz_density_w3": np.random.randn(n),
            "haz_density_w7": np.random.randn(n),
            "haz_density_w11": np.random.randn(n),
            "haz_radius": np.random.randn(n),
            "haz_edge_distance": np.random.randn(n),
            "haz_nearest_old_fail": np.random.randn(n),
            "haz_wafer_old_fail_rate": np.random.randn(n),
            "block_level": np.random.randn(n),
            "block_median": np.random.randn(n),
            "block_scale": np.random.randn(n),
        })

        self.model = Fusion(use_block=True, use_parametric=True, use_hazard=False, correct_prior=False, random_state=42)
        rows = np.ones(n, dtype=bool)
        self.model.fit(self.frame, self.x, rows)

    def test_logit_decomposition_sum_check(self):
        row_idx = 0
        prob_before = float(self.model.predict_proba(self.frame, self.x, np.array([True] + [False]*49))[0])

        decomp = decompose_logit(self.model, self.frame, self.x, row_idx)

        # Check exact sum: C_spatial + C_param + C_block == final_logit
        sum_components = decomp.c_spatial + decomp.c_param + decomp.c_block
        self.assertAlmostEqual(sum_components, decomp.final_logit, places=6)

        # Check residual against model predict_proba
        self.assertLess(decomp.residual, 1e-5)

        # Non-mutation sanity check
        prob_after = float(self.model.predict_proba(self.frame, self.x, np.array([True] + [False]*49))[0])
        self.assertEqual(prob_before, prob_after)

    def test_parametric_feature_attribution_sum(self):
        row_idx = 0
        diag_score = self.model.diagonal_
        x_row = self.x[row_idx]

        df_top = explain_parametric(diag_score, x_row, top_k=500)

        total_attr_sum = df_top["contribution"].sum()
        actual_diag_score = float(diag_score.transform(x_row.reshape(1, -1))[0])

        self.assertAlmostEqual(total_attr_sum, actual_diag_score, places=5)

    def test_explain_die_pipeline(self):
        exp = explain_die(self.model, self.frame, self.x, self.readings, row_idx=0)

        self.assertIn("predicted_probability", exp)
        self.assertIn("logit_decomposition", exp)
        self.assertIn("top_parametric_features", exp)
        self.assertIn("spatial_context", exp)
        self.assertIn("block_context", exp)

        p = exp["predicted_probability"]
        self.assertTrue(0.0 <= p <= 1.0)
        self.assertFalse(np.isnan(p))

    def test_block_anomaly_localization(self):
        block_exp = explain_block(self.model, self.frame, self.readings[0], row_idx=0)

        peak_idx = block_exp.get("peak_anomaly_index")
        self.assertIsNotNone(peak_idx)
        self.assertTrue(0 <= peak_idx <= 1999)


if __name__ == "__main__":
    unittest.main()
