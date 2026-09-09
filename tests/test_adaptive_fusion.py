"""Unit tests for Adaptive / Gated Multi-Resolution Fusion module."""

import unittest
import numpy as np
import pandas as pd
import torch

from tuned.adaptive_fusion import SoftmaxGateModule, AdaptiveFusion
from tuned.pipeline import Fusion, LOG_SHAPE, HAZARD_SHAPE


class TestAdaptiveFusion(unittest.TestCase):

    def test_gate_module_shapes_and_constraints(self):
        gate = SoftmaxGateModule(in_features=3, out_components=3)
        g = torch.randn(10, 3)
        e_matrix = torch.randn(10, 3)

        logits, weights = gate(g, e_matrix)
        
        # Check shapes
        self.assertEqual(logits.shape, (10,))
        self.assertEqual(weights.shape, (10, 3))

        # Check non-negativity
        self.assertTrue(torch.all(weights >= 0.0))

        # Check sum to 1.0
        sums = torch.sum(weights, dim=-1)
        np.testing.assert_allclose(sums.detach().numpy(), 1.0, atol=1e-5)

        # Check no NaNs or Infs
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())
        self.assertFalse(torch.isnan(weights).any())
        self.assertFalse(torch.isinf(weights).any())

    def test_adaptive_fusion_fit_predict(self):
        np.random.seed(42)
        n = 100
        x = np.random.randn(n, 500).astype(np.float32)
        frame = pd.DataFrame({
            "wafer_id": ["W1"] * 50 + ["W2"] * 50,
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
        })

        base_fusion = Fusion(use_block=False, use_parametric=True, use_hazard=False, correct_prior=False)
        adaptive = AdaptiveFusion(base_fusion=base_fusion, gate_epochs=10, random_state=42)

        rows = np.ones(n, dtype=bool)
        adaptive.fit(frame, x, rows)

        probs, weights = adaptive.predict_proba(frame, x, rows)

        self.assertEqual(len(probs), n)
        self.assertEqual(weights.shape, (n, 3))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        self.assertTrue(np.all(weights >= 0.0))
        np.testing.assert_allclose(np.sum(weights, axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
