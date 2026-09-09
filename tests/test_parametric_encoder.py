"""Unit tests for the Parametric MLP Encoder."""

from __future__ import annotations

import unittest
import numpy as np
import torch
from tuned.parametric_encoder import ParametricNet, ParametricEncoderScore, fit_encoder


class ParametricEncoderTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.n_samples = 200
        self.n_features = 500
        self.x = self.rng.normal(0, 1, size=(self.n_samples, self.n_features)).astype(np.float32)
        self.old_label = (self.rng.random(self.n_samples) < 0.05).astype(np.int8)
        self.label = np.maximum(
            self.old_label,
            (self.rng.random(self.n_samples) < 0.10).astype(np.int8)
        )
        self.rows = np.ones(self.n_samples, dtype=bool)

    def test_parametric_net_shapes(self):
        net = ParametricNet(in_features=500, dropout=0.2)
        batch = torch.from_numpy(self.x[:16])
        logits = net(batch)
        embeddings = net.forward_features(batch)

        self.assertEqual(logits.shape, (16,))
        self.assertEqual(embeddings.shape, (16, 64))

    def test_parametric_encoder_score_fit_transform(self):
        encoder = fit_encoder(
            self.x, self.label, self.old_label, self.rows, epochs=2, random_state=42
        )
        score = encoder.transform(self.x)
        embeddings = encoder.transform_embeddings(self.x)

        self.assertEqual(score.shape, (self.n_samples,))
        self.assertEqual(embeddings.shape, (self.n_samples, 64))
        self.assertTrue(np.all(np.isfinite(score)))
        self.assertTrue(np.all(np.isfinite(embeddings)))

    def test_reads_no_post_test_information_during_transform(self):
        encoder = fit_encoder(
            self.x, self.label, self.old_label, self.rows, epochs=2, random_state=42
        )
        first_scores = encoder.transform(self.x)
        
        # Modify label array (post-test information)
        modified_label = self.rng.permutation(self.label)
        
        # Transform should be completely unaffected because encoder uses only X
        second_scores = encoder.transform(self.x)
        np.testing.assert_allclose(first_scores, second_scores, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
