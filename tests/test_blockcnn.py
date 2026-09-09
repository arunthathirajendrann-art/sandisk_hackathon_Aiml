"""Focused unit tests for the 1D BlockCNN and LRT+CNN block encoder functionality."""

import unittest
import numpy as np
import torch
import pandas as pd

from tuned.blockcnn import BlockCNN, fit_predict_oof_cnn
from tuned.pipeline import Fusion


class TestBlockCNN(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.n_samples = 100
        self.seq_len = 2000
        self.readings = np.random.randn(self.n_samples, self.seq_len).astype(np.float32)
        # Inject anomaly pattern into some samples
        self.labels = np.random.binomial(1, 0.1, size=self.n_samples).astype(np.int8)
        self.readings[self.labels == 1, 500:600] += 2.0
        self.old_labels = np.zeros(self.n_samples, dtype=np.int8)
        self.wafers = np.array([f"wafer_{i % 5}" for i in range(self.n_samples)])

    def test_input_output_shapes(self):
        """Verify input sequence length 2000 produces correct output shape."""
        model = BlockCNN()
        x_tensor = torch.from_numpy(self.readings[:10])
        out = model(x_tensor)
        self.assertEqual(out.shape, (10,))

    def test_no_nans_or_infs(self):
        """Verify forward pass produces finite values without NaNs or infs."""
        model = BlockCNN()
        x_tensor = torch.from_numpy(self.readings)
        out = model(x_tensor)
        self.assertTrue(torch.all(torch.isfinite(out)).item())

    def test_fit_predict_oof_cnn_path(self):
        """Verify end-to-end 5-fold OOF training and inference path."""
        oof_logits = fit_predict_oof_cnn(
            readings=self.readings,
            label=self.labels,
            old_label=self.old_labels,
            wafer=self.wafers,
            n_splits=3,
            epochs=1,
            batch_size=32,
            verbose=False,
        )
        self.assertEqual(oof_logits.shape, (self.n_samples,))
        self.assertTrue(np.all(np.isfinite(oof_logits)))

    def test_lrt_cnn_fusion(self):
        """Verify LRT+CNN block score fusion in Fusion pipeline."""
        df = pd.DataFrame({
            "wafer_id": self.wafers,
            "die_row": np.arange(self.n_samples),
            "die_col": np.arange(self.n_samples),
            "old_label": self.old_labels,
            "label": self.labels,
            "haz_shape": np.ones(self.n_samples),
            "haz_log_shape": np.zeros(self.n_samples),
            "block_feat1": self.readings[:, 0],
            "block_feat2": self.readings[:, 1],
        })
        x_param = np.random.randn(self.n_samples, 10).astype(np.float32)
        cnn_scores = np.random.randn(self.n_samples)

        fusion = Fusion(
            use_block=True,
            use_block_lrt=True,
            use_block_cnn=True,
            use_parametric=False,
            block_cnn_scores=cnn_scores,
        )
        train_rows = np.ones(self.n_samples, dtype=bool)
        fusion.fit(df, x_param, train_rows)
        probs = fusion.predict_proba(df, x_param, train_rows)

        self.assertEqual(len(probs), self.n_samples)
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))


if __name__ == "__main__":
    unittest.main()
