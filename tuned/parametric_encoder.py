"""Learned compact parametric encoder for the 500 die-level measurements.

Architecture:
500 -> 256 -> 128 -> 64 -> 1

Features:
- Lightweight PyTorch MLP with BatchNorm1d, SiLU activations, and Dropout (0.2).
- L2 regularization (weight_decay=1e-4) and class-imbalance aware BCE loss.
- Exposes both 64-dimensional embeddings and 1D logit parametric scores.
- Scikit-learn compatible interface matching tuned.channels.DiagonalScore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ParametricNet(nn.Module):
    """PyTorch MLP encoder: 500 -> 256 -> 128 -> 64 -> 1."""

    def __init__(self, in_features: int = 500, dropout: float = 0.2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.head = nn.Linear(64, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.head(feat).squeeze(-1)


@dataclass
class ParametricEncoderScore:
    """Scikit-Learn style wrapper around PyTorch ParametricNet."""

    in_features: int = 500
    epochs: int = 8
    batch_size: int = 2048
    lr: float = 5e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    random_state: int = 42
    scaler_: StandardScaler | None = field(default=None, repr=False)
    net_: ParametricNet | None = field(default=None, repr=False)
    scale_: float = 1.0

    def fit(
        self,
        x: np.ndarray,
        positive_mask: np.ndarray,
        negative_mask: np.ndarray,
    ) -> ParametricEncoderScore:
        """Fit the MLP on selected training rows.

        `positive_mask` and `negative_mask` are boolean masks over full `x`.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        train_mask = positive_mask | negative_mask
        if not np.any(train_mask):
            raise ValueError("No rows selected for fitting ParametricEncoderScore")

        # Standardize features on train rows only
        self.scaler_ = StandardScaler()
        x_train = self.scaler_.fit_transform(x[train_mask].astype(np.float32))
        y_train = positive_mask[train_mask].astype(np.float32)

        n_pos = float(y_train.sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)

        dataset = TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        self.net_ = ParametricNet(
            in_features=self.in_features, dropout=self.dropout
        )
        self.net_.train()

        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.net_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        # Compute scaling standard deviation over negative training class
        self.net_.eval()
        with torch.no_grad():
            neg_x = self.scaler_.transform(x[negative_mask].astype(np.float32))
            neg_logits = (
                self.net_(torch.from_numpy(neg_x)).cpu().numpy().astype(np.float64)
            )
            self.scale_ = float(neg_logits.std()) or 1.0

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Compute standardized 1D logit parametric score for input rows."""
        if self.net_ is None or self.scaler_ is None:
            raise RuntimeError("ParametricEncoderScore must be fitted before transform")
        self.net_.eval()
        x_scaled = self.scaler_.transform(x.astype(np.float32))
        with torch.no_grad():
            logits = (
                self.net_(torch.from_numpy(x_scaled))
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        return logits / self.scale_

    def transform_embeddings(self, x: np.ndarray) -> np.ndarray:
        """Compute 64-dimensional feature embeddings for input rows."""
        if self.net_ is None or self.scaler_ is None:
            raise RuntimeError("ParametricEncoderScore must be fitted before transform")
        self.net_.eval()
        x_scaled = self.scaler_.transform(x.astype(np.float32))
        with torch.no_grad():
            embeddings = (
                self.net_.forward_features(torch.from_numpy(x_scaled))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        return embeddings


def fit_encoder(
    x: np.ndarray,
    label: np.ndarray,
    old_label: np.ndarray,
    rows: np.ndarray,
    use_old_fails: bool = True,
    random_state: int = 42,
    epochs: int = 8,
) -> ParametricEncoderScore:
    """Fit parametric encoder on rows selected by `rows`.

    Positives are post-test failures among eligible dies, plus -- when specified --
    every `old_label == 1` pre-test failure die. Negatives are passing eligible dies.
    """
    label = np.asarray(label).astype(bool)
    old_label = np.asarray(old_label).astype(bool)
    rows = np.asarray(rows).astype(bool)

    positive = label & ~old_label
    if use_old_fails:
        positive = positive | old_label
    negative = (~label) & (~old_label)

    encoder = ParametricEncoderScore(
        in_features=x.shape[1], random_state=random_state, epochs=epochs
    )
    return encoder.fit(x, positive & rows, negative & rows)
