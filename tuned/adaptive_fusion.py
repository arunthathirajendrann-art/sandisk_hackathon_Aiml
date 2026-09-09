"""Adaptive / Gated Multi-Resolution Fusion module for SanDisk Yield AI.

Dynamically weights component evidence (Parametric, Spatial, Block LRT) based on die-level
compressed evidence representations and pre-test wafer context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from tuned.pipeline import Fusion, CAP, _logit


class SoftmaxGateModule(nn.Module):
    """Small linear softmax gating network mapping 3 evidence inputs to 3 weights."""

    def __init__(self, in_features: int = 3, out_components: int = 3):
        super().__init__()
        self.linear = nn.Linear(in_features, out_components)
        # Initialize near zero for initial equal weights (1/3, 1/3, 1/3)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.scale_bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.scale_weight = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(self, g: torch.Tensor, e_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # g: [N, 3], e_matrix: [N, 3] (e_param, e_spatial, e_block)
        logits = self.linear(g)  # [N, 3]
        weights = torch.softmax(logits, dim=-1)  # [N, 3], sums to 1.0
        # Weighted sum multiplied by 3 (so equal weights 1/3 sum to e_param + e_spatial + e_block)
        weighted_e = torch.sum(weights * e_matrix, dim=-1, keepdim=True) * 3.0  # [N, 1]
        out_logits = self.scale_bias + self.scale_weight * weighted_e
        return out_logits.squeeze(-1), weights


@dataclass
class AdaptiveFusion:
    """Wrapper around Fusion pipeline with learned adaptive gating."""

    base_fusion: Fusion = field(default_factory=Fusion)
    gate_epochs: int = 150
    lr: float = 0.02
    random_state: int = 42
    scaler_: StandardScaler | None = field(default=None, repr=False)
    gate_module_: SoftmaxGateModule | None = field(default=None, repr=False)

    def fit(self, frame: pd.DataFrame, x: np.ndarray, rows: np.ndarray) -> AdaptiveFusion:
        rows = np.asarray(rows).astype(bool)
        # 1. Fit base Fusion pipeline
        self.base_fusion.fit(frame, x, rows)

        # 2. Extract compressed component evidence on training eligible dies (old_label == 0)
        old_label = frame["old_label"].to_numpy(dtype=np.int8)
        label = frame["label"].to_numpy(dtype=np.int8)
        fit_rows = rows & (old_label == 0)

        design = self.base_fusion._design(frame, x, fit_rows)
        parts = self.base_fusion.head_.partial(design)
        zero = np.zeros(len(design))

        # Component evidence log-odds
        e_param = parts.get("parametric_score", zero).astype(np.float32)
        e_block = parts.get("block_score", zero).astype(np.float32)
        
        # Spatial hazard & prior evidence
        correction, _ = self.base_fusion._split(design)
        hazard = self.base_fusion._shape(frame, fit_rows) * np.exp(correction)
        wafer = frame.loc[fit_rows, "wafer_id"].astype(str).to_numpy()
        wafer_rates = np.array([self.base_fusion.fitted_rates_.get(w, self.base_fusion.overall_rate_) for w in wafer], dtype=np.float32)
        prior_p = np.clip(wafer_rates * hazard, 1e-9, CAP)
        e_spatial = _logit(prior_p).astype(np.float32)

        # Strictly 3 leakage-free evidence inputs: [e_param, e_spatial, e_block]
        g_raw = np.column_stack([e_param, e_spatial, e_block])
        self.scaler_ = StandardScaler().fit(g_raw)
        g_scaled = self.scaler_.transform(g_raw)

        e_mat = np.column_stack([e_param, e_spatial, e_block])

        # 3. Train SoftmaxGateModule using PyTorch Adam
        torch.manual_seed(self.random_state)
        self.gate_module_ = SoftmaxGateModule(in_features=3, out_components=3)

        g_tensor = torch.tensor(g_scaled, dtype=torch.float32)
        e_tensor = torch.tensor(e_mat, dtype=torch.float32)
        y_tensor = torch.tensor(label[fit_rows], dtype=torch.float32)

        # Class weight for BCE loss
        pos_weight = torch.tensor([(len(y_tensor) - y_tensor.sum()) / (y_tensor.sum() + 1e-5)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.gate_module_.parameters(), lr=self.lr, weight_decay=1e-4)

        self.gate_module_.train()
        for epoch in range(self.gate_epochs):
            optimizer.zero_grad()
            pred_logits, _ = self.gate_module_(g_tensor, e_tensor)
            loss = criterion(pred_logits, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def predict_proba(self, frame: pd.DataFrame, x: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (predicted_probabilities, gate_weights matrix [N, 3])."""
        rows = np.asarray(rows).astype(bool)
        design = self.base_fusion._design(frame, x, rows)
        parts = self.base_fusion.head_.partial(design)
        zero = np.zeros(len(design))

        e_param = parts.get("parametric_score", zero).astype(np.float32)
        e_block = parts.get("block_score", zero).astype(np.float32)

        correction, _ = self.base_fusion._split(design)
        hazard = self.base_fusion._shape(frame, rows) * np.exp(correction)
        wafer = frame.loc[rows, "wafer_id"].astype(str).to_numpy()
        wafer_rates = np.array([self.base_fusion.fitted_rates_.get(w, self.base_fusion.overall_rate_) for w in wafer], dtype=np.float32)
        prior_p = np.clip(wafer_rates * hazard, 1e-9, CAP)
        e_spatial = _logit(prior_p).astype(np.float32)

        # Strictly 3 leakage-free evidence inputs: [e_param, e_spatial, e_block]
        g_raw = np.column_stack([e_param, e_spatial, e_block])
        g_scaled = self.scaler_.transform(g_raw)
        e_mat = np.column_stack([e_param, e_spatial, e_block])

        self.gate_module_.eval()
        with torch.no_grad():
            g_tensor = torch.tensor(g_scaled, dtype=torch.float32)
            e_tensor = torch.tensor(e_mat, dtype=torch.float32)
            pred_logits, weights = self.gate_module_(g_tensor, e_tensor)
            probs = torch.sigmoid(pred_logits).numpy()
            w_np = weights.numpy()

        return probs, w_np
