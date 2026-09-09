"""Graph Neural Network Spatial Representation for Wafer Die Yield Prediction.

Learns per-die spatial context via 8-neighbor grid message passing within each wafer.
Strictly respects wafer boundaries (no cross-wafer edges).
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def build_split_graph(df: pd.DataFrame) -> torch.Tensor:
    """Construct a block-diagonal 8-neighbor sparse adjacency matrix for a split.

    Nodes are dies. Edges only exist between 8-neighbor dies on the SAME wafer.
    No message passing across wafer boundaries or fold boundaries.
    """
    rows, cols, vals = [], [], []
    offset = 0

    for _, wafer in df.groupby("wafer_id", sort=False):
        die_row = wafer["die_row"].to_numpy(dtype=np.int64)
        die_col = wafer["die_col"].to_numpy(dtype=np.int64)
        n = len(die_row)
        coords = {(r, c): idx for idx, (r, c) in enumerate(zip(die_row, die_col))}

        w_rows, w_cols = [], []
        # Self-loops
        for i in range(n):
            w_rows.append(i)
            w_cols.append(i)

        # 8-neighbor grid edges
        for (r, c), i in coords.items():
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = (r + dr, c + dc)
                    if neighbor in coords:
                        w_rows.append(i)
                        w_cols.append(coords[neighbor])

        # Degree normalization: D^(-1/2) A D^(-1/2)
        deg = np.zeros(n, dtype=np.float32)
        for r in w_rows:
            deg[r] += 1.0
        deg_inv_sqrt = np.power(deg, -0.5)

        for r, c in zip(w_rows, w_cols):
            rows.append(r + offset)
            cols.append(c + offset)
            vals.append(deg_inv_sqrt[r] * deg_inv_sqrt[c])

        offset += n

    total_n = len(df)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)
    shape = torch.Size([total_n, total_n])
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def extract_node_features(frame: pd.DataFrame, parametric_score: np.ndarray | None = None) -> np.ndarray:
    """Extract pre-test node features per die."""
    row = frame["die_row"].to_numpy(dtype=np.float64)
    col = frame["die_col"].to_numpy(dtype=np.float64)

    # Wafer grid shapes per wafer
    norm_row = np.zeros(len(frame), dtype=np.float32)
    norm_col = np.zeros(len(frame), dtype=np.float32)

    for _, wafer in frame.groupby("wafer_id", sort=False):
        idx = wafer.index
        r = wafer["die_row"].to_numpy(dtype=np.float64)
        c = wafer["die_col"].to_numpy(dtype=np.float64)
        nr, nc = max(float(r.max()), 1.0), max(float(c.max()), 1.0)
        norm_row[idx] = (r - nr / 2.0) / (nr / 2.0)
        norm_col[idx] = (c - nc / 2.0) / (nc / 2.0)

    feature_list = [
        frame["old_label"].to_numpy(dtype=np.float32)[:, None],
        frame["haz_radius"].to_numpy(dtype=np.float32)[:, None],
        frame["haz_edge_distance"].to_numpy(dtype=np.float32)[:, None],
        frame["haz_density_w5"].to_numpy(dtype=np.float32)[:, None],
        frame["haz_density_w3"].to_numpy(dtype=np.float32)[:, None],
        frame["haz_density_w11"].to_numpy(dtype=np.float32)[:, None],
        frame["haz_nearest_old_fail"].to_numpy(dtype=np.float32)[:, None],
        frame["haz_wafer_old_fail_rate"].to_numpy(dtype=np.float32)[:, None],
        norm_row[:, None],
        norm_col[:, None],
    ]

    if parametric_score is not None:
        feature_list.append(np.asarray(parametric_score, dtype=np.float32)[:, None])

    return np.hstack(feature_list)


class GCNLayer(nn.Module):
    """Normalized sparse Graph Convolutional layer: A_hat @ X @ W."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj_sparse: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        return torch.sparse.mm(adj_sparse, h)


class WaferGNN(nn.Module):
    """Lightweight 2-layer Graph Convolutional Network for spatial die yield modeling."""

    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)

        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj_sparse: torch.Tensor) -> torch.Tensor:
        h = self.drop1(self.act1(self.norm1(self.gcn1(x, adj_sparse))))
        h = self.drop2(self.act2(self.norm2(self.gcn2(h, adj_sparse))))
        return self.head(h).squeeze(-1)


def fit_gnn_spatial(
    frame: pd.DataFrame,
    parametric_score: np.ndarray | None,
    train_mask: np.ndarray,
    hidden_dim: int = 32,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    random_state: int = 42,
) -> tuple[WaferGNN, StandardScaler, float]:
    """Fit WaferGNN spatial representation on training fold wafers."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    raw_features = extract_node_features(frame, parametric_score)
    scaler = StandardScaler()
    
    # Fit scaler on training rows only
    train_mask = np.asarray(train_mask, dtype=bool)
    scaler.fit(raw_features[train_mask])
    scaled_features = scaler.transform(raw_features)

    # Build graph for full frame (wafer boundaries strictly respected)
    adj = build_split_graph(frame)
    x_tensor = torch.tensor(scaled_features, dtype=torch.float32)

    labels = frame["label"].to_numpy(dtype=np.float32)
    old_labels = frame["old_label"].to_numpy(dtype=np.int8)
    eligible_train = train_mask & (old_labels == 0)

    y_train = torch.tensor(labels[eligible_train], dtype=torch.float32)
    train_indices = torch.tensor(np.flatnonzero(eligible_train), dtype=torch.long)

    # Calculate class pos_weight for BCE
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32)

    model = WaferGNN(in_dim=scaled_features.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_tensor, adj)
        loss = criterion(logits[train_indices], y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        all_logits = model(x_tensor, adj).numpy()

    return model, scaler, all_logits
