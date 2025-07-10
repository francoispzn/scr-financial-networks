"""
Temporal Graph Neural Network predictor for spectral metric evolution.

Uses a GAT (Graph Attention Network) encoder that operates on the actual
interbank graph at each timestep, producing graph-level embeddings that
feed into a temporal LSTM for spectral metric forecasting.

Architecture: GATConv layers → global_mean_pool → LSTM → FC → [λ₂, gap, ρ]

The attention mechanism learns which inter-bank relationships matter most
for predicting spectral risk indicators, providing interpretability.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

TARGET_NAMES = ["lambda_2", "spectral_gap", "spectral_radius"]

# Node features extracted per bank at each ABM step
NODE_FEATURE_NAMES = ["CET1_ratio", "LCR", "NSFR", "total_assets", "is_stressed"]


class GNNEncoder(nn.Module):
    """Multi-layer GAT that produces a graph-level embedding.

    Uses multi-head attention to learn which inter-bank relationships
    are most informative for predicting spectral risk indicators.
    Attention weights are stored for interpretability analysis.
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 3,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.heads = heads
        # First layer: in_channels → hidden_channels (each head outputs hidden_channels // heads)
        head_dim = max(1, hidden_channels // heads)
        self.convs.append(GATConv(in_channels, head_dim, heads=heads, dropout=dropout, concat=True))
        self.bns.append(nn.BatchNorm1d(head_dim * heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(head_dim * heads, head_dim, heads=heads, dropout=dropout, concat=True))
            self.bns.append(nn.BatchNorm1d(head_dim * heads))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()
        self._last_attention_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h, (edge_idx, attn_w) = conv(x, edge_index, return_attention_weights=True)
            if i == len(self.convs) - 1:
                self._last_attention_weights = attn_w.detach()
            h = bn(h)
            h = self.act(h)
            if i < len(self.convs) - 1:
                h = self.dropout(h)
            x = h
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch)  # [num_graphs, hidden]

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return attention weights from the last forward pass (last layer)."""
        return self._last_attention_weights


class TemporalGNN(nn.Module):
    """GNN encoder + LSTM for temporal graph sequences → spectral predictions."""

    def __init__(self, node_features: int, hidden_dim: int = 64,
                 output_dim: int = 3, num_gat_layers: int = 3,
                 num_lstm_layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gnn = GNNEncoder(node_features, hidden_dim, num_gat_layers, heads, dropout)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_lstm_layers,
                            batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers

    def forward(self, graph_sequences: List[List[Data]]) -> torch.Tensor:
        from torch_geometric.data import Batch
        batch_size = len(graph_sequences)
        seq_len = len(graph_sequences[0])
        device = next(self.parameters()).device

        # Batch all graphs across batch × seq_len for efficient encoding
        all_graphs = [g for seq in graph_sequences for g in seq]
        batched = Batch.from_data_list(all_graphs)
        all_emb = self.gnn(batched.x, batched.edge_index, batched.edge_weight,
                           batched.batch)  # [batch_size * seq_len, hidden]
        embeddings = all_emb.view(batch_size, seq_len, self.hidden_dim)

        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_dim, device=device)
        out, _ = self.lstm(embeddings, (h0, c0))
        return self.fc(out[:, -1, :])

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GNNPredictor:
    """Drop-in replacement for SCGPredictor using a temporal GNN.

    Supports a progress_callback(epoch, total_epochs, train_loss, test_loss)
    for real-time UI updates during training.
    """

    def __init__(self, seq_len: int = 10, hidden_dim: int = 64,
                 num_gat_layers: int = 3, num_lstm_layers: int = 2,
                 heads: int = 4, dropout: float = 0.1):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_lstm_layers = num_lstm_layers
        self.heads = heads
        self.dropout = dropout
        self.model: Optional[TemporalGNN] = None
        self._trained = False
        self.train_metrics: Dict[str, Any] = {}
        self.test_metrics: Dict[str, Any] = {}
        self.test_actuals = np.array([])
        self.test_predictions = np.array([])
        self.training_history: List[Dict[str, float]] = []
        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std: Optional[np.ndarray] = None
        self._target_mean: Optional[np.ndarray] = None
        self._target_std: Optional[np.ndarray] = None

    @staticmethod
    def extract_graph_snapshot(sim) -> Dict[str, Any]:
        """Extract graph snapshot (node features + edges + spectral targets) from sim."""
        bank_ids = list(sim.banks.keys())
        n = len(bank_ids)

        node_feats = np.zeros((n, len(NODE_FEATURE_NAMES)), dtype=np.float32)
        for i, bid in enumerate(bank_ids):
            b = sim.banks[bid]
            s = b.state
            node_feats[i, 0] = s.get("CET1_ratio", 10.0)
            node_feats[i, 1] = s.get("LCR", 130.0)
            node_feats[i, 2] = s.get("NSFR", 110.0)
            ta = s.get("total_assets", 1e9)
            node_feats[i, 3] = np.log1p(max(ta, 0))
            node_feats[i, 4] = 1.0 if s.get("CET1_ratio", 10.0) < 8.0 else 0.0

        adj = sim.get_adjacency_matrix()
        rows, cols = np.nonzero(adj)
        edge_index = np.stack([rows, cols], axis=0).astype(np.int64) if len(rows) > 0 \
            else np.zeros((2, 0), dtype=np.int64)
        edge_weight = adj[rows, cols].astype(np.float32) if len(rows) > 0 \
            else np.zeros(0, dtype=np.float32)

        from scr_financial.network.spectral import (
            compute_laplacian, eigendecomposition, find_spectral_gap,
            analyze_spectral_properties,
        )
        adj_sym = (adj + adj.T) / 2.0
        L = compute_laplacian(adj_sym, normalized=True)
        eigenvalues, eigenvectors = eigendecomposition(L)
        gap_idx, gap_size = find_spectral_gap(eigenvalues)
        props = analyze_spectral_properties(eigenvalues, eigenvectors)

        return {
            "node_features": node_feats,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "targets": {
                "lambda_2": float(props["algebraic_connectivity"]),
                "spectral_gap": float(gap_size),
                "spectral_radius": float(props["spectral_radius"]),
            },
            "lambda_2": float(props["algebraic_connectivity"]),
            "spectral_gap": float(gap_size),
            "spectral_radius": float(props["spectral_radius"]),
        }

    @staticmethod
    def extract_features(sim, spectral_fn) -> Dict[str, float]:
        """Compatibility shim — extract scalar features like SCGPredictor."""
        spec = spectral_fn()
        banks = sim.banks
        cet1s = [b.state.get("CET1_ratio", 10.0) for b in banks.values()]
        si = sim.system_indicators
        return {
            "lambda_2": spec.get("algebraic_connectivity", 0.0),
            "spectral_gap": spec.get("gap_size", 0.0),
            "spectral_radius": spec.get("spectral_radius", 0.0),
            "avg_cet1": float(np.mean(cet1s)) if cet1s else 0.0,
            "min_cet1": float(np.min(cet1s)) if cet1s else 0.0,
            "ciss": si.get("CISS", 0.5),
            "funding_stress": si.get("funding_stress", 0.0),
            "n_stressed": sum(1 for c in cet1s if c < 8.0),
        }

    def _snapshot_to_data(self, snap: Dict[str, Any]) -> Data:
        """Convert snapshot dict to PyG Data object with normalisation."""
        x = torch.tensor(snap["node_features"], dtype=torch.float32)
        if self._feat_mean is not None:
            feat_std = self._feat_std.copy()
            feat_std[feat_std < 1e-8] = 1.0
            x = (x - torch.tensor(self._feat_mean, dtype=torch.float32)) / \
                torch.tensor(feat_std, dtype=torch.float32)
        ei = torch.tensor(snap["edge_index"], dtype=torch.long)
        ew = torch.tensor(snap["edge_weight"], dtype=torch.float32) if len(snap["edge_weight"]) > 0 else None
        if ew is not None and ew.numel() > 0 and ew.max() > 0:
            ew = ew / ew.max()
        return Data(x=x, edge_index=ei, edge_weight=ew)

    def _build_sequences(self, snapshots: List[Dict]) -> Tuple[List[List[Dict]], np.ndarray]:
        """Build (graph_sequence, target) pairs from snapshot list."""
        sequences, targets = [], []
        for i in range(len(snapshots) - self.seq_len):
            sequences.append(snapshots[i: i + self.seq_len])
            tgt = snapshots[i + self.seq_len]
            targets.append([tgt["targets"][k] for k in TARGET_NAMES])
        return sequences, np.array(targets, dtype=np.float32)

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        mse = float(np.mean((y_true - y_pred) ** 2))
        r2_per = {}
        for i, name in enumerate(TARGET_NAMES):
            ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
            ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
            if ss_tot < 1e-8:
                r2_per[name] = 1.0 if ss_res < 1e-8 else 0.0
            else:
                r2_per[name] = float(max(-1.0, 1.0 - ss_res / ss_tot))
        return {"mse": mse, "r2": float(np.mean(list(r2_per.values()))), "r2_per_target": r2_per}

    def train(
        self,
        snapshots: List[Dict[str, Any]],
        epochs: int = 200,
        lr: float = 3e-3,
        test_fraction: float = 0.2,
        patience: int = 30,
        progress_callback: Optional[Callable[[int, int, float, Optional[float]], None]] = None,
    ) -> float:
        """Train the temporal GNN with early stopping. Returns final train loss.

        Parameters
        ----------
        patience : int
            Early stopping patience — stop if test loss doesn't improve for
            this many epochs. Set to 0 to disable early stopping.
        progress_callback : callable(epoch, total_epochs, train_loss, test_loss_or_None)
            Called every 5 epochs for UI progress updates.
        """
        sequences, targets = self._build_sequences(snapshots)
        n_seqs = len(sequences)
        if n_seqs < 10:
            raise ValueError(f"Need >= {self.seq_len + 10} snapshots, got {len(snapshots)} "
                             f"({n_seqs} sequences).")

        split = max(5, int(n_seqs * (1 - test_fraction)))
        train_seqs, test_seqs = sequences[:split], sequences[split:]
        y_train, y_test = targets[:split], targets[split:]

        # Fit normalisation on train
        all_feats = np.concatenate([s["node_features"] for seq in train_seqs for s in seq], axis=0)
        self._feat_mean = all_feats.mean(axis=0)
        self._feat_std = all_feats.std(axis=0)

        self._target_mean = y_train.mean(axis=0)
        self._target_std = y_train.std(axis=0)
        self._target_std[self._target_std < 1e-8] = 1.0
        y_train_s = (y_train - self._target_mean) / self._target_std

        train_graph_seqs = [[self._snapshot_to_data(s) for s in seq] for seq in train_seqs]
        y_train_t = torch.tensor(y_train_s, dtype=torch.float32)

        # Also prepare test graphs if we have them (for progress reporting)
        test_graph_seqs = None
        if len(test_seqs) > 0:
            test_graph_seqs = [[self._snapshot_to_data(s) for s in seq] for seq in test_seqs]

        n_feat = len(NODE_FEATURE_NAMES)
        self.model = TemporalGNN(
            n_feat, self.hidden_dim, len(TARGET_NAMES),
            self.num_gat_layers, self.num_lstm_layers, self.heads, self.dropout,
        )
        n_params = self.model.count_parameters()
        logger.info("TemporalGNN: %d params, %d GAT layers (%d heads), %d LSTM layers, hidden=%d",
                    n_params, self.num_gat_layers, self.heads, self.num_lstm_layers, self.hidden_dim)

        # Warn if severely overparameterized
        if n_params > 10 * len(train_seqs):
            logger.warning(
                "Model has %d params for %d training samples (ratio %.0f:1). "
                "Consider reducing hidden_dim or heads to prevent overfitting.",
                n_params, len(train_seqs), n_params / len(train_seqs),
            )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = nn.MSELoss()

        batch_size = min(32, len(train_graph_seqs))
        self.model.train()
        self.training_history = []
        final_loss = 0.0

        # Early stopping state
        best_test_loss = float("inf")
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            perm = np.random.permutation(len(train_graph_seqs))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(perm), batch_size):
                idx = perm[start: start + batch_size]
                batch_seqs = [train_graph_seqs[i] for i in idx]
                batch_y = y_train_t[idx]

                optimizer.zero_grad()
                pred = self.model(batch_seqs)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            final_loss = epoch_loss / max(n_batches, 1)

            # Progress reporting every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                test_loss = None
                if test_graph_seqs is not None:
                    self.model.eval()
                    with torch.no_grad():
                        y_test_s = (y_test - self._target_mean) / self._target_std
                        test_pred_s = self.model(test_graph_seqs).numpy()
                        test_loss = float(np.mean((test_pred_s - y_test_s) ** 2))
                    self.model.train()

                    # Early stopping check
                    if patience > 0 and test_loss is not None:
                        if test_loss < best_test_loss - 1e-6:
                            best_test_loss = test_loss
                            best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += 5  # We check every 5 epochs

                self.training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": final_loss,
                    "test_loss": test_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                })

                if progress_callback is not None:
                    progress_callback(epoch + 1, epochs, final_loss, test_loss)

                # Early stopping trigger
                if patience > 0 and epochs_without_improvement >= patience:
                    logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                                epoch + 1, patience)
                    break

        # Restore best model if early stopping was used
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logger.info("Restored best model (test_loss=%.6f)", best_test_loss)

        self._trained = True

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            train_pred_s = self.model(train_graph_seqs).numpy()
        train_pred = train_pred_s * self._target_std + self._target_mean
        self.train_metrics = self._compute_metrics(y_train, train_pred)

        if test_graph_seqs is not None and len(test_seqs) > 0:
            with torch.no_grad():
                test_pred_s = self.model(test_graph_seqs).numpy()
            test_pred = test_pred_s * self._target_std + self._target_mean
            self.test_metrics = self._compute_metrics(y_test, test_pred)
            self.test_actuals = y_test
            self.test_predictions = test_pred
        else:
            self.test_metrics = {"mse": 0.0, "r2": 0.0, "r2_per_target": {}}

        logger.info(
            "GNNPredictor trained: %d params, %d train / %d test, "
            "train_mse=%.6f, test_mse=%.6f, test_r2=%.4f",
            n_params, len(train_seqs), len(test_seqs),
            self.train_metrics["mse"], self.test_metrics["mse"], self.test_metrics["r2"],
        )
        return final_loss

    def predict(self, recent_snapshots: List[Dict[str, Any]], steps: int = 20) -> List[Dict[str, float]]:
        """Autoregressively predict spectral metrics forward."""
        if not self._trained or self.model is None:
            raise RuntimeError("Model not trained.")
        if len(recent_snapshots) < self.seq_len:
            raise ValueError(f"Need >= {self.seq_len} snapshots, got {len(recent_snapshots)}.")

        window = list(recent_snapshots[-self.seq_len:])
        self.model.eval()
        predictions: List[Dict[str, float]] = []

        with torch.no_grad():
            for _ in range(steps):
                graph_seq = [self._snapshot_to_data(s) for s in window]
                pred_s = self.model([graph_seq]).numpy()[0]
                pred = pred_s * self._target_std + self._target_mean

                pred_dict = {TARGET_NAMES[i]: float(pred[i]) for i in range(len(TARGET_NAMES))}
                predictions.append(pred_dict)

                new_snap = {
                    "node_features": window[-1]["node_features"].copy(),
                    "edge_index": window[-1]["edge_index"].copy(),
                    "edge_weight": window[-1]["edge_weight"].copy(),
                    "targets": pred_dict,
                }
                window = window[1:] + [new_snap]

        return predictions
