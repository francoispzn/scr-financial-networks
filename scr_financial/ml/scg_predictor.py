"""
SCG-aware predictor for spectral metric evolution.

Wraps the existing LSTMPredictor to forecast spectral features
(algebraic connectivity, spectral gap, spectral radius) from
stochastic ABM trajectories.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .predictors import LSTMPredictor

logger = logging.getLogger(__name__)

# Feature columns extracted per ABM step
FEATURE_NAMES = [
    "lambda_2",          # algebraic connectivity
    "spectral_gap",      # gap size δ
    "spectral_radius",   # largest eigenvalue
    "avg_cet1",          # mean CET1 across banks
    "min_cet1",          # worst CET1
    "ciss",              # composite indicator of systemic stress
    "funding_stress",    # funding stress indicator
    "n_stressed",        # count of banks with CET1 < 8%
]

TARGET_NAMES = ["lambda_2", "spectral_gap", "spectral_radius"]


class SCGPredictor:
    """Train an LSTM on ABM-generated spectral trajectories and predict forward."""

    def __init__(self, seq_len: int = 10, hidden_dim: int = 32, num_layers: int = 2):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model: Optional[LSTMPredictor] = None
        self._trained = False

    # ── Feature extraction ────────────────────────────────────────────────

    @staticmethod
    def extract_features(sim, spectral_fn) -> Dict[str, float]:
        """Extract one feature vector from the current simulation state.

        Parameters
        ----------
        sim : BankingSystemSimulation
        spectral_fn : callable returning spectral data dict (eigenvalues, gap_size, etc.)
        """
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

    # ── Training ──────────────────────────────────────────────────────────

    def build_training_data(
        self, feature_history: List[Dict[str, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a list of per-step feature dicts into (X, y) arrays.

        X: shape (samples, seq_len, n_features)
        y: shape (samples, n_targets)
        """
        mat = np.array([[row[f] for f in FEATURE_NAMES] for row in feature_history])
        targets = np.array([[row[f] for f in TARGET_NAMES] for row in feature_history])

        X_seqs, y_seqs = [], []
        for i in range(len(mat) - self.seq_len):
            X_seqs.append(mat[i : i + self.seq_len])
            y_seqs.append(targets[i + self.seq_len])

        return np.array(X_seqs), np.array(y_seqs)

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute MSE, R² per target, and overall R²."""
        mse = float(np.mean((y_true - y_pred) ** 2))
        r2_per = {}
        for i, name in enumerate(TARGET_NAMES):
            ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
            ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
            # If target has negligible variance, R² is meaningless —
            # report 1.0 if predictions also have negligible error, else 0.0
            if ss_tot < 1e-8:
                r2_per[name] = 1.0 if ss_res < 1e-8 else 0.0
            else:
                r2_per[name] = float(max(-1.0, 1.0 - ss_res / ss_tot))
        r2_avg = float(np.mean(list(r2_per.values())))
        return {"mse": mse, "r2": r2_avg, "r2_per_target": r2_per}

    def train(
        self,
        feature_history: List[Dict[str, float]],
        epochs: int = 60,
        lr: float = 1e-3,
        test_fraction: float = 0.2,
    ) -> float:
        """Train the LSTM with chronological train/test split. Returns final train loss."""
        X_raw, y_raw = self.build_training_data(feature_history)
        if len(X_raw) < 10:
            raise ValueError("Need at least seq_len + 10 steps of history to train.")

        # Chronological split
        split = max(5, int(len(X_raw) * (1 - test_fraction)))
        X_train, X_test = X_raw[:split], X_raw[split:]
        y_train, y_test = y_raw[:split], y_raw[split:]

        # Scale (fit on train only)
        n_train, seq_len, n_feat = X_train.shape
        X_flat = X_train.reshape(-1, n_feat)
        self.scaler_X.fit(X_flat)
        self.scaler_y.fit(y_train)

        X_train_s = self.scaler_X.transform(X_train.reshape(-1, n_feat)).reshape(n_train, seq_len, n_feat)
        y_train_s = self.scaler_y.transform(y_train)

        X_t = torch.tensor(X_train_s, dtype=torch.float32)
        y_t = torch.tensor(y_train_s, dtype=torch.float32)

        self.model = LSTMPredictor(
            input_dim=n_feat,
            hidden_dim=self.hidden_dim,
            output_dim=len(TARGET_NAMES),
            num_layers=self.num_layers,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        self.model.train()
        final_loss = 0.0
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self._trained = True

        # Evaluate on train and test sets
        self.model.eval()
        with torch.no_grad():
            train_pred_s = self.model(X_t).numpy()
            train_pred = self.scaler_y.inverse_transform(train_pred_s)
        self.train_metrics = self._compute_metrics(y_train, train_pred)

        if len(X_test) > 0:
            n_test = X_test.shape[0]
            X_test_s = self.scaler_X.transform(X_test.reshape(-1, n_feat)).reshape(n_test, seq_len, n_feat)
            with torch.no_grad():
                test_pred_s = self.model(torch.tensor(X_test_s, dtype=torch.float32)).numpy()
                test_pred = self.scaler_y.inverse_transform(test_pred_s)
            self.test_metrics = self._compute_metrics(y_test, test_pred)
            # Store actuals and predictions for scatter plot
            self.test_actuals = y_test
            self.test_predictions = test_pred
        else:
            self.test_metrics = {"mse": 0.0, "r2": 0.0, "r2_per_target": {}}
            self.test_actuals = np.array([])
            self.test_predictions = np.array([])

        logger.info(
            "SCGPredictor trained: %d train / %d test, train_mse=%.6f, test_mse=%.6f, test_r2=%.4f",
            n_train, len(X_test), self.train_metrics["mse"],
            self.test_metrics["mse"], self.test_metrics["r2"],
        )
        return final_loss

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, recent_features: List[Dict[str, float]], steps: int = 20) -> List[Dict[str, float]]:
        """Autoregressively predict spectral metrics forward.

        Parameters
        ----------
        recent_features : list of dicts
            Must have at least ``seq_len`` entries.
        steps : int
            Number of future steps to predict.

        Returns
        -------
        list of dicts with keys from TARGET_NAMES
        """
        if not self._trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        mat = np.array([[row[f] for f in FEATURE_NAMES] for row in recent_features])
        if len(mat) < self.seq_len:
            raise ValueError(f"Need >= {self.seq_len} recent features, got {len(mat)}.")

        window = mat[-self.seq_len:].copy()
        self.model.eval()
        predictions: List[Dict[str, float]] = []

        with torch.no_grad():
            for _ in range(steps):
                w_scaled = self.scaler_X.transform(window.reshape(-1, window.shape[-1])).reshape(
                    1, self.seq_len, -1
                )
                inp = torch.tensor(w_scaled, dtype=torch.float32)
                pred_scaled = self.model(inp).numpy()[0]
                pred = self.scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]

                predictions.append({TARGET_NAMES[i]: float(pred[i]) for i in range(len(TARGET_NAMES))})

                # Shift window: use predicted targets for first 3 cols, carry forward the rest
                new_row = window[-1].copy()
                for i, name in enumerate(TARGET_NAMES):
                    feat_idx = FEATURE_NAMES.index(name)
                    new_row[feat_idx] = pred[i]
                window = np.vstack([window[1:], new_row.reshape(1, -1)])

        return predictions
