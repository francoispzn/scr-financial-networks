"""Baseline models for spectral property prediction."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

TARGET_NAMES = ["lambda_2", "spectral_gap", "spectral_radius"]


class PersistenceBaseline:
    """Naive baseline: y_{t+1} = y_t."""
    def fit(self, X, y): pass
    def predict(self, X):
        return np.array(X)[:, :3] if np.array(X).shape[1] >= 3 else np.zeros((len(X), 3))
    def get_name(self): return "Persistence"


class MovingAverageBaseline:
    """Moving average: y_{t+1} = mean(y_{t-w:t})."""
    def __init__(self, window=5):
        self.window = window
        self._history = None

    def fit(self, X, y):
        self._history = np.array(y)

    def predict(self, X):
        if self._history is None or len(self._history) < self.window:
            return np.zeros((len(X), 3))
        last_w = self._history[-self.window:]
        pred = np.mean(last_w, axis=0)
        return np.tile(pred, (len(X), 1))

    def get_name(self): return f"MA({self.window})"


class ARIMABaseline:
    """Per-target ARIMA (falls back to AR(1) if statsmodels unavailable)."""
    def __init__(self, order=(1, 0, 0)):
        self.order = order
        self._models = {}

    def fit(self, X, y):
        y = np.array(y)
        for i, name in enumerate(TARGET_NAMES):
            series = y[:, i]
            try:
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(series, order=self.order).fit()
                self._models[name] = ("arima", model)
            except Exception:
                # AR(1) fallback: y_t = a + b * y_{t-1}
                if len(series) > 2:
                    X_ar = series[:-1].reshape(-1, 1)
                    y_ar = series[1:]
                    X_aug = np.column_stack([np.ones(len(X_ar)), X_ar])
                    try:
                        coefs = np.linalg.lstsq(X_aug, y_ar, rcond=None)[0]
                        self._models[name] = ("ar1", coefs, series[-1])
                    except Exception:
                        self._models[name] = ("const", np.mean(series))

    def predict(self, X):
        preds = np.zeros((len(X), 3))
        for i, name in enumerate(TARGET_NAMES):
            if name not in self._models:
                continue
            entry = self._models[name]
            if entry[0] == "arima":
                try:
                    fc = entry[1].forecast(steps=len(X))
                    preds[:, i] = fc
                except Exception:
                    preds[:, i] = entry[1].predict(start=0, end=0)[0]
            elif entry[0] == "ar1":
                coefs, last_val = entry[1], entry[2]
                for j in range(len(X)):
                    pred = coefs[0] + coefs[1] * last_val
                    preds[j, i] = pred
                    last_val = pred
            elif entry[0] == "const":
                preds[:, i] = entry[1]
        return preds

    def get_name(self): return f"ARIMA{self.order}"


class RandomForestBaseline:
    """Random Forest on flat features."""
    def __init__(self, n_estimators=100, seed=42):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        self.model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1))

    def fit(self, X, y):
        self.model.fit(np.array(X), np.array(y))

    def predict(self, X):
        return self.model.predict(np.array(X))

    def get_name(self): return "RandomForest"


class GradientBoostingBaseline:
    """Gradient Boosting on flat features."""
    def __init__(self, n_estimators=200, seed=42):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        self.model = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=n_estimators, random_state=seed, max_depth=4))

    def fit(self, X, y):
        self.model.fit(np.array(X), np.array(y))

    def predict(self, X):
        return self.model.predict(np.array(X))

    def get_name(self): return "GradientBoosting"


def extract_flat_features(sequences):
    """Convert graph-sequence data into flat feature vectors for sklearn baselines.

    For each sequence of snapshots, extract 14 features:
    - Last value of each spectral target (3)
    - Mean of each target over sequence (3)
    - Std of each target (3)
    - Trend (last - first) of each target (3)
    - Mean volatility across banks (1)
    - Mean number of edges (1)
    """
    features = []
    for seq in sequences:
        if isinstance(seq, list):
            targets = [s.get("targets", s) for s in seq]
            tgt_arr = np.array([[t.get("lambda_2", 0), t.get("spectral_gap", 0),
                                  t.get("spectral_radius", 0)] for t in targets])
        else:
            tgt_arr = np.array(seq).reshape(1, -1)[:, :3]

        last = tgt_arr[-1]
        mean = np.mean(tgt_arr, axis=0)
        std = np.std(tgt_arr, axis=0)
        trend = tgt_arr[-1] - tgt_arr[0]

        # Additional features from node features if available
        avg_vol = 0.0
        avg_edges = 0.0
        if isinstance(seq, list) and "node_features" in seq[-1]:
            avg_vol = float(np.mean(seq[-1]["node_features"][:, 0]))
            avg_edges = seq[-1]["edge_index"].shape[1] / 2 if seq[-1]["edge_index"].shape[1] > 0 else 0

        feat = np.concatenate([last, mean, std, trend, [avg_vol, avg_edges]])
        features.append(feat)
    return np.array(features)
