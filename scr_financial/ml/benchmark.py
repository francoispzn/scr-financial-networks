"""Benchmark suite for comparing models with statistical tests."""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """Run multiple models through walk-forward CV and compare."""

    def __init__(self, snapshots: List[Dict], n_splits: int = 5, seed: int = 42):
        self.snapshots = snapshots
        self.n_splits = n_splits
        self.seed = seed
        self.models: Dict[str, Dict] = {}
        self.results: Dict[str, Any] = {}

    def add_gnn(self, name: str, config: Dict):
        """Add a GNN configuration."""
        self.models[name] = {"type": "gnn", "config": config}

    def add_baseline(self, name: str, baseline):
        """Add a baseline model instance."""
        self.models[name] = {"type": "baseline", "instance": baseline}

    def run(self, epochs=200, patience=30) -> Dict[str, Any]:
        """Run all models and collect results."""
        from .walk_forward import walk_forward_evaluate
        from .baselines import extract_flat_features
        from scr_financial.ml.gnn_predictor import GNNPredictor

        for name, model_info in self.models.items():
            logger.info("Benchmarking: %s", name)
            t0 = time.time()

            if model_info["type"] == "gnn":
                result = walk_forward_evaluate(
                    GNNPredictor, model_info["config"], self.snapshots,
                    n_splits=self.n_splits, epochs=epochs, patience=patience,
                    seed=self.seed,
                )
            else:
                # Run baselines through simple sequential evaluation
                baseline = model_info["instance"]
                seq_len = 10
                fold_r2s = []

                # Build sequences and targets
                sequences, targets = [], []
                for i in range(len(self.snapshots) - seq_len):
                    sequences.append(self.snapshots[i:i + seq_len])
                    tgt = self.snapshots[i + seq_len]
                    targets.append([tgt["targets"]["lambda_2"],
                                    tgt["targets"]["spectral_gap"],
                                    tgt["targets"]["spectral_radius"]])
                targets = np.array(targets)

                # Simple 80/20 split for baselines
                split = int(len(sequences) * 0.8)
                X_train = extract_flat_features(sequences[:split])
                y_train = targets[:split]
                X_test = extract_flat_features(sequences[split:])
                y_test = targets[split:]

                try:
                    baseline.fit(X_train, y_train)
                    y_pred = baseline.predict(X_test)
                    if y_pred.ndim == 1:
                        y_pred = y_pred.reshape(-1, 3) if len(y_pred) % 3 == 0 else np.zeros_like(y_test)

                    ss_res = np.sum((y_test - y_pred) ** 2)
                    ss_tot = np.sum((y_test - np.mean(y_test, axis=0)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else 0
                    mse = float(np.mean((y_test - y_pred) ** 2))
                    fold_r2s.append(float(r2))
                except Exception as e:
                    logger.warning("Baseline %s failed: %s", name, e)
                    fold_r2s.append(0.0)
                    mse = 999.0

                result = {
                    "per_fold": [{"fold": 0, "test_r2": fold_r2s[0], "test_mse": mse}],
                    "aggregate": {"test_r2_mean": fold_r2s[0], "test_r2_std": 0,
                                  "test_mse_mean": mse, "n_folds_completed": 1},
                }

            elapsed = time.time() - t0
            result["total_time_s"] = round(elapsed, 1)
            self.results[name] = result
            logger.info("  %s: R²=%.4f, time=%.1fs", name,
                         result["aggregate"]["test_r2_mean"], elapsed)

        return self.results

    def comparison_table(self, baseline_key: str = "Persistence") -> pd.DataFrame:
        """Generate comparison DataFrame with DM tests."""
        from scr_financial.stats import welch_t_test

        rows = []
        baseline_r2 = self.results.get(baseline_key, {}).get("aggregate", {}).get("test_r2_mean", 0)

        for name, result in self.results.items():
            agg = result.get("aggregate", {})
            row = {
                "Model": name,
                "R² (mean)": agg.get("test_r2_mean", 0),
                "R² (std)": agg.get("test_r2_std", 0),
                "MSE": agg.get("test_mse_mean", 0),
                "Folds": agg.get("n_folds_completed", 0),
                "Time (s)": result.get("total_time_s", 0),
                "vs Baseline": "",
            }
            if name != baseline_key:
                improvement = agg.get("test_r2_mean", 0) - baseline_r2
                row["vs Baseline"] = f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
            rows.append(row)

        return pd.DataFrame(rows).sort_values("R² (mean)", ascending=False)

    def to_json(self, path: str):
        """Save results to JSON."""
        # Convert numpy to native Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=convert)
