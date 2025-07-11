"""Ablation study framework for systematic component analysis."""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

TARGET_NAMES = ["lambda_2", "spectral_gap", "spectral_radius"]


def run_ablation_study(
    snapshots: List[Dict],
    base_config: Dict[str, Any],
    ablation_dims: Optional[Dict[str, List]] = None,
    n_seeds: int = 3,
    epochs: int = 200,
    patience: int = 30,
) -> Dict[str, Any]:
    """Run systematic ablation study across hyperparameter dimensions.

    Args:
        snapshots: Graph snapshot data.
        base_config: Default GNNPredictor kwargs.
        ablation_dims: {dim_name: [values_to_test]}. If None, uses defaults.
        n_seeds: Random seeds per configuration.
        epochs: Training epochs.
        patience: Early stopping patience.

    Returns:
        Dict with per-dimension results and aggregate comparison.
    """
    from scr_financial.ml.gnn_predictor import GNNPredictor

    if ablation_dims is None:
        ablation_dims = {
            "hidden_dim": [16, 32, 64, 128],
            "num_gat_layers": [1, 2, 3, 4],
            "heads": [1, 2, 4, 8],
            "seq_len": [5, 10, 15, 20],
            "dropout": [0.0, 0.1, 0.2, 0.3],
        }

    results = {}

    for dim_name, values in ablation_dims.items():
        logger.info("Ablation: %s = %s", dim_name, values)
        dim_results = []

        for val in values:
            config = base_config.copy()
            config[dim_name] = val
            seed_r2s = []

            for seed in range(n_seeds):
                np.random.seed(seed * 42)
                try:
                    import torch
                    torch.manual_seed(seed * 42)
                except ImportError:
                    pass

                try:
                    predictor = GNNPredictor(**config)
                    t0 = time.time()
                    predictor.train(snapshots, epochs=epochs, patience=patience)
                    train_time = time.time() - t0
                    seed_r2s.append(predictor.test_metrics.get("r2", 0))
                except Exception as e:
                    logger.warning("  %s=%s seed=%d failed: %s", dim_name, val, seed, e)
                    seed_r2s.append(0.0)

            dim_results.append({
                "value": val,
                "r2_mean": round(float(np.mean(seed_r2s)), 4),
                "r2_std": round(float(np.std(seed_r2s)), 4),
                "r2_values": [round(v, 4) for v in seed_r2s],
            })
            logger.info("  %s=%s: R²=%.4f±%.4f", dim_name, val,
                         dim_results[-1]["r2_mean"], dim_results[-1]["r2_std"])

        results[dim_name] = dim_results

    return results


def feature_ablation(
    snapshots: List[Dict],
    config: Dict[str, Any],
    feature_groups: Optional[Dict[str, List[int]]] = None,
    n_seeds: int = 3,
    epochs: int = 150,
    patience: int = 25,
) -> Dict[str, Any]:
    """Remove feature groups one at a time and measure impact.

    Args:
        feature_groups: {group_name: list of feature indices to REMOVE}.
    """
    if feature_groups is None:
        feature_groups = {
            "volatility": [0],
            "return": [1],
            "log_price": [2],
            "beta": [3],
            "momentum": [4],
        }

    from scr_financial.ml.gnn_predictor import GNNPredictor

    # Baseline (all features)
    baseline_r2s = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        try:
            import torch; torch.manual_seed(seed)
        except ImportError: pass
        p = GNNPredictor(**config)
        p.train(snapshots, epochs=epochs, patience=patience)
        baseline_r2s.append(p.test_metrics.get("r2", 0))

    results = {"baseline": {"r2_mean": round(float(np.mean(baseline_r2s)), 4),
                             "r2_std": round(float(np.std(baseline_r2s)), 4)}}

    for group_name, indices in feature_groups.items():
        # Zero out specified features in all snapshots
        modified = []
        for snap in snapshots:
            s = {k: v for k, v in snap.items()}
            feats = snap["node_features"].copy()
            feats[:, indices] = 0.0
            s["node_features"] = feats
            modified.append(s)

        group_r2s = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            try:
                import torch; torch.manual_seed(seed)
            except ImportError: pass
            p = GNNPredictor(**config)
            p.train(modified, epochs=epochs, patience=patience)
            group_r2s.append(p.test_metrics.get("r2", 0))

        r2_drop = np.mean(baseline_r2s) - np.mean(group_r2s)
        results[f"remove_{group_name}"] = {
            "r2_mean": round(float(np.mean(group_r2s)), 4),
            "r2_std": round(float(np.std(group_r2s)), 4),
            "r2_drop": round(float(r2_drop), 4),
            "importance": round(float(r2_drop / max(np.mean(baseline_r2s), 0.01)), 4),
        }
        logger.info("  Remove %s: R²=%.4f (drop=%.4f)", group_name,
                     np.mean(group_r2s), r2_drop)

    return results


def channel_ablation(
    build_and_run_fn,
    shock_scenario: Dict,
    n_runs: int = 50,
) -> Dict[str, Any]:
    """Remove ABM channels one at a time and measure impact on defaults."""
    configs = {
        "all_3_channels": {"disable_funding": False, "disable_firesale": False},
        "no_funding": {"disable_funding": True, "disable_firesale": False},
        "no_firesale": {"disable_funding": False, "disable_firesale": True},
        "credit_only": {"disable_funding": True, "disable_firesale": True},
    }

    results = {}
    for name, channel_config in configs.items():
        defaults = []
        for run_i in range(n_runs):
            metrics = build_and_run_fn(shock_scenario, seed=1000 + run_i, **channel_config)
            defaults.append(metrics.get("n_defaults", 0))

        results[name] = {
            "mean_defaults": round(float(np.mean(defaults)), 2),
            "std_defaults": round(float(np.std(defaults)), 2),
            "max_defaults": int(np.max(defaults)),
            "default_rate": round(float(np.mean([d > 0 for d in defaults])), 3),
        }
        logger.info("  %s: %.1f±%.1f defaults", name, np.mean(defaults), np.std(defaults))

    return results
