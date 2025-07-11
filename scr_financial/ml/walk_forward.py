"""Walk-forward cross-validation for temporal graph sequences."""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class WalkForwardCV:
    """Expanding-window walk-forward cross-validation.

    Produces splits where the training window grows and a fixed-size
    test window follows after a gap to prevent data leakage.
    """

    def __init__(self, n_splits: int = 5, min_train_size: int = 100,
                 test_size: int = 40, gap_size: int = 10):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.gap_size = gap_size

    def split(self, n_total: int):
        """Yield (train_indices, test_indices) for each fold."""
        max_test_end = n_total
        available = max_test_end - self.min_train_size - self.gap_size - self.test_size
        if available <= 0:
            raise ValueError(f"Not enough data ({n_total}) for {self.n_splits} splits "
                             f"with min_train={self.min_train_size}, gap={self.gap_size}, "
                             f"test={self.test_size}")

        step = max(1, available // self.n_splits)
        for fold in range(self.n_splits):
            train_end = self.min_train_size + fold * step
            test_start = train_end + self.gap_size
            test_end = min(test_start + self.test_size, n_total)
            if test_end <= test_start:
                break
            train_idx = list(range(0, train_end))
            test_idx = list(range(test_start, test_end))
            yield train_idx, test_idx

    def __repr__(self):
        return (f"WalkForwardCV(n_splits={self.n_splits}, min_train={self.min_train_size}, "
                f"test={self.test_size}, gap={self.gap_size})")


def walk_forward_evaluate(
    predictor_class,
    predictor_kwargs: Dict,
    snapshots: List[Dict],
    n_splits: int = 5,
    min_train_size: int = 100,
    test_size: int = 40,
    gap_size: int = 10,
    epochs: int = 200,
    lr: float = 3e-3,
    patience: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run walk-forward CV on a GNNPredictor-compatible model."""
    cv = WalkForwardCV(n_splits, min_train_size, test_size, gap_size)
    n_total = len(snapshots)

    fold_results = []
    all_test_preds = []
    all_test_actuals = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(n_total)):
        logger.info("Fold %d/%d: train=%d [%d-%d], test=%d [%d-%d]",
                     fold_i + 1, n_splits, len(train_idx), train_idx[0], train_idx[-1],
                     len(test_idx), test_idx[0], test_idx[-1])

        train_snaps = [snapshots[i] for i in train_idx]
        test_snaps = [snapshots[i] for i in test_idx]

        np.random.seed(seed + fold_i)
        try:
            import torch
            torch.manual_seed(seed + fold_i)
        except ImportError:
            pass

        predictor = predictor_class(**predictor_kwargs)
        t0 = time.time()

        # Train on this fold's training data
        # Combine train + test for the predictor's internal split
        # But ensure test_fraction captures the actual test portion
        all_fold_snaps = train_snaps + test_snaps
        test_frac = len(test_snaps) / len(all_fold_snaps)

        try:
            predictor.train(all_fold_snaps, epochs=epochs, lr=lr,
                           test_fraction=test_frac, patience=patience)
        except Exception as e:
            logger.warning("Fold %d failed: %s", fold_i + 1, e)
            continue

        train_time = time.time() - t0

        fold_result = {
            "fold": fold_i,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "train_r2": predictor.train_metrics.get("r2", 0),
            "test_r2": predictor.test_metrics.get("r2", 0),
            "test_mse": predictor.test_metrics.get("mse", 0),
            "test_r2_per_target": predictor.test_metrics.get("r2_per_target", {}),
            "training_time_s": round(train_time, 1),
            "epochs_trained": predictor.training_history[-1]["epoch"] if predictor.training_history else 0,
        }
        fold_results.append(fold_result)

        if len(predictor.test_actuals) > 0:
            all_test_preds.append(predictor.test_predictions)
            all_test_actuals.append(predictor.test_actuals)

        logger.info("  Fold %d: test_r2=%.4f, test_mse=%.6f, time=%.1fs",
                     fold_i + 1, fold_result["test_r2"], fold_result["test_mse"], train_time)

    # Aggregate
    if fold_results:
        test_r2s = [r["test_r2"] for r in fold_results]
        test_mses = [r["test_mse"] for r in fold_results]
        aggregate = {
            "test_r2_mean": float(np.mean(test_r2s)),
            "test_r2_std": float(np.std(test_r2s)),
            "test_r2_median": float(np.median(test_r2s)),
            "test_mse_mean": float(np.mean(test_mses)),
            "test_mse_std": float(np.std(test_mses)),
            "n_folds_completed": len(fold_results),
        }
    else:
        aggregate = {"test_r2_mean": 0, "test_r2_std": 0, "n_folds_completed": 0}

    return {
        "per_fold": fold_results,
        "aggregate": aggregate,
        "cv_config": {"n_splits": n_splits, "min_train_size": min_train_size,
                      "test_size": test_size, "gap_size": gap_size},
    }
