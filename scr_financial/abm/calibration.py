"""
ABM parameter calibration against empirical data.

Optimizes ABM parameters to match observed CET1 trajectories
from EBA stress tests or historical regulatory data.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ABMCalibrator:
    """Calibrate ABM parameters against empirical targets.

    Uses scipy.optimize.differential_evolution to find parameters
    that minimize the distance between simulated and target outcomes.
    """

    def __init__(
        self,
        build_and_run: Callable[[Dict[str, float]], Dict[str, float]],
        target_metrics: Dict[str, float],
        param_bounds: Dict[str, Tuple[float, float]],
        metric_weights: Optional[Dict[str, float]] = None,
        n_runs_per_eval: int = 20,
    ):
        """
        Args:
            build_and_run: Function mapping params -> simulated metrics dict.
            target_metrics: Target values to match (e.g., from EBA stress test).
            param_bounds: {param_name: (lower, upper)} for optimization.
            metric_weights: Optional weights for each metric in the loss.
            n_runs_per_eval: Monte Carlo runs per parameter evaluation.
        """
        self.build_and_run = build_and_run
        self.target_metrics = target_metrics
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.metric_weights = metric_weights or {k: 1.0 for k in target_metrics}
        self.n_runs = n_runs_per_eval
        self.eval_count = 0
        self.best_loss = float("inf")
        self.history: List[Dict] = []

    def _objective(self, x: np.ndarray) -> float:
        """Objective function for optimization."""
        params = {self.param_names[i]: x[i] for i in range(len(self.param_names))}

        # Monte Carlo average
        all_metrics: List[Dict[str, float]] = []
        for _ in range(self.n_runs):
            try:
                metrics = self.build_and_run(params)
                all_metrics.append(metrics)
            except Exception:
                pass

        if not all_metrics:
            return 1e6

        # Average metrics across runs
        avg_metrics = {}
        for key in self.target_metrics:
            vals = [m.get(key, 0.0) for m in all_metrics]
            avg_metrics[key] = np.mean(vals)

        # Weighted squared error
        loss = 0.0
        for key, target in self.target_metrics.items():
            if target != 0:
                # Relative error
                err = (avg_metrics.get(key, 0) - target) / abs(target)
            else:
                err = avg_metrics.get(key, 0) - target
            loss += self.metric_weights.get(key, 1.0) * err ** 2

        self.eval_count += 1
        if loss < self.best_loss:
            self.best_loss = loss
            logger.info("Calibration eval %d: loss=%.6f (best), params=%s",
                        self.eval_count, loss, {k: round(v, 4) for k, v in params.items()})

        self.history.append({"eval": self.eval_count, "loss": loss, "params": params, "metrics": avg_metrics})
        return loss

    def calibrate(
        self,
        maxiter: int = 50,
        popsize: int = 10,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run differential evolution optimization.

        Args:
            maxiter: Maximum generations.
            popsize: Population size multiplier.
            tol: Convergence tolerance.
            seed: Random seed.

        Returns:
            {optimal_params, loss, n_evals, history}
        """
        from scipy.optimize import differential_evolution

        bounds = [self.param_bounds[p] for p in self.param_names]

        logger.info("Starting calibration: %d parameters, %d target metrics, maxiter=%d",
                     len(self.param_names), len(self.target_metrics), maxiter)

        result = differential_evolution(
            self._objective,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            seed=seed,
            disp=False,
        )

        optimal_params = {self.param_names[i]: result.x[i] for i in range(len(self.param_names))}

        logger.info("Calibration complete: loss=%.6f, evals=%d", result.fun, self.eval_count)

        return {
            "optimal_params": optimal_params,
            "loss": float(result.fun),
            "n_evals": self.eval_count,
            "converged": result.success,
            "message": result.message,
            "history": self.history,
        }


# ── EBA stress test targets ──────────────────────────────────────

# Approximate targets from EBA 2023 adverse scenario results
# These are system-average CET1 impacts for major European banks
EBA_2023_ADVERSE_TARGETS = {
    "avg_cet1_decline": -4.6,      # Average CET1 decline (percentage points)
    "min_cet1": 7.2,                # Minimum system-average CET1 (%)
    "n_below_threshold": 2,         # Number of banks below 8% CET1
    "max_single_decline": -8.3,     # Worst single-bank CET1 decline (pp)
}

# ECB 2024 stress test targets
ECB_2024_TARGETS = {
    "avg_cet1_decline": -3.8,
    "min_cet1": 8.1,
    "n_below_threshold": 1,
}
