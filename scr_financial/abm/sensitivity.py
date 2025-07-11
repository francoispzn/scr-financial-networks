"""
Parameter sensitivity analysis for the ABM.

Implements one-at-a-time (OAT) sweeps and Sobol global sensitivity indices
to quantify how ABM parameters affect systemic outcomes.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def oat_sweep(
    build_and_run: Callable[[Dict[str, float]], Dict[str, float]],
    base_params: Dict[str, float],
    param_ranges: Dict[str, Tuple[float, float, int]],
    n_runs: int = 50,
) -> Dict[str, pd.DataFrame]:
    """One-at-a-time parameter sweep.

    For each parameter, vary it across its range while holding others at base values.
    Run n_runs Monte Carlo simulations at each setting.

    Args:
        build_and_run: Function mapping params -> outcome metrics dict.
        base_params: Baseline parameter values.
        param_ranges: {param_name: (low, high, n_steps)}.
        n_runs: Monte Carlo runs per setting.

    Returns:
        {param_name: DataFrame with columns [param_value, metric_mean, metric_std, ...]}
    """
    results = {}

    for param_name, (low, high, n_steps) in param_ranges.items():
        logger.info("OAT sweep: %s [%.3f, %.3f] (%d steps × %d runs)",
                     param_name, low, high, n_steps, n_runs)
        values = np.linspace(low, high, n_steps)
        rows = []

        for val in values:
            params = base_params.copy()
            params[param_name] = val
            metrics_list = []
            for _ in range(n_runs):
                try:
                    metrics = build_and_run(params)
                    metrics_list.append(metrics)
                except Exception as e:
                    logger.warning("Run failed for %s=%.3f: %s", param_name, val, e)

            if metrics_list:
                row = {"param_value": val}
                for key in metrics_list[0]:
                    vals = [m[key] for m in metrics_list]
                    row[f"{key}_mean"] = np.mean(vals)
                    row[f"{key}_std"] = np.std(vals)
                    row[f"{key}_p10"] = np.percentile(vals, 10)
                    row[f"{key}_p90"] = np.percentile(vals, 90)
                rows.append(row)

        results[param_name] = pd.DataFrame(rows)

    return results


def sobol_sensitivity(
    build_and_run: Callable[[Dict[str, float]], Dict[str, float]],
    param_bounds: Dict[str, Tuple[float, float]],
    n_samples: int = 256,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Sobol global sensitivity analysis.

    Uses Saltelli sampling to compute first-order (S1) and total-order (ST)
    Sobol indices for each parameter.

    Falls back to a simpler Morris screening method if SALib is not available.

    Args:
        build_and_run: Function mapping params -> metrics dict.
        param_bounds: {param_name: (lower, upper)}.
        n_samples: Base sample size (total evaluations ≈ n_samples * (2*D + 2)).
        seed: Random seed.

    Returns:
        {metric_name: {param_name: {S1, S1_conf, ST, ST_conf}}}
    """
    param_names = list(param_bounds.keys())
    D = len(param_names)

    try:
        from SALib.sample import saltelli as saltelli_sample
        from SALib.analyze import sobol as sobol_analyze

        problem = {
            "num_vars": D,
            "names": param_names,
            "bounds": [param_bounds[p] for p in param_names],
        }

        X = saltelli_sample.sample(problem, n_samples, calc_second_order=False, seed=seed)
        logger.info("Sobol: %d evaluations for %d parameters", len(X), D)

        # Run all parameter combinations
        all_metrics: List[Dict[str, float]] = []
        for i, x_row in enumerate(X):
            params = {param_names[j]: x_row[j] for j in range(D)}
            try:
                metrics = build_and_run(params)
                all_metrics.append(metrics)
            except Exception:
                all_metrics.append({})

            if (i + 1) % 100 == 0:
                logger.info("  Sobol progress: %d/%d evaluations", i + 1, len(X))

        # Analyze per metric
        metric_names = list(all_metrics[0].keys()) if all_metrics else []
        results = {}
        for metric in metric_names:
            Y = np.array([m.get(metric, 0.0) for m in all_metrics])
            if np.std(Y) < 1e-10:
                continue
            Si = sobol_analyze.analyze(problem, Y, calc_second_order=False)
            results[metric] = {
                param_names[j]: {
                    "S1": float(Si["S1"][j]),
                    "S1_conf": float(Si["S1_conf"][j]),
                    "ST": float(Si["ST"][j]),
                    "ST_conf": float(Si["ST_conf"][j]),
                }
                for j in range(D)
            }
        return results

    except ImportError:
        logger.warning("SALib not available. Using Morris screening fallback.")
        return _morris_screening(build_and_run, param_bounds, n_samples, seed)


def _morris_screening(
    build_and_run: Callable,
    param_bounds: Dict[str, Tuple[float, float]],
    n_trajectories: int = 20,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Simple Morris screening as fallback when SALib is unavailable.

    Computes elementary effects for each parameter to rank importance.
    """
    rng = np.random.default_rng(seed)
    param_names = list(param_bounds.keys())
    D = len(param_names)
    delta = 0.5  # Step size as fraction of range

    all_effects: Dict[str, Dict[str, List[float]]] = {}

    for traj in range(n_trajectories):
        # Random base point
        base = {p: rng.uniform(lo, hi) for p, (lo, hi) in param_bounds.items()}
        base_metrics = build_and_run(base)

        for p in param_names:
            lo, hi = param_bounds[p]
            perturbed = base.copy()
            perturbed[p] = min(hi, base[p] + delta * (hi - lo))
            pert_metrics = build_and_run(perturbed)

            for metric in base_metrics:
                key = metric
                if key not in all_effects:
                    all_effects[key] = {pp: [] for pp in param_names}
                effect = (pert_metrics.get(metric, 0) - base_metrics[metric]) / (delta * (hi - lo))
                all_effects[key][p].append(effect)

    # Summarize
    results = {}
    for metric, effects in all_effects.items():
        results[metric] = {}
        for p, effs in effects.items():
            mu_star = float(np.mean(np.abs(effs)))
            sigma = float(np.std(effs))
            results[metric][p] = {"S1": mu_star, "S1_conf": sigma, "ST": mu_star, "ST_conf": sigma}
    return results


def tornado_data(
    oat_results: Dict[str, pd.DataFrame],
    metric: str = "n_defaults_mean",
    base_value: float = 0.0,
) -> pd.DataFrame:
    """Prepare data for tornado diagram from OAT results.

    Returns DataFrame with columns: [param, low_value, high_value, low_metric, high_metric, swing]
    sorted by swing (largest first).
    """
    rows = []
    for param_name, df in oat_results.items():
        if metric not in df.columns:
            continue
        vals = df["param_value"].values
        metrics = df[metric].values
        rows.append({
            "param": param_name,
            "low_value": vals[0],
            "high_value": vals[-1],
            "low_metric": metrics[0],
            "high_metric": metrics[-1],
            "swing": abs(metrics[-1] - metrics[0]),
        })

    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)


# ── Default ABM parameter ranges for sensitivity analysis ────────

ABM_BASE_PARAMS = {
    "lgd": 0.40,
    "funding_lcr_threshold": 100.0,
    "funding_sensitivity": 0.05,
    "funding_withdrawal_rate": 0.03,
    "fire_sale_haircut": 0.02,
    "fire_sale_amplifier": 1.5,
    "ou_theta_cet1": 0.05,
    "ou_sigma_cet1": 0.25,
    "ou_theta_lcr": 0.04,
    "ou_sigma_lcr": 1.5,
}

ABM_PARAM_RANGES = {
    "lgd": (0.20, 0.60, 5),
    "funding_lcr_threshold": (80.0, 120.0, 5),
    "funding_sensitivity": (0.01, 0.10, 5),
    "funding_withdrawal_rate": (0.01, 0.08, 5),
    "fire_sale_haircut": (0.005, 0.05, 5),
    "fire_sale_amplifier": (1.0, 3.0, 5),
    "ou_theta_cet1": (0.01, 0.10, 5),
    "ou_sigma_cet1": (0.10, 0.50, 5),
}
