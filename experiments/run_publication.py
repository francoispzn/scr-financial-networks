#!/usr/bin/env python3
"""Publication experiment suite -- produces all results for the journal paper.

Run with:
    cd /Users/francoispetizon/scr-financial-networks
    /opt/anaconda3/envs/systemic_risk/bin/python experiments/run_publication.py

Experiments
-----------
 1. Multi-scale SCG validation (N=10, 50 synthetic networks, multiple topologies)
 2. RMT denoising comparison (raw vs denoised correlation, spectral properties)
 3. Alternative network construction comparison (threshold, MST, PMFG, partial corr)
 4. Walk-forward CV for GAT with baselines comparison
 5. Per-target analysis with bootstrap CIs
 6. Ablation study (hidden_dim, heads, layers, seq_len, dropout)
 7. Feature ablation (remove each feature group one at a time)
 8. Conditional spectral indicators
 9. Change-point detection on spectral evolution
10. ABM with multiple dynamics (OU, jump-diffusion, regime-switching)
11. ABM parameter sensitivity (Sobol indices)
12. Threshold sensitivity with principled selection (MDL, CV, percolation)
13. Backtesting with bootstrap CIs on all correlations
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("publication")

# ── Output directories ─────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results" / "publication"
FIGURES_DIR = Path(__file__).parent / "figures" / "publication"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication-quality matplotlib ──────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Fixed seeds ────────────────────────────────────────────────────
GLOBAL_SEED = 42

# ── Utility: JSON-safe conversion ──────────────────────────────────

def _json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    return obj


def _save_json(data, name: str):
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_safe)
    logger.info("Saved %s", path)


def _save_fig(name: str):
    path = FIGURES_DIR / f"{name}.png"
    plt.savefig(path)
    plt.close()
    logger.info("Saved %s", path)


# ── Utility: profiling decorator ───────────────────────────────────

def profiled(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        logger.info("=== START: %s ===", fn.__name__)
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        logger.info("=== DONE: %s (%.1fs) ===", fn.__name__, elapsed)
        return result
    return wrapper


# ── Utility: bootstrap CI ──────────────────────────────────────────

def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=GLOBAL_SEED):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    if n < 2:
        m = float(values[0]) if n == 1 else 0.0
        return {"mean": m, "ci_lo": m, "ci_hi": m}
    boot_means = np.array([np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(values)),
        "ci_lo": float(np.percentile(boot_means, alpha * 100)),
        "ci_hi": float(np.percentile(boot_means, (1 - alpha) * 100)),
    }


# ── Utility: Diebold-Mariano test ─────────────────────────────────

def diebold_mariano(e1, e2, h=1):
    """Two-sided Diebold-Mariano test for equal predictive accuracy.

    e1, e2: forecast error arrays (same length).
    Returns (DM statistic, two-sided p-value).
    """
    from scipy.stats import t as t_dist
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    d_bar = np.mean(d)
    # Newey-West variance with h-1 lags
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1] if len(d) > k + 1 else 0
        gamma_sum += gamma_k
    var_d = (gamma_0 + 2 * gamma_sum) / n
    if var_d <= 0:
        return 0.0, 1.0
    dm = d_bar / np.sqrt(var_d)
    p_val = 2 * (1 - t_dist.cdf(abs(dm), df=n - 1))
    return float(dm), float(p_val)


# ── Utility: fetch snapshots with retry ────────────────────────────

def _fetch_snapshots(lookback_years=3, stride=5, min_corr=0.3, rmt_denoise=False):
    from dashboard.data_api import build_daily_graph_snapshots
    for attempt in range(3):
        try:
            snaps = build_daily_graph_snapshots(
                lookback_years=lookback_years, corr_window=60,
                min_corr=min_corr, stride=stride, rmt_denoise=rmt_denoise,
            )
            if len(snaps) > 50:
                logger.info("Fetched %d snapshots (attempt %d)", len(snaps), attempt + 1)
                return snaps
            logger.warning("Only %d snapshots (attempt %d), retrying...", len(snaps), attempt + 1)
            time.sleep(5)
        except Exception as e:
            logger.warning("yfinance fetch failed (attempt %d): %s", attempt + 1, e)
            time.sleep(5)
    logger.error("Could not fetch sufficient data from yfinance after 3 attempts.")
    return []


# ======================================================================
# Experiment 1: Multi-scale SCG validation
# ======================================================================

@profiled
def exp01_multiscale_scg():
    from scr_financial.network.spectral import (
        compute_laplacian, eigendecomposition, find_spectral_gap,
    )
    from scr_financial.network.coarse_graining import SpectralCoarseGraining
    from scr_financial.network.multiscale_scg import MultiScaleSCG

    rng = np.random.default_rng(GLOBAL_SEED)
    results = []

    # Topology generators
    def gen_erdos_renyi(n, p, rng):
        adj = (rng.random((n, n)) < p).astype(float)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)
        # Random positive weights
        weights = rng.uniform(0.1, 0.8, (n, n))
        weights = (weights + weights.T) / 2
        return adj * weights

    def gen_block_model(n, k, intra, inter, rng):
        adj = np.zeros((n, n))
        size = n // k
        for c in range(k):
            for i in range(size):
                for j in range(i + 1, size):
                    w = intra + rng.uniform(-0.1, 0.1)
                    if w > 0:
                        adj[c * size + i, c * size + j] = w
                        adj[c * size + j, c * size + i] = w
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                for i in range(size):
                    for j in range(size):
                        w = inter + rng.uniform(-0.05, 0.05)
                        if w > 0:
                            adj[c1 * size + i, c2 * size + j] = w
                            adj[c2 * size + j, c1 * size + i] = w
        return adj

    def gen_star(n, rng):
        adj = np.zeros((n, n))
        for i in range(1, n):
            w = rng.uniform(0.3, 0.8)
            adj[0, i] = w
            adj[i, 0] = w
        return adj

    def gen_scale_free(n, rng):
        import networkx as nx
        G = nx.barabasi_albert_graph(n, 3, seed=int(rng.integers(0, 10000)))
        adj = nx.to_numpy_array(G).astype(float)
        weights = rng.uniform(0.1, 0.8, adj.shape)
        weights = (weights + weights.T) / 2
        return adj * weights

    topologies = [
        ("erdos_renyi_dense", 10, lambda rng: gen_erdos_renyi(10, 0.6, rng)),
        ("erdos_renyi_sparse", 10, lambda rng: gen_erdos_renyi(10, 0.3, rng)),
        ("block_2", 10, lambda rng: gen_block_model(10, 2, 0.7, 0.15, rng)),
        ("block_3", 12, lambda rng: gen_block_model(12, 3, 0.8, 0.10, rng)),
        ("star", 10, lambda rng: gen_star(10, rng)),
        ("scale_free", 10, lambda rng: gen_scale_free(10, rng)),
    ]

    N_NETWORKS = 50

    for topo_name, n, gen_fn in topologies:
        logger.info("  Topology: %s (%d networks)", topo_name, N_NETWORKS)
        topo_results = []
        for net_i in range(N_NETWORKS):
            seed_rng = np.random.default_rng(GLOBAL_SEED + net_i)
            adj = gen_fn(seed_rng)
            adj = (adj + adj.T) / 2
            np.fill_diagonal(adj, 0)
            node_ids = [f"bank_{i}" for i in range(n)]

            ms = MultiScaleSCG(adj, node_ids)
            scale_results = ms.run_all_scales()
            opt_k = ms.optimal_scale()

            # Spectral properties of original
            L = compute_laplacian(adj, normalized=False)
            evals, _ = eigendecomposition(L)
            gap_idx, gap_size = find_spectral_gap(evals, adjacency_matrix=adj)
            lam2 = float(evals[1]) if len(evals) > 1 else 0

            topo_results.append({
                "network_id": net_i,
                "optimal_k": opt_k,
                "lambda_2": round(lam2, 4),
                "spectral_gap": round(float(gap_size), 4),
                "n_scales_tested": len(scale_results),
                "best_r2": max((r.get("r2_mean", 0) for r in scale_results), default=0),
            })

        r2_vals = [r["best_r2"] for r in topo_results]
        results.append({
            "topology": topo_name,
            "n_networks": N_NETWORKS,
            "r2_bootstrap": bootstrap_ci(r2_vals),
            "optimal_k_mean": float(np.mean([r["optimal_k"] for r in topo_results])),
            "per_network": topo_results,
        })
        logger.info("    R2=%.4f [%.4f, %.4f]", results[-1]["r2_bootstrap"]["mean"],
                     results[-1]["r2_bootstrap"]["ci_lo"], results[-1]["r2_bootstrap"]["ci_hi"])

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["topology"] for r in results]
    means = [r["r2_bootstrap"]["mean"] for r in results]
    ci_lo = [r["r2_bootstrap"]["ci_lo"] for r in results]
    ci_hi = [r["r2_bootstrap"]["ci_hi"] for r in results]
    errs = [[m - lo for m, lo in zip(means, ci_lo)],
            [hi - m for m, hi in zip(means, ci_hi)]]
    ax.barh(names, means, xerr=errs, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xlabel("SCG Reconstruction $R^2$")
    ax.set_title("Multi-Scale SCG Validation (50 Networks per Topology)")
    ax.set_xlim(0, 1.05)
    _save_fig("exp01_scg_validation")

    _save_json(results, "exp01_scg_validation")
    return results


# ======================================================================
# Experiment 2: RMT denoising comparison
# ======================================================================

@profiled
def exp02_rmt_denoising():
    from scr_financial.network.rmt import denoise_correlation, marchenko_pastur_bounds, fit_marchenko_pastur
    from scr_financial.network.spectral import (
        compute_laplacian, eigendecomposition, find_spectral_gap,
    )

    rng = np.random.default_rng(GLOBAL_SEED)
    N, T = 10, 252
    # Generate synthetic returns with known correlation structure
    true_corr = np.full((N, N), 0.3)
    np.fill_diagonal(true_corr, 1.0)
    # Add block structure
    true_corr[:5, :5] = 0.6
    true_corr[5:, 5:] = 0.6
    np.fill_diagonal(true_corr, 1.0)

    L_chol = np.linalg.cholesky(true_corr)
    returns = (L_chol @ rng.standard_normal((N, T))).T  # (T, N)
    sample_corr = np.corrcoef(returns.T)

    methods = ["raw", "constant", "shrinkage", "targeted"]
    results = {}

    for method in methods:
        if method == "raw":
            corr = sample_corr
        else:
            corr = denoise_correlation(sample_corr, T=T, method=method)

        np.fill_diagonal(corr, 0)
        adj = np.clip(corr, 0, None)
        adj[adj < 0.3] = 0

        L = compute_laplacian(adj, normalized=True)
        evals, _ = eigendecomposition(L)
        gap_idx, gap_size = find_spectral_gap(evals)
        lam2 = float(evals[1]) if len(evals) > 1 else 0

        # Frobenius distance from true correlation
        corr_with_diag = corr.copy()
        np.fill_diagonal(corr_with_diag, 1.0)
        frob_error = float(np.linalg.norm(corr_with_diag - true_corr, "fro"))

        results[method] = {
            "lambda_2": round(lam2, 4),
            "spectral_gap": round(float(gap_size), 4),
            "spectral_radius": round(float(evals[-1]), 4),
            "frobenius_error": round(frob_error, 4),
            "n_edges": int(np.count_nonzero(adj) // 2),
            "eigenvalues": [round(float(e), 4) for e in evals],
        }

    # MP fit on raw correlation
    evals_raw = np.linalg.eigvalsh(sample_corr)
    mp_fit = fit_marchenko_pastur(evals_raw, T, N)
    results["mp_fit"] = mp_fit

    # Figure: eigenvalue comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for method in methods:
        ax1.plot(results[method]["eigenvalues"], marker="o", label=method, markersize=4)
    _, lam_plus = marchenko_pastur_bounds(T, N)
    ax1.axhline(y=lam_plus, color="red", linestyle="--", alpha=0.5, label=f"MP $\\lambda_+$={lam_plus:.2f}")
    ax1.set_xlabel("Eigenvalue index")
    ax1.set_ylabel("$\\lambda$")
    ax1.set_title("Laplacian Eigenvalues: Raw vs Denoised")
    ax1.legend(fontsize=8)

    frob_vals = [results[m]["frobenius_error"] for m in methods]
    ax2.bar(methods, frob_vals, color=["grey", "steelblue", "coral", "green"], alpha=0.8)
    ax2.set_ylabel("Frobenius Distance from True $C$")
    ax2.set_title("Denoising Quality")
    _save_fig("exp02_rmt_denoising")

    _save_json(results, "exp02_rmt_denoising")
    return results


# ======================================================================
# Experiment 3: Alternative network construction comparison
# ======================================================================

@profiled
def exp03_network_construction():
    from scr_financial.network.alternative_builders import compare_networks

    snapshots = _fetch_snapshots(lookback_years=3, stride=5)
    if not snapshots:
        return {"error": "insufficient data"}

    # Build returns from snapshots (use underlying price data)
    from dashboard.data_api import _fetch_prices, ALL_BANKS
    prices = _fetch_prices(ALL_BANKS, period="3y")
    if prices.empty:
        return {"error": "no price data"}

    returns = prices.pct_change().dropna().tail(252)
    corr_matrix = returns.corr().values

    comparison = compare_networks(returns, corr_matrix, threshold=0.3)
    results = comparison.to_dict(orient="records")

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    methods = [r["method"] for r in results]
    edges = [r["n_edges"] for r in results]
    lam2s = [r["lambda_2"] for r in results]

    ax1.bar(methods, edges, color="steelblue", alpha=0.8)
    ax1.set_ylabel("Number of Edges")
    ax1.set_title("Network Density by Construction Method")

    ax2.bar(methods, lam2s, color="coral", alpha=0.8)
    ax2.set_ylabel("$\\lambda_2$ (Algebraic Connectivity)")
    ax2.set_title("Spectral Properties by Method")
    _save_fig("exp03_network_construction")

    _save_json(results, "exp03_network_construction")
    return results


# ======================================================================
# Experiment 4: Walk-forward CV for GAT with baselines
# ======================================================================

@profiled
def exp04_walkforward_cv():
    import torch
    from scr_financial.ml.gnn_predictor import GNNPredictor
    from scr_financial.ml.walk_forward import walk_forward_evaluate
    from scr_financial.ml.baselines import (
        PersistenceBaseline, MovingAverageBaseline, ARIMABaseline,
        RandomForestBaseline, GradientBoostingBaseline,
        extract_flat_features,
    )
    from scr_financial.ml.gnn_variants import create_encoder

    snapshots = _fetch_snapshots(lookback_years=3, stride=5)
    if not snapshots:
        return {"error": "insufficient data"}

    n_snaps = len(snapshots)
    logger.info("Walk-forward CV with %d snapshots", n_snaps)

    # CV params
    cv_kwargs = dict(n_splits=5, min_train_size=100, test_size=40, gap_size=10)

    # -- GNN models --
    gat_config = dict(seq_len=10, hidden_dim=64, num_gat_layers=2, heads=8, dropout=0.2)
    gcn_config = dict(seq_len=10, hidden_dim=64, num_gat_layers=2, heads=1, dropout=0.2,
                      encoder_type="gcn")

    results = {}

    # GAT (wide)
    logger.info("  Running GAT walk-forward CV...")
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    gat_result = walk_forward_evaluate(
        GNNPredictor, gat_config, snapshots,
        epochs=300, lr=3e-3, patience=40, seed=GLOBAL_SEED, **cv_kwargs,
    )
    results["GAT"] = gat_result

    # GCN
    logger.info("  Running GCN walk-forward CV...")
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    gcn_result = walk_forward_evaluate(
        GNNPredictor, gcn_config, snapshots,
        epochs=300, lr=3e-3, patience=40, seed=GLOBAL_SEED, **cv_kwargs,
    )
    results["GCN"] = gcn_result

    # -- Sklearn baselines via flat features --
    seq_len = 10
    sequences, targets = [], []
    for i in range(len(snapshots) - seq_len):
        sequences.append(snapshots[i:i + seq_len])
        tgt = snapshots[i + seq_len]
        targets.append([tgt["targets"]["lambda_2"],
                        tgt["targets"]["spectral_gap"],
                        tgt["targets"]["spectral_radius"]])
    targets = np.array(targets)

    split = int(len(sequences) * 0.8)
    X_train = extract_flat_features(sequences[:split])
    y_train = targets[:split]
    X_test = extract_flat_features(sequences[split:])
    y_test = targets[split:]

    baselines = [
        ("Persistence", PersistenceBaseline()),
        ("MA(5)", MovingAverageBaseline(5)),
        ("ARIMA(1,0,0)", ARIMABaseline((1, 0, 0))),
        ("RandomForest", RandomForestBaseline(seed=GLOBAL_SEED)),
        ("GradientBoosting", GradientBoostingBaseline(seed=GLOBAL_SEED)),
    ]

    for bl_name, bl in baselines:
        logger.info("  Running baseline: %s", bl_name)
        try:
            bl.fit(X_train, y_train)
            y_pred = bl.predict(X_test)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 3) if len(y_pred) % 3 == 0 else np.zeros_like(y_test)

            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test, axis=0)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
            mse = float(np.mean((y_test - y_pred) ** 2))

            # Per-target R2
            per_target = {}
            for ti, tname in enumerate(["lambda_2", "spectral_gap", "spectral_radius"]):
                ss_r = np.sum((y_test[:, ti] - y_pred[:, ti]) ** 2)
                ss_t = np.sum((y_test[:, ti] - np.mean(y_test[:, ti])) ** 2)
                per_target[tname] = round(float(1 - ss_r / ss_t) if ss_t > 1e-8 else 0.0, 4)

            results[bl_name] = {
                "aggregate": {"test_r2_mean": round(r2, 4), "test_r2_std": 0,
                              "test_mse_mean": round(mse, 6), "n_folds_completed": 1},
                "per_target": per_target,
                "y_pred": y_pred.tolist(),
                "y_test": y_test.tolist(),
            }
        except Exception as e:
            logger.warning("  Baseline %s failed: %s", bl_name, e)
            results[bl_name] = {"aggregate": {"test_r2_mean": 0, "test_r2_std": 0}}

    # Diebold-Mariano tests vs Persistence
    if "Persistence" in results and "y_pred" in results["Persistence"]:
        y_pred_persist = np.array(results["Persistence"]["y_pred"])
        e_persist = (y_test - y_pred_persist).flatten()
        dm_results = {}
        for name, res in results.items():
            if name == "Persistence" or "y_pred" not in res:
                continue
            y_pred_m = np.array(res["y_pred"])
            e_m = (y_test - y_pred_m).flatten()
            dm_stat, dm_p = diebold_mariano(e_persist, e_m, h=1)
            dm_results[name] = {"DM_stat": round(dm_stat, 4), "DM_pvalue": round(dm_p, 4)}
        results["diebold_mariano_vs_persistence"] = dm_results

    # Bootstrap CIs on R2
    r2_ci = {}
    for name, res in results.items():
        if "aggregate" in res:
            r2_val = res["aggregate"]["test_r2_mean"]
            r2_ci[name] = bootstrap_ci([r2_val])  # single-point for baselines
    results["r2_bootstrap_ci"] = r2_ci

    # Figure: model comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = [n for n in results if "aggregate" in results[n]]
    r2_vals = [results[n]["aggregate"]["test_r2_mean"] for n in model_names]
    colors = ["coral" if "GAT" in n or "GCN" in n else "steelblue" for n in model_names]
    ax.barh(model_names, r2_vals, color=colors, alpha=0.8)
    ax.set_xlabel("Test $R^2$")
    ax.set_title("Walk-Forward CV: Model Comparison")
    ax.axvline(x=0, color="black", linewidth=0.5)
    _save_fig("exp04_model_comparison")

    # Remove large arrays before saving
    for name in results:
        if isinstance(results[name], dict):
            results[name].pop("y_pred", None)
            results[name].pop("y_test", None)

    _save_json(results, "exp04_walkforward_cv")
    return results


# ======================================================================
# Experiment 5: Per-target analysis with bootstrap CIs
# ======================================================================

@profiled
def exp05_per_target_analysis():
    import torch
    from scr_financial.ml.gnn_predictor import GNNPredictor

    snapshots = _fetch_snapshots(lookback_years=3, stride=5)
    if not snapshots:
        return {"error": "insufficient data"}

    N_SEEDS = 5
    config = dict(seq_len=10, hidden_dim=64, num_gat_layers=2, heads=8, dropout=0.2)
    target_names = ["lambda_2", "spectral_gap", "spectral_radius"]

    per_target_r2 = {t: [] for t in target_names}

    for seed in range(N_SEEDS):
        np.random.seed(GLOBAL_SEED + seed)
        torch.manual_seed(GLOBAL_SEED + seed)
        predictor = GNNPredictor(**config)
        predictor.train(snapshots, epochs=300, lr=3e-3, patience=40)
        r2_pt = predictor.test_metrics.get("r2_per_target", {})
        for t in target_names:
            per_target_r2[t].append(r2_pt.get(t, 0))

    results = {}
    for t in target_names:
        results[t] = bootstrap_ci(per_target_r2[t])
        logger.info("  %s: R2=%.4f [%.4f, %.4f]", t,
                     results[t]["mean"], results[t]["ci_lo"], results[t]["ci_hi"])

    # Figure
    fig, ax = plt.subplots(figsize=(7, 4))
    means = [results[t]["mean"] for t in target_names]
    ci_lo = [results[t]["ci_lo"] for t in target_names]
    ci_hi = [results[t]["ci_hi"] for t in target_names]
    errs = [[m - lo for m, lo in zip(means, ci_lo)],
            [hi - m for m, hi in zip(means, ci_hi)]]
    labels = ["$\\lambda_2$", "Spectral Gap", "$\\rho$"]
    ax.bar(labels, means, yerr=errs, capsize=8, color=["steelblue", "coral", "green"], alpha=0.8)
    ax.set_ylabel("Test $R^2$")
    ax.set_title("Per-Target GAT Performance (95% Bootstrap CI, 5 seeds)")
    _save_fig("exp05_per_target")

    _save_json(results, "exp05_per_target")
    return results


# ======================================================================
# Experiment 6: Hyperparameter ablation study
# ======================================================================

@profiled
def exp06_ablation():
    from scr_financial.ml.ablation import run_ablation_study

    snapshots = _fetch_snapshots(lookback_years=3, stride=5)
    if not snapshots:
        return {"error": "insufficient data"}

    base_config = dict(seq_len=10, hidden_dim=32, num_gat_layers=2, heads=4, dropout=0.2)
    ablation_dims = {
        "hidden_dim": [16, 32, 64],
        "heads": [2, 4, 8],
        "num_gat_layers": [1, 2, 3],
        "seq_len": [5, 10, 15],
        "dropout": [0.1, 0.2, 0.3],
    }

    results = run_ablation_study(
        snapshots, base_config, ablation_dims,
        n_seeds=3, epochs=200, patience=30,
    )

    # Figure: ablation grid
    n_dims = len(ablation_dims)
    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 4), sharey=True)
    if n_dims == 1:
        axes = [axes]
    for ax, (dim, vals) in zip(axes, results.items()):
        x = [str(v["value"]) for v in vals]
        y = [v["r2_mean"] for v in vals]
        yerr = [v["r2_std"] for v in vals]
        ax.bar(x, y, yerr=yerr, capsize=5, color="steelblue", alpha=0.8)
        ax.set_title(dim)
        ax.set_xlabel("Value")
    axes[0].set_ylabel("Test $R^2$")
    plt.suptitle("Hyperparameter Ablation (3 seeds each)")
    plt.tight_layout()
    _save_fig("exp06_ablation")

    _save_json(results, "exp06_ablation")
    return results


# ======================================================================
# Experiment 7: Feature ablation
# ======================================================================

@profiled
def exp07_feature_ablation():
    from scr_financial.ml.ablation import feature_ablation

    snapshots = _fetch_snapshots(lookback_years=3, stride=5)
    if not snapshots:
        return {"error": "insufficient data"}

    config = dict(seq_len=10, hidden_dim=64, num_gat_layers=2, heads=8, dropout=0.2)
    results = feature_ablation(snapshots, config, n_seeds=3, epochs=150, patience=25)

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [k for k in results if k != "baseline"]
    drops = [results[k]["r2_drop"] for k in groups]
    colors = ["coral" if d > 0 else "steelblue" for d in drops]
    ax.barh(groups, drops, color=colors, alpha=0.8)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("$R^2$ Drop (higher = more important)")
    ax.set_title(f"Feature Ablation (baseline R2={results['baseline']['r2_mean']:.4f})")
    _save_fig("exp07_feature_ablation")

    _save_json(results, "exp07_feature_ablation")
    return results


# ======================================================================
# Experiment 8: Conditional spectral indicators
# ======================================================================

@profiled
def exp08_conditional_spectral():
    from scr_financial.network.conditional_spectral import (
        vol_regressed_lambda2, spectral_momentum, conditional_scg_risk,
        marchenko_pastur_relative_connectivity,
    )

    snapshots = _fetch_snapshots(lookback_years=3, stride=5)
    if not snapshots:
        return {"error": "insufficient data"}

    lambda2_series = [s["targets"]["lambda_2"] for s in snapshots]
    rho_series = [s["targets"]["spectral_radius"] for s in snapshots]
    vol_series = [float(np.mean(s["node_features"][:, 0])) for s in snapshots]
    dates = [s.get("date", str(i)) for i, s in enumerate(snapshots)]

    # Vol-regressed lambda_2
    vr = vol_regressed_lambda2(lambda2_series, vol_series)

    # Spectral momentum
    sm = spectral_momentum(lambda2_series, window=20)

    # Conditional SCG risk
    cond_risk = conditional_scg_risk(lambda2_series, rho_series, vol_series)

    # MP-relative connectivity
    mp_rel = marchenko_pastur_relative_connectivity(lambda2_series, T=60, N=10)

    results = {
        "vol_regression_r2": vr["r_squared"],
        "vol_regression_beta": vr["beta"],
        "n_momentum_alerts": len(sm["alerts"]),
        "cond_risk_mean": float(np.mean(cond_risk)),
        "cond_risk_std": float(np.std(cond_risk)),
        "mp_relative_mean": float(np.mean(mp_rel)),
        "mp_relative_above_1_frac": float(np.mean(mp_rel > 1)),
    }

    # Figure: 4-panel conditional spectral
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    n = len(lambda2_series)
    x = np.arange(n)

    ax = axes[0, 0]
    ax.plot(x, lambda2_series, label="$\\lambda_2$", color="steelblue", alpha=0.8)
    ax.plot(x, vr["residuals"], label="Vol-regressed residual", color="coral", alpha=0.8)
    ax.set_title(f"Vol-Regressed $\\lambda_2$ ($R^2$={vr['r_squared']:.3f})")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x, sm["z_scores"], color="steelblue", alpha=0.8)
    ax.axhline(y=2, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=-2, color="red", linestyle="--", alpha=0.5)
    for alert in sm["alerts"]:
        ax.axvline(x=alert, color="red", alpha=0.2, linewidth=0.5)
    ax.set_title(f"Spectral Momentum ({len(sm['alerts'])} alerts)")

    ax = axes[1, 0]
    ax.plot(x, cond_risk, color="coral", alpha=0.8)
    ax.set_title("Conditional SCG Risk")
    ax.set_ylabel("Risk")

    ax = axes[1, 1]
    ax.plot(x, mp_rel, color="green", alpha=0.8)
    ax.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="MP bound")
    ax.set_title("MP-Relative Connectivity")
    ax.legend()

    plt.suptitle("Conditional Spectral Indicators")
    plt.tight_layout()
    _save_fig("exp08_conditional_spectral")

    _save_json(results, "exp08_conditional_spectral")
    return results


# ======================================================================
# Experiment 9: Change-point detection
# ======================================================================

@profiled
def exp09_change_points():
    from scr_financial.network.change_point import (
        cusum_detector, lambda2_rate_of_change, binary_segmentation,
    )

    snapshots = _fetch_snapshots(lookback_years=3, stride=5)
    if not snapshots:
        return {"error": "insufficient data"}

    lambda2_series = [s["targets"]["lambda_2"] for s in snapshots]
    dates = [s.get("date", str(i)) for i, s in enumerate(snapshots)]

    cusum = cusum_detector(lambda2_series, threshold=3.0)
    roc = lambda2_rate_of_change(lambda2_series, window=20)
    binseg = binary_segmentation(lambda2_series, min_segment=20, max_breaks=10)

    results = {
        "cusum_change_points": cusum["change_points"],
        "roc_alerts": roc["alerts"],
        "binseg_breakpoints": binseg["breakpoints"],
        "binseg_n_segments": binseg["n_segments"],
        "binseg_segment_means": binseg["segment_means"],
    }

    # Figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    n = len(lambda2_series)
    x = np.arange(n)

    ax = axes[0]
    ax.plot(x, lambda2_series, color="steelblue", alpha=0.8)
    for cp in cusum["change_points"]:
        ax.axvline(x=cp, color="red", alpha=0.5, linewidth=1)
    ax.set_title(f"CUSUM Change Points ({len(cusum['change_points'])} detected)")
    ax.set_ylabel("$\\lambda_2$")

    ax = axes[1]
    ax.plot(x, roc["z_scores"], color="coral", alpha=0.8)
    ax.axhline(y=2, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=-2, color="red", linestyle="--", alpha=0.5)
    ax.set_title(f"Rate-of-Change Z-Scores ({len(roc['alerts'])} alerts)")

    ax = axes[2]
    ax.plot(x, lambda2_series, color="steelblue", alpha=0.8)
    for bp in binseg["breakpoints"]:
        ax.axvline(x=bp, color="green", alpha=0.7, linewidth=1.5)
    # Segment means
    for seg_i, (start, end) in enumerate(binseg["segments"]):
        mean_val = binseg["segment_means"][seg_i]
        ax.hlines(mean_val, start, end - 1, colors="red", linewidth=2, alpha=0.7)
    ax.set_title(f"Binary Segmentation ({binseg['n_segments']} segments)")
    ax.set_xlabel("Snapshot Index")

    plt.suptitle("Change-Point Detection on $\\lambda_2$ Evolution")
    plt.tight_layout()
    _save_fig("exp09_change_points")

    _save_json(results, "exp09_change_points")
    return results


# ======================================================================
# Experiment 10: ABM with multiple dynamics
# ======================================================================

@profiled
def exp10_abm_dynamics():
    from scr_financial.abm.simulation import BankingSystemSimulation
    from scr_financial.abm.dynamics import (
        DEFAULT_OU, DEFAULT_JUMP_DIFFUSION, DEFAULT_REGIME_SWITCHING,
    )

    bank_names = ["DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
                  "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA"]

    def build_sim(seed, dynamics=None):
        rng = np.random.default_rng(seed)
        bank_data = {}
        for name in bank_names:
            bank_data[name] = {
                "CET1_ratio": rng.uniform(10, 16), "LCR": rng.uniform(110, 180),
                "NSFR": rng.uniform(100, 140), "total_assets": rng.uniform(5e11, 2.5e12),
                "cash": rng.uniform(2e10, 8e10),
                "interbank_assets": rng.uniform(5e9, 5e10),
                "interbank_liabilities": rng.uniform(5e9, 5e10),
            }
        network = {}
        for i, bid in enumerate(bank_names):
            network[bid] = {}
            for j, oid in enumerate(bank_names):
                if i != j:
                    network[bid][oid] = rng.uniform(0.15, 0.65)
        sim = BankingSystemSimulation(
            bank_data, network, {"CISS": 0.5, "funding_stress": 0.3}, seed=seed,
        )
        # Assign dynamics if provided
        if dynamics is not None:
            for bank in sim.banks.values():
                if hasattr(bank, "dynamics"):
                    bank.dynamics = dynamics
        return sim

    shock = {"DE_DBK": {"CET1_ratio": -5.0, "LCR": -60.0, "cash": -2e10}}
    N_RUNS = 100

    dynamics_configs = [
        ("OU", DEFAULT_OU),
        ("JumpDiffusion", DEFAULT_JUMP_DIFFUSION),
        ("RegimeSwitching", DEFAULT_REGIME_SWITCHING),
    ]

    results = {}
    for dyn_name, dyn in dynamics_configs:
        logger.info("  Dynamics: %s (%d runs)", dyn_name, N_RUNS)
        defaults_list = []
        final_cet1_list = []

        for run_i in range(N_RUNS):
            sim = build_sim(1000 + run_i, dynamics=dyn)
            sim.run_simulation(5)
            sim.apply_external_shock(shock)
            sim.run_simulation(30)

            n_def = sum(1 for b in sim.banks.values() if b._defaulted)
            cet1s = [b.state.get("CET1_ratio", 0) for b in sim.banks.values()]
            defaults_list.append(n_def)
            final_cet1_list.append(float(np.mean(cet1s)))

        results[dyn_name] = {
            "mean_defaults": round(float(np.mean(defaults_list)), 2),
            "std_defaults": round(float(np.std(defaults_list)), 2),
            "max_defaults": int(np.max(defaults_list)),
            "default_rate": round(float(np.mean([d > 0 for d in defaults_list])), 3),
            "mean_final_cet1": round(float(np.mean(final_cet1_list)), 2),
            "defaults_bootstrap": bootstrap_ci(defaults_list),
        }
        logger.info("    %s: %.1f+/-%.1f defaults (rate=%.1f%%)", dyn_name,
                     results[dyn_name]["mean_defaults"], results[dyn_name]["std_defaults"],
                     results[dyn_name]["default_rate"] * 100)

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    names = list(results.keys())
    means = [results[n]["mean_defaults"] for n in names]
    stds = [results[n]["std_defaults"] for n in names]
    ax1.bar(names, means, yerr=stds, capsize=8, color=["steelblue", "coral", "green"], alpha=0.8)
    ax1.set_ylabel("Mean Defaults")
    ax1.set_title("Default Count by Dynamics Model")

    cet1s = [results[n]["mean_final_cet1"] for n in names]
    ax2.bar(names, cet1s, color=["steelblue", "coral", "green"], alpha=0.8)
    ax2.axhline(y=4.5, color="red", linestyle="--", alpha=0.5, label="Min CET1")
    ax2.set_ylabel("Mean Final CET1 (%)")
    ax2.set_title("Final CET1 by Dynamics Model")
    ax2.legend()
    _save_fig("exp10_abm_dynamics")

    _save_json(results, "exp10_abm_dynamics")
    return results


# ======================================================================
# Experiment 11: ABM parameter sensitivity (Sobol indices)
# ======================================================================

@profiled
def exp11_abm_sensitivity():
    from scr_financial.abm.simulation import BankingSystemSimulation
    from scr_financial.abm.sensitivity import (
        sobol_sensitivity, ABM_BASE_PARAMS, ABM_PARAM_RANGES,
    )

    bank_names = ["DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
                  "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA"]

    def build_and_run(params: Dict[str, float]) -> Dict[str, float]:
        rng = np.random.default_rng(GLOBAL_SEED)
        bank_data = {}
        for name in bank_names:
            bank_data[name] = {
                "CET1_ratio": rng.uniform(10, 16), "LCR": rng.uniform(110, 180),
                "NSFR": rng.uniform(100, 140), "total_assets": rng.uniform(5e11, 2.5e12),
                "cash": rng.uniform(2e10, 8e10),
                "interbank_assets": rng.uniform(5e9, 5e10),
                "interbank_liabilities": rng.uniform(5e9, 5e10),
            }
        network = {}
        for i, bid in enumerate(bank_names):
            network[bid] = {}
            for j, oid in enumerate(bank_names):
                if i != j:
                    network[bid][oid] = rng.uniform(0.15, 0.65)
        sim = BankingSystemSimulation(
            bank_data, network, {"CISS": 0.5, "funding_stress": 0.3}, seed=GLOBAL_SEED,
        )
        shock = {"DE_DBK": {"CET1_ratio": -5.0, "LCR": -60.0}}
        sim.run_simulation(5)
        sim.apply_external_shock(shock)
        sim.run_simulation(30)

        n_def = sum(1 for b in sim.banks.values() if b._defaulted)
        cet1s = [b.state.get("CET1_ratio", 0) for b in sim.banks.values()]
        return {
            "n_defaults": n_def,
            "mean_cet1": float(np.mean(cet1s)),
            "min_cet1": float(np.min(cet1s)),
        }

    # Sobol bounds (use ranges from ABM_PARAM_RANGES: (lo, hi, _))
    param_bounds = {k: (lo, hi) for k, (lo, hi, _) in ABM_PARAM_RANGES.items()}

    results = sobol_sensitivity(build_and_run, param_bounds, n_samples=128, seed=GLOBAL_SEED)

    # Figure: Sobol indices for n_defaults
    if "n_defaults" in results:
        fig, ax = plt.subplots(figsize=(10, 5))
        params = list(results["n_defaults"].keys())
        s1 = [results["n_defaults"][p]["S1"] for p in params]
        st = [results["n_defaults"][p]["ST"] for p in params]
        x = np.arange(len(params))
        w = 0.35
        ax.bar(x - w / 2, s1, w, label="$S_1$ (First-order)", color="steelblue", alpha=0.8)
        ax.bar(x + w / 2, st, w, label="$S_T$ (Total-order)", color="coral", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(params, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Sobol Index")
        ax.set_title("ABM Parameter Sensitivity (Sobol Indices for Default Count)")
        ax.legend()
        plt.tight_layout()
        _save_fig("exp11_sobol_indices")

    _save_json(results, "exp11_abm_sensitivity")
    return results


# ======================================================================
# Experiment 12: Threshold sensitivity with principled selection
# ======================================================================

@profiled
def exp12_threshold_sensitivity():
    from scr_financial.network.threshold import (
        information_theoretic_threshold, cv_based_threshold, percolation_threshold,
    )
    from dashboard.data_api import _fetch_prices, ALL_BANKS

    prices = _fetch_prices(ALL_BANKS, period="3y")
    if prices.empty:
        return {"error": "no price data"}

    returns = prices.pct_change().dropna().tail(252)
    corr_matrix = returns.corr().values

    # MDL
    mdl = information_theoretic_threshold(corr_matrix)

    # CV-based
    cv = cv_based_threshold(returns.values)

    # Percolation
    perc = percolation_threshold(corr_matrix)

    results = {
        "mdl_optimal": mdl["optimal_threshold"],
        "cv_optimal": cv["optimal_threshold"],
        "percolation_threshold": perc["percolation_threshold"],
        "mdl_scores": mdl["mdl_scores"],
        "cv_scores": cv["cv_scores"],
        "percolation_results": perc["results"],
    }

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    thrs = [r["threshold"] for r in mdl["mdl_scores"]]
    vals = [r["mdl"] for r in mdl["mdl_scores"]]
    ax.plot(thrs, vals, color="steelblue", marker="o", markersize=3)
    ax.axvline(x=mdl["optimal_threshold"], color="red", linestyle="--", label=f"MDL opt={mdl['optimal_threshold']:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("MDL")
    ax.set_title("Information-Theoretic (MDL)")
    ax.legend()

    ax = axes[1]
    thrs = [r["threshold"] for r in cv["cv_scores"]]
    vals = [r["gap_cv"] for r in cv["cv_scores"]]
    ax.plot(thrs, vals, color="coral", marker="o", markersize=3)
    ax.axvline(x=cv["optimal_threshold"], color="red", linestyle="--", label=f"CV opt={cv['optimal_threshold']:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Gap CV")
    ax.set_title("Cross-Validated Spectral Stability")
    ax.legend()

    ax = axes[2]
    thrs = [r["threshold"] for r in perc["results"]]
    vals = [r["fraction_in_giant"] for r in perc["results"]]
    ax.plot(thrs, vals, color="green", marker="o", markersize=3)
    ax.axvline(x=perc["percolation_threshold"], color="red", linestyle="--",
               label=f"Perc={perc['percolation_threshold']:.2f}")
    ax.axhline(y=0.9, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Fraction in Giant Component")
    ax.set_title("Percolation Analysis")
    ax.legend()

    plt.suptitle("Principled Threshold Selection")
    plt.tight_layout()
    _save_fig("exp12_threshold_sensitivity")

    _save_json(results, "exp12_threshold_sensitivity")
    return results


# ======================================================================
# Experiment 13: Backtesting with bootstrap CIs
# ======================================================================

@profiled
def exp13_backtesting():
    from dashboard.data_api import build_daily_graph_snapshots

    thresholds = [0.2, 0.3, 0.4, 0.5]
    results = {}

    for min_corr in thresholds:
        logger.info("  Backtesting with threshold=%.1f", min_corr)
        snapshots = _fetch_snapshots(lookback_years=3, stride=10, min_corr=min_corr)
        if len(snapshots) < 20:
            logger.warning("  Skipping threshold %.1f (only %d snapshots)", min_corr, len(snapshots))
            continue

        scg_risk = np.array([
            float(1 - s["targets"]["lambda_2"] / s["targets"]["spectral_radius"])
            if s["targets"]["spectral_radius"] > 1e-8 else 1.0
            for s in snapshots
        ])
        vol = np.array([float(np.mean(s["node_features"][:, 0])) for s in snapshots])
        lam2 = np.array([s["targets"]["lambda_2"] for s in snapshots])
        gap = np.array([s["targets"]["spectral_gap"] for s in snapshots])

        def safe_corr(x, y):
            if len(x) < 5 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        def bootstrap_corr(x, y, n_boot=2000):
            rng = np.random.default_rng(GLOBAL_SEED)
            n = len(x)
            corrs = []
            for _ in range(n_boot):
                idx = rng.choice(n, size=n, replace=True)
                c = safe_corr(x[idx], y[idx])
                corrs.append(c)
            return {
                "corr": safe_corr(x, y),
                "ci_lo": float(np.percentile(corrs, 2.5)),
                "ci_hi": float(np.percentile(corrs, 97.5)),
            }

        horizon_results = {}
        for h in [1, 2, 5, 10]:
            if h >= len(snapshots) - 5:
                continue
            window = 3
            pairs = {"scg": ([], []), "vol": ([], []), "lam2": ([], []), "gap": ([], [])}
            for t in range(window, len(snapshots) - h):
                future_vol = vol[t + h]
                pairs["scg"][0].append(np.mean(scg_risk[t - window:t]))
                pairs["scg"][1].append(future_vol)
                pairs["vol"][0].append(np.mean(vol[t - window:t]))
                pairs["vol"][1].append(future_vol)
                pairs["lam2"][0].append(np.mean(lam2[t - window:t]))
                pairs["lam2"][1].append(future_vol)
                pairs["gap"][0].append(np.mean(gap[t - window:t]))
                pairs["gap"][1].append(future_vol)

            h_result = {}
            for ind_name in pairs:
                x_arr = np.array(pairs[ind_name][0])
                y_arr = np.array(pairs[ind_name][1])
                h_result[ind_name] = bootstrap_corr(x_arr, y_arr)
            horizon_results[f"h={h}"] = h_result

        results[f"corr_{min_corr}"] = {
            "min_corr": min_corr,
            "n_snapshots": len(snapshots),
            "scg_risk_stats": {"mean": round(float(np.mean(scg_risk)), 4),
                               "std": round(float(np.std(scg_risk)), 4)},
            "horizons": horizon_results,
        }

    # Figure: heatmap for best threshold
    best_key = "corr_0.3"
    if best_key in results:
        horizons = list(results[best_key]["horizons"].keys())
        indicators = ["scg", "vol", "lam2", "gap"]
        indicator_labels = ["SCG Risk", "Volatility", "$\\lambda_2$", "Spectral Gap"]
        heatmap = np.zeros((len(indicators), len(horizons)))
        for j, h in enumerate(horizons):
            for i, ind in enumerate(indicators):
                heatmap[i, j] = results[best_key]["horizons"][h].get(ind, {}).get("corr", 0)

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(heatmap, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels(horizons)
        ax.set_yticks(range(len(indicators)))
        ax.set_yticklabels(indicator_labels)
        ax.set_title("Predictive Correlations with Future Volatility (threshold=0.3)")
        plt.colorbar(im, ax=ax)
        # Annotate with CI
        for i in range(len(indicators)):
            for j in range(len(horizons)):
                h = horizons[j]
                ind = indicators[i]
                entry = results[best_key]["horizons"][h].get(ind, {})
                c = entry.get("corr", 0)
                ci_lo = entry.get("ci_lo", 0)
                ci_hi = entry.get("ci_hi", 0)
                ax.text(j, i, f"{c:.2f}\n[{ci_lo:.2f},{ci_hi:.2f}]",
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(c) > 0.25 else "black")
        _save_fig("exp13_backtesting_heatmap")

    _save_json(results, "exp13_backtesting")
    return results


# ======================================================================
# Main
# ======================================================================

def main():
    logger.info("=" * 70)
    logger.info("PUBLICATION EXPERIMENT SUITE")
    logger.info("=" * 70)
    logger.info("Results -> %s", RESULTS_DIR)
    logger.info("Figures -> %s", FIGURES_DIR)

    t_start = time.time()
    all_results = {}

    experiments = [
        ("exp01_multiscale_scg", exp01_multiscale_scg),
        ("exp02_rmt_denoising", exp02_rmt_denoising),
        ("exp03_network_construction", exp03_network_construction),
        ("exp04_walkforward_cv", exp04_walkforward_cv),
        ("exp05_per_target_analysis", exp05_per_target_analysis),
        ("exp06_ablation", exp06_ablation),
        ("exp07_feature_ablation", exp07_feature_ablation),
        ("exp08_conditional_spectral", exp08_conditional_spectral),
        ("exp09_change_points", exp09_change_points),
        ("exp10_abm_dynamics", exp10_abm_dynamics),
        ("exp11_abm_sensitivity", exp11_abm_sensitivity),
        ("exp12_threshold_sensitivity", exp12_threshold_sensitivity),
        ("exp13_backtesting", exp13_backtesting),
    ]

    for name, fn in experiments:
        try:
            result = fn()
            all_results[name] = "completed"
        except Exception as e:
            logger.error("Experiment %s FAILED: %s", name, e, exc_info=True)
            all_results[name] = f"FAILED: {e}"

    elapsed = time.time() - t_start

    # Summary
    summary = {
        "total_time_s": round(elapsed, 1),
        "experiments": all_results,
        "n_succeeded": sum(1 for v in all_results.values() if v == "completed"),
        "n_failed": sum(1 for v in all_results.values() if v != "completed"),
    }
    _save_json(summary, "summary")

    logger.info("=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE in %.1fs", elapsed)
    logger.info("  Succeeded: %d / %d", summary["n_succeeded"], len(experiments))
    if summary["n_failed"] > 0:
        for name, status in all_results.items():
            if status != "completed":
                logger.error("  FAILED: %s -> %s", name, status)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
