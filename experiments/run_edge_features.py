#!/usr/bin/env python3
"""
Experiment: Edge-level temporal features for GNN.

Hypothesis: The GNN fails because node features alone don't capture HOW the
network is changing. Edge features that encode correlation CHANGES between
bank pairs should help.

Approach: For each snapshot, compute edge-level temporal features, then
AGGREGATE them per node (mean, count) to create ~5 extra node features:
  - mean_edge_weight: average correlation of this node's edges
  - mean_delta_corr: average change in correlation from previous snapshot
  - mean_smoothed_corr: rolling 5-step average of correlation (smoothed trend)
  - edge_stability: fraction of current edges that also existed in previous snapshot
  - degree_change: change in node degree from previous snapshot

This gives 10 node features per bank (original 5 + 5 edge-aggregated).

Run with:
    cd /Users/francoispetizon/scr-financial-networks
    /opt/anaconda3/envs/systemic_risk/bin/python experiments/run_edge_features.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, '.')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("edge_features")

RESULTS_DIR = Path("experiments/results/publication")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 42


def _json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_json(data, name: str):
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_safe)
    logger.info("Saved %s", path)


# ── Step 1: Fetch prices and build snapshots with retry ──────────────────

def fetch_prices_with_retry(tickers, period="4y", max_attempts=3):
    """Download prices with retry logic."""
    import yfinance as yf
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("yfinance download attempt %d/%d (%d tickers)...",
                        attempt, max_attempts, len(tickers))
            raw = yf.download(tickers, period=period, progress=False, auto_adjust=True)
            prices = raw["Close"] if "Close" in raw else raw
            if prices.empty:
                logger.warning("Empty download on attempt %d", attempt)
                time.sleep(2 * attempt)
                continue
            logger.info("Downloaded %d rows x %d columns", len(prices), prices.shape[1])
            return prices
        except Exception as e:
            logger.warning("Attempt %d failed: %s", attempt, e)
            if attempt < max_attempts:
                time.sleep(3 * attempt)
    raise RuntimeError("All yfinance download attempts failed")


def build_base_snapshots(prices, ids, corr_window=60, min_corr=0.6, stride=3):
    """Build snapshots with 5 base node features (same as build_daily_graph_snapshots)."""
    from scr_financial.network.spectral import (
        compute_laplacian, eigendecomposition, find_spectral_gap,
        analyze_spectral_properties,
    )

    returns = prices.pct_change().dropna()
    n_banks = len(ids)
    logger.info("Building base snapshots: %d days, %d banks, corr_window=%d, "
                "threshold=%.1f, stride=%d",
                len(returns), n_banks, corr_window, min_corr, stride)

    snapshots = []
    valid_dates = returns.index[corr_window:]

    for count, date_idx in enumerate(range(0, len(valid_dates), stride)):
        date = valid_dates[date_idx]
        window_end = corr_window + date_idx
        ret_window = returns.iloc[window_end - corr_window: window_end]

        # Correlation matrix
        corr = ret_window.reindex(columns=ids).corr()
        corr_vals = corr.reindex(index=ids, columns=ids).values
        corr_vals = np.nan_to_num(corr_vals, nan=0.0)
        np.fill_diagonal(corr_vals, 1.0)

        # Adjacency matrix (threshold)
        adj = np.zeros((n_banks, n_banks), dtype=np.float32)
        for i in range(n_banks):
            for j in range(i + 1, n_banks):
                w = corr_vals[i, j]
                if w >= min_corr:
                    adj[i, j] = w
                    adj[j, i] = w

        # Node features: [N, 5]
        node_feats = np.zeros((n_banks, 5), dtype=np.float32)
        for i, bid in enumerate(ids):
            if bid in ret_window.columns:
                rets_i = ret_window[bid].values
                node_feats[i, 0] = float(np.std(rets_i) * np.sqrt(252))  # vol
                node_feats[i, 1] = float(np.mean(rets_i) * 252)  # return
                cum_ret = (1 + ret_window[bid]).prod()
                node_feats[i, 2] = float(np.log(max(cum_ret, 0.01)))  # log price
                mkt = ret_window.mean(axis=1).values
                cov = np.cov(rets_i, mkt)[0, 1] if len(rets_i) > 2 else 0
                var_mkt = np.var(mkt) if np.var(mkt) > 1e-10 else 1.0
                node_feats[i, 3] = float(cov / var_mkt)  # beta
                if len(rets_i) >= 20:
                    node_feats[i, 4] = float(np.sum(rets_i[-20:]))  # momentum

        # Edge index / weight
        rows, cols = np.nonzero(adj)
        edge_index = np.stack([rows, cols], axis=0).astype(np.int64) if len(rows) > 0 \
            else np.zeros((2, 0), dtype=np.int64)
        edge_weight = adj[rows, cols] if len(rows) > 0 else np.zeros(0, dtype=np.float32)

        # Spectral targets
        adj_sym = (adj + adj.T) / 2.0
        L = compute_laplacian(adj_sym, normalized=True)
        eigenvalues, eigenvectors = eigendecomposition(L)
        gap_idx, gap_size = find_spectral_gap(eigenvalues)
        props = analyze_spectral_properties(eigenvalues, eigenvectors)

        lam2 = float(props["algebraic_connectivity"])
        gap = float(gap_size)
        radius = float(props["spectral_radius"])

        snapshots.append({
            "node_features": node_feats,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "targets": {"lambda_2": lam2, "spectral_gap": gap, "spectral_radius": radius},
            "lambda_2": lam2,
            "spectral_gap": gap,
            "spectral_radius": radius,
            "time": count,
            "date": str(date.date()) if hasattr(date, 'date') else str(date),
        })

    logger.info("Built %d base snapshots (stride=%d)", len(snapshots), stride)
    return snapshots


# ── Step 2-3: Compute edge-aggregated node features ──────────────────────

def _adj_from_snapshot(snap: Dict[str, Any]) -> np.ndarray:
    """Reconstruct the full adjacency matrix from edge_index + edge_weight."""
    ei = snap["edge_index"]
    ew = snap["edge_weight"]
    nf = snap["node_features"]
    n = nf.shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    if ei.shape[1] > 0:
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            adj[i, j] = ew[k]
    return adj


def augment_snapshots_with_edge_features(
    snapshots: List[Dict[str, Any]],
    smooth_window: int = 5,
) -> List[Dict[str, Any]]:
    """Add 5 edge-aggregated node features to each snapshot.

    New features per node (appended to existing node_features):
      [5] mean_edge_weight      - mean correlation weight of this node's edges
      [6] mean_delta_corr       - mean change in edge correlation from prev snapshot
      [7] mean_smoothed_corr    - rolling mean of edge weight over `smooth_window` steps
      [8] edge_stability        - fraction of edges present in both current and prev snapshot
      [9] degree_change         - change in degree from previous snapshot (normalised)
    """
    if not snapshots:
        return snapshots

    n = snapshots[0]["node_features"].shape[0]
    n_new = 5

    # Precompute adjacency matrices for all snapshots
    adjs = [_adj_from_snapshot(s) for s in snapshots]

    # Rolling correlation history per node (for smoothing)
    mean_weight_history: List[np.ndarray] = []

    augmented = []
    for t, snap in enumerate(snapshots):
        adj = adjs[t]
        new_feats = np.zeros((n, n_new), dtype=np.float32)

        # Feature 0: mean edge weight per node
        degrees = (adj > 0).sum(axis=1).astype(np.float32)
        sum_weights = adj.sum(axis=1)
        mean_w = np.divide(sum_weights, degrees, out=np.zeros_like(sum_weights), where=degrees > 0)
        new_feats[:, 0] = mean_w
        mean_weight_history.append(mean_w.copy())

        # Feature 1: mean delta_corr per node (change from previous snapshot)
        if t > 0:
            prev_adj = adjs[t - 1]
            delta_adj = adj - prev_adj
            abs_delta = np.abs(delta_adj)
            delta_sum = abs_delta.sum(axis=1)
            d = np.maximum(degrees, 1.0)
            new_feats[:, 1] = delta_sum / d

        # Feature 2: smoothed mean edge weight (rolling average)
        start_idx = max(0, t - smooth_window + 1)
        if len(mean_weight_history) > 0:
            recent = mean_weight_history[start_idx: t + 1]
            new_feats[:, 2] = np.mean(recent, axis=0)

        # Feature 3: edge stability
        if t > 0:
            prev_adj = adjs[t - 1]
            current_edges = (adj > 0)
            prev_edges = (prev_adj > 0)
            shared = (current_edges & prev_edges).sum(axis=1).astype(np.float32)
            curr_deg = current_edges.sum(axis=1).astype(np.float32)
            stability = np.divide(shared, curr_deg, out=np.zeros(n, dtype=np.float32), where=curr_deg > 0)
            new_feats[:, 3] = stability
        else:
            new_feats[:, 3] = 1.0

        # Feature 4: degree change (normalised by n)
        if t > 0:
            prev_deg = (adjs[t - 1] > 0).sum(axis=1).astype(np.float32)
            degree_delta = degrees - prev_deg
            new_feats[:, 4] = degree_delta / max(n, 1)

        # Concatenate with original node features
        orig = snap["node_features"][:, :5]  # ensure only base 5
        combined = np.concatenate([orig, new_feats], axis=1)  # [n, 10]

        aug_snap = dict(snap)
        aug_snap["node_features"] = combined
        augmented.append(aug_snap)

    logger.info(
        "Augmented %d snapshots: node features %d -> %d",
        len(augmented), 5, augmented[0]["node_features"].shape[1],
    )
    return augmented


# ── Step 6-7: Train and evaluate ──────────────────────────────────────────

def run_walk_forward(snapshots, label, predictor_kwargs, cv_kwargs):
    """Run walk-forward CV and return results dict."""
    from scr_financial.ml.gnn_predictor import GNNPredictor
    from scr_financial.ml.walk_forward import walk_forward_evaluate

    np.random.seed(GLOBAL_SEED)
    import torch
    torch.manual_seed(GLOBAL_SEED)

    logger.info("Running walk-forward CV: %s (%d snapshots)", label, len(snapshots))
    t0 = time.time()
    results = walk_forward_evaluate(
        GNNPredictor,
        predictor_kwargs,
        snapshots,
        **cv_kwargs,
    )
    elapsed = time.time() - t0
    results["label"] = label
    results["total_time_s"] = round(elapsed, 1)
    results["n_snapshots"] = len(snapshots)

    agg = results["aggregate"]
    logger.info(
        "%s: R2 = %.4f +/- %.4f (median %.4f), MSE = %.6f, time = %.1fs",
        label,
        agg["test_r2_mean"], agg["test_r2_std"], agg.get("test_r2_median", 0),
        agg.get("test_mse_mean", 0), elapsed,
    )
    return results


def main():
    logger.info("=" * 70)
    logger.info("EXPERIMENT: Edge-level temporal features for GNN")
    logger.info("=" * 70)

    # --- Load eu_50 universe ---
    from scr_financial.config.loader import load_universe
    universe = load_universe("eu_50")
    ids = universe.ids
    tickers = universe.tickers
    logger.info("Universe: %s (%d banks)", universe.name, len(ids))

    # --- Step 1: Fetch prices with retry ---
    prices = fetch_prices_with_retry(tickers, period="4y", max_attempts=3)
    # Map ticker columns back to bank IDs
    rev = {b.ticker: b.id for b in universe.banks}
    prices.columns = [rev.get(str(c), str(c)) for c in prices.columns]
    # Keep only columns that are bank IDs
    valid_cols = [c for c in prices.columns if c in ids]
    prices = prices[valid_cols]
    # Drop banks with too many NaNs (>30%)
    nan_frac = prices.isna().mean()
    keep_cols = nan_frac[nan_frac < 0.3].index.tolist()
    dropped = [c for c in valid_cols if c not in keep_cols]
    if dropped:
        logger.info("Dropping %d banks with >30%% NaN: %s", len(dropped), dropped)
    prices = prices[keep_cols].dropna()
    ids = keep_cols
    logger.info("Final: %d banks, %d trading days", len(ids), len(prices))

    if len(prices) < 150:
        logger.error("Not enough price history (%d rows). Aborting.", len(prices))
        return

    # --- Build base snapshots (threshold=0.6, stride=3, corr_window=60) ---
    snapshots = build_base_snapshots(
        prices, ids, corr_window=60, min_corr=0.6, stride=3,
    )
    n_snaps = len(snapshots)
    logger.info("Total base snapshots: %d", n_snaps)

    if n_snaps < 100:
        logger.error("Not enough snapshots (%d). Need >= 100. Aborting.", n_snaps)
        return

    # --- Step 2-3: Augment with edge features ---
    augmented = augment_snapshots_with_edge_features(snapshots, smooth_window=5)

    # Verify shapes
    assert snapshots[0]["node_features"].shape[1] == 5
    assert augmented[0]["node_features"].shape[1] == 10
    n_banks = snapshots[0]["node_features"].shape[0]
    logger.info("Node features: baseline=%d, augmented=%d, n_banks=%d",
                5, 10, n_banks)

    # --- Override NODE_FEATURE_NAMES for baseline and augmented ---
    import scr_financial.ml.gnn_predictor as gnn_mod
    BASE_FEATURE_NAMES = [
        "volatility", "return", "log_price", "beta_proxy", "momentum",
    ]
    EDGE_FEATURE_NAMES = BASE_FEATURE_NAMES + [
        "mean_edge_weight", "mean_delta_corr", "mean_smoothed_corr",
        "edge_stability", "degree_change",
    ]

    # --- Shared CV and training config (small for speed) ---
    cv_kwargs = dict(
        n_splits=5,
        min_train_size=max(60, n_snaps // 4),
        test_size=max(20, n_snaps // 10),
        gap_size=5,
        epochs=150,
        lr=3e-3,
        patience=30,
        seed=GLOBAL_SEED,
    )

    baseline_kwargs = dict(
        seq_len=8,
        hidden_dim=48,
        num_gat_layers=2,
        num_lstm_layers=1,
        heads=4,
        dropout=0.1,
    )

    edge_kwargs = dict(
        seq_len=8,
        hidden_dim=48,
        num_gat_layers=2,
        num_lstm_layers=1,
        heads=4,
        dropout=0.1,
    )

    # --- Step 7: Run baseline (no edge features) ---
    gnn_mod.NODE_FEATURE_NAMES = BASE_FEATURE_NAMES
    logger.info("NODE_FEATURE_NAMES set to %d for baseline", len(gnn_mod.NODE_FEATURE_NAMES))
    baseline_results = run_walk_forward(
        snapshots, "baseline_5feat", baseline_kwargs, cv_kwargs
    )

    # --- Step 7: Run with edge-aggregated features ---
    gnn_mod.NODE_FEATURE_NAMES = EDGE_FEATURE_NAMES
    logger.info("NODE_FEATURE_NAMES set to %d for edge features", len(gnn_mod.NODE_FEATURE_NAMES))
    edge_results = run_walk_forward(
        augmented, "edge_features_10feat", edge_kwargs, cv_kwargs
    )

    # --- Summary ---
    b_r2 = baseline_results["aggregate"]["test_r2_mean"]
    e_r2 = edge_results["aggregate"]["test_r2_mean"]
    b_mse = baseline_results["aggregate"].get("test_mse_mean", 0)
    e_mse = edge_results["aggregate"].get("test_mse_mean", 0)

    print("\n" + "=" * 70)
    print("RESULTS: Edge-level temporal features experiment")
    print("=" * 70)
    print(f"  Snapshots: {n_snaps} (eu_50, N={n_banks} banks, threshold=0.6, stride=3, corr_window=60)")
    print(f"  Node features: baseline=5, edge-augmented=10")
    print(f"  Architecture: GAT(L=2,H=4,d=48) + LSTM(L=1), seq_len=8")
    print(f"  Walk-forward CV: 5 folds, patience=30, epochs<=150")
    print()
    print(f"  {'Model':<30s} {'R2 mean':>10s} {'R2 std':>10s} {'MSE mean':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Baseline (5 feat)':<30s} {b_r2:>10.4f} {baseline_results['aggregate']['test_r2_std']:>10.4f} {b_mse:>10.6f}")
    print(f"  {'Edge features (10 feat)':<30s} {e_r2:>10.4f} {edge_results['aggregate']['test_r2_std']:>10.4f} {e_mse:>10.6f}")
    print()
    delta = e_r2 - b_r2
    print(f"  Delta R2: {delta:+.4f} ({'improvement' if delta > 0 else 'no improvement'})")
    print()

    # Per-target breakdown
    print("  Per-target R2 (mean across folds):")
    for target in ["lambda_2", "spectral_gap", "spectral_radius"]:
        b_vals = [f["test_r2_per_target"].get(target, 0) for f in baseline_results["per_fold"]]
        e_vals = [f["test_r2_per_target"].get(target, 0) for f in edge_results["per_fold"]]
        b_mean = np.mean(b_vals) if b_vals else 0
        e_mean = np.mean(e_vals) if e_vals else 0
        print(f"    {target:<20s}: baseline={b_mean:.4f}, edge={e_mean:.4f}, delta={e_mean - b_mean:+.4f}")
    print("=" * 70)

    # --- Step 8: Save results ---
    output = {
        "experiment": "edge_level_temporal_features",
        "hypothesis": "Edge-aggregated temporal features (delta_corr, smoothed_corr, "
                      "edge_stability, degree_change) improve GNN spectral prediction "
                      "by encoding HOW the network is changing.",
        "data": {
            "universe": "eu_50",
            "n_banks_after_cleaning": n_banks,
            "threshold": 0.6,
            "stride": 3,
            "corr_window": 60,
            "lookback_years": 3,
            "n_snapshots": n_snaps,
        },
        "edge_features_added": [
            "mean_edge_weight",
            "mean_delta_corr",
            "mean_smoothed_corr (5-step rolling)",
            "edge_stability",
            "degree_change (normalised)",
        ],
        "baseline": {
            "node_features": 5,
            "walk_forward": baseline_results,
        },
        "edge_augmented": {
            "node_features": 10,
            "walk_forward": edge_results,
        },
        "comparison": {
            "baseline_r2": b_r2,
            "edge_r2": e_r2,
            "delta_r2": delta,
            "baseline_mse": b_mse,
            "edge_mse": e_mse,
            "conclusion": (
                "Edge features improve R2" if delta > 0.01
                else "Edge features show marginal/no improvement" if delta > -0.01
                else "Edge features hurt performance"
            ),
        },
    }

    _save_json(output, "exp_edge_features")
    logger.info("Done.")


if __name__ == "__main__":
    main()
