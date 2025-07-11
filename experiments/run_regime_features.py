#!/usr/bin/env python3
"""Experiment: Macro regime features improve GNN spectral prediction.

Hypothesis: GNN fails partly because spectral properties change in response to
market REGIME shifts (VIX spikes, yield inversions, correlation clustering) that
are absent from the 5 per-node features. Adding macro-regime indicators as
broadcast node features should improve both GNN and GradientBoosting prediction.

Macro features (broadcast to all nodes):
  1. vix_proxy       — realized vol of equal-weight bank portfolio (20d std * sqrt(252))
  2. yield_spread    — spread between highest and lowest sovereign yields
  3. corr_regime     — binary: is avg pairwise corr above its 60-day median?
  4. momentum_regime — fraction of banks with positive 20-day momentum
  5. vol_regime      — binary: is avg volatility above its 60-day rolling mean?

Feature interaction:
  6. vol_x_corr      — vix_proxy * corr_regime  (correlated stress)

Run:
    cd /Users/francoispetizon/scr-financial-networks
    /opt/anaconda3/envs/systemic_risk/bin/python experiments/run_regime_features.py
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
import pandas as pd

sys.path.insert(0, '.')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("regime_features")

RESULTS_DIR = Path("experiments/results/publication")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 42

# ── JSON-safe serialization ──────────────────────────────────────────

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


# ── Build snapshots with macro regime features ───────────────────────

def build_regime_snapshots(
    universe: str = "eu_50",
    lookback_years: int = 3,
    corr_window: int = 60,
    min_corr: float = 0.6,
    stride: int = 3,
) -> List[Dict[str, Any]]:
    """Build daily graph snapshots augmented with 5 macro regime features + 1 interaction.

    Returns snapshots with node_features shape [N, 11]:
      Original 5: vol_30d, mean_return_30d, log_price, beta_proxy, momentum_20d
      Macro 5 (broadcast): vix_proxy, yield_spread, corr_regime, momentum_regime, vol_regime
      Interaction 1: vol_x_corr = vix_proxy * corr_regime
    """
    from scr_financial.config.loader import load_universe
    from scr_financial.network.spectral import (
        compute_laplacian, eigendecomposition, find_spectral_gap,
        analyze_spectral_properties,
    )
    from dashboard.data_api import _fetch_prices, BANK_TICKERS, fetch_sovereign_spreads

    # Resolve universe
    univ = load_universe(universe)
    ids = univ.ids
    for b in univ.banks:
        if b.id not in BANK_TICKERS:
            BANK_TICKERS[b.id] = b.ticker
    n_banks = len(ids)

    # Fetch prices
    prices = _fetch_prices(ids, period=f"{lookback_years * 365 + 60}d")
    if prices.empty:
        logger.error("No price data returned")
        return []

    # Drop banks with insufficient data (delisted, etc.)
    min_rows = corr_window + 60
    good_cols = [c for c in prices.columns if prices[c].notna().sum() >= min_rows]
    prices = prices[good_cols].dropna(how='all')
    if len(good_cols) < 5:
        logger.error("Only %d banks with sufficient data", len(good_cols))
        return []

    # Update ids to only include banks with data
    ids = [bid for bid in ids if bid in good_cols]
    n_banks = len(ids)
    prices = prices[ids]

    # Forward-fill small gaps, then drop remaining NaN rows
    prices = prices.ffill(limit=5)
    returns = prices.pct_change(fill_method=None).dropna()
    if len(returns) < min_rows:
        logger.error("Insufficient return history: %d rows (need %d)", len(returns), min_rows)
        return []

    logger.info("Price data: %d trading days, %d banks (after filtering)", len(returns), n_banks)

    # --- Precompute macro time series across ALL dates ---
    # Equal-weight portfolio returns
    ew_returns = returns.mean(axis=1)

    # Rolling 20-day realized vol (annualized) = VIX proxy
    vix_proxy_series = ew_returns.rolling(20).std() * np.sqrt(252)

    # Per-bank 30-day rolling vol (annualized)
    bank_vols = returns.rolling(30).std() * np.sqrt(252)
    avg_vol_series = bank_vols.mean(axis=1)

    # Per-bank 20-day momentum (cumulative return)
    bank_momentum_20d = returns.rolling(20).sum()

    # Rolling pairwise correlation average (60-day window)
    # Too expensive to compute full corr every day, so use a shortcut:
    # avg_corr ≈ (var(portfolio) / avg_var_individual - 1) / (N-1) * N
    # But simpler: rolling corr between EW portfolio and each stock, averaged
    # Instead, compute rolling correlation matrix at each snapshot date below

    # --- Yield spread: fetch sovereign yields ---
    # Try ECB sovereign yields for spread between highest and lowest
    try:
        sovereign = fetch_sovereign_spreads()
        # Get all country yields (exclude spread entries)
        yield_vals = {k: v for k, v in sovereign.items()
                      if k in ("DE", "FR", "IT", "ES", "NL", "SE") and v is not None}
        if len(yield_vals) >= 2:
            yield_spread_static = max(yield_vals.values()) - min(yield_vals.values())
            logger.info("Sovereign yield spread: %.3f%% (%s)", yield_spread_static, yield_vals)
        else:
            yield_spread_static = 1.5  # fallback
            logger.warning("Insufficient sovereign data, using fallback yield spread=%.1f", yield_spread_static)
    except Exception as e:
        yield_spread_static = 1.5
        logger.warning("Sovereign fetch failed (%s), using fallback yield spread=%.1f", e, yield_spread_static)

    # We'll create a synthetic time-varying yield spread from the VIX proxy
    # (since we only get a single current snapshot from ECB).
    # Scale: high-vol days -> wider spread
    vix_normed = vix_proxy_series / vix_proxy_series.rolling(252, min_periods=60).mean()
    yield_spread_series = yield_spread_static * vix_normed.fillna(1.0)

    # --- Build snapshots ---
    snapshots: List[Dict[str, Any]] = []
    valid_dates = returns.index[corr_window:]
    total = len(range(0, len(valid_dates), stride))
    logger.info("Building %d regime-augmented snapshots (stride=%d, threshold=%.1f)",
                total, stride, min_corr)

    # Precompute rolling medians for regime indicators
    avg_vol_rolling_mean = avg_vol_series.rolling(60, min_periods=30).mean()

    for count, date_idx in enumerate(range(0, len(valid_dates), stride)):
        date = valid_dates[date_idx]
        window_end = corr_window + date_idx
        ret_window = returns.iloc[window_end - corr_window: window_end]

        # --- Correlation adjacency ---
        corr = ret_window.corr()
        n = n_banks
        adj = np.zeros((n, n), dtype=np.float32)
        for i, src in enumerate(ids):
            for j, tgt in enumerate(ids):
                if i >= j or src not in corr.index or tgt not in corr.index:
                    continue
                w = float(corr.loc[src, tgt])
                if w >= min_corr:
                    adj[i, j] = w
                    adj[j, i] = w

        # --- Original node features [N, 5] ---
        node_feats_base = np.zeros((n, 5), dtype=np.float32)
        for i, bid in enumerate(ids):
            if bid in ret_window.columns:
                rets_i = ret_window[bid].values
                node_feats_base[i, 0] = float(np.std(rets_i) * np.sqrt(252))
                node_feats_base[i, 1] = float(np.mean(rets_i) * 252)
                cum_ret = (1 + ret_window[bid]).prod()
                node_feats_base[i, 2] = float(np.log(max(cum_ret, 0.01)))
                mkt = ret_window.mean(axis=1).values
                cov_val = np.cov(rets_i, mkt)[0, 1] if len(rets_i) > 2 else 0
                var_mkt = np.var(mkt) if np.var(mkt) > 1e-10 else 1.0
                node_feats_base[i, 3] = float(cov_val / var_mkt)
                if len(rets_i) >= 20:
                    node_feats_base[i, 4] = float(np.sum(rets_i[-20:]))

        # --- Macro feature 1: VIX proxy (realized vol) ---
        vix_val = vix_proxy_series.iloc[window_end - 1] if window_end - 1 < len(vix_proxy_series) else 0.2
        if np.isnan(vix_val):
            vix_val = 0.2

        # --- Macro feature 2: Yield spread ---
        ys_val = yield_spread_series.iloc[window_end - 1] if window_end - 1 < len(yield_spread_series) else yield_spread_static
        if np.isnan(ys_val):
            ys_val = yield_spread_static

        # --- Macro feature 3: Correlation regime ---
        # Average pairwise correlation from today's corr matrix
        corr_vals = corr.reindex(index=ids, columns=ids).values
        np.fill_diagonal(corr_vals, np.nan)
        avg_corr_today = float(np.nanmean(corr_vals))

        # For the regime: compare with 60-day rolling median of avg_corr
        # We approximate: use a running list
        if not hasattr(build_regime_snapshots, '_corr_history'):
            build_regime_snapshots._corr_history = []
        build_regime_snapshots._corr_history.append(avg_corr_today)
        hist = build_regime_snapshots._corr_history
        if len(hist) >= 20:  # use available history (up to 60 points given stride)
            median_corr = float(np.median(hist[-60:]))
        else:
            median_corr = avg_corr_today
        corr_regime = 1.0 if avg_corr_today > median_corr else 0.0

        # --- Macro feature 4: Momentum regime ---
        # Fraction of banks with positive 20-day momentum
        mom_vals = bank_momentum_20d.iloc[window_end - 1] if window_end - 1 < len(bank_momentum_20d) else pd.Series(0, index=ids)
        mom_positive_frac = 0.5
        if hasattr(mom_vals, 'values'):
            valid_mom = mom_vals.reindex(ids).dropna()
            if len(valid_mom) > 0:
                mom_positive_frac = float((valid_mom > 0).sum() / len(valid_mom))

        # --- Macro feature 5: Volatility regime ---
        avg_vol_today = avg_vol_series.iloc[window_end - 1] if window_end - 1 < len(avg_vol_series) else 0.2
        avg_vol_mean_60 = avg_vol_rolling_mean.iloc[window_end - 1] if window_end - 1 < len(avg_vol_rolling_mean) else avg_vol_today
        if np.isnan(avg_vol_today):
            avg_vol_today = 0.2
        if np.isnan(avg_vol_mean_60):
            avg_vol_mean_60 = avg_vol_today
        vol_regime = 1.0 if avg_vol_today > avg_vol_mean_60 else 0.0

        # --- Feature interaction: vol x correlation regime ---
        vol_x_corr = vix_val * corr_regime

        # --- Broadcast macro features to all nodes: [N, 6] ---
        macro_feats = np.zeros((n, 6), dtype=np.float32)
        macro_feats[:, 0] = vix_val
        macro_feats[:, 1] = ys_val
        macro_feats[:, 2] = corr_regime
        macro_feats[:, 3] = mom_positive_frac
        macro_feats[:, 4] = vol_regime
        macro_feats[:, 5] = vol_x_corr

        # Concatenate: [N, 11]
        node_feats = np.hstack([node_feats_base, macro_feats])

        # --- Edge index / weight ---
        rows, cols = np.nonzero(adj)
        edge_index = np.stack([rows, cols], axis=0).astype(np.int64) if len(rows) > 0 \
            else np.zeros((2, 0), dtype=np.int64)
        edge_weight = adj[rows, cols] if len(rows) > 0 else np.zeros(0, dtype=np.float32)

        # --- Spectral targets ---
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
            # Store macro values for flat-feature extraction
            "macro": {
                "vix_proxy": float(vix_val),
                "yield_spread": float(ys_val),
                "corr_regime": float(corr_regime),
                "momentum_regime": float(mom_positive_frac),
                "vol_regime": float(vol_regime),
                "vol_x_corr": float(vol_x_corr),
            },
        })

        if (count + 1) % 50 == 0:
            logger.info("  Snapshot %d/%d (%s)", count + 1, total, date.date())

    # Reset corr history for re-runs
    build_regime_snapshots._corr_history = []

    logger.info("Built %d regime-augmented snapshots (stride=%d)", len(snapshots), stride)
    return snapshots


# ── Extended flat features for GradientBoosting ──────────────────────

def extract_extended_flat_features(sequences):
    """Extract 36 graph features + 5 macro features + 1 interaction = 42 features.

    Graph features (36):
      Per-target (lambda_2, spectral_gap, spectral_radius) x 4 stats:
        - last, mean, std, trend  = 12
      Node feature stats:
        - mean, std, min, max of each of the 5 base node features = 20
      Graph structure:
        - n_edges, avg_edge_weight, density, max_degree = 4

    Macro features (6):
      vix_proxy, yield_spread, corr_regime, momentum_regime, vol_regime, vol_x_corr
    """
    features = []
    for seq in sequences:
        feat_parts = []

        # Spectral target stats (12)
        tgt_arr = np.array([
            [s["targets"]["lambda_2"], s["targets"]["spectral_gap"],
             s["targets"]["spectral_radius"]] for s in seq
        ])
        feat_parts.append(tgt_arr[-1])           # last (3)
        feat_parts.append(np.mean(tgt_arr, 0))   # mean (3)
        feat_parts.append(np.std(tgt_arr, 0))    # std (3)
        feat_parts.append(tgt_arr[-1] - tgt_arr[0])  # trend (3)

        # Node feature stats from last snapshot - base features only (20)
        nf = seq[-1]["node_features"][:, :5]  # [N, 5] base features
        feat_parts.append(np.mean(nf, 0))    # mean (5)
        feat_parts.append(np.std(nf, 0))     # std (5)
        feat_parts.append(np.min(nf, 0))     # min (5)
        feat_parts.append(np.max(nf, 0))     # max (5)

        # Graph structure (4)
        n_edges = seq[-1]["edge_index"].shape[1] / 2 if seq[-1]["edge_index"].shape[1] > 0 else 0
        ew = seq[-1]["edge_weight"]
        avg_ew = float(np.mean(ew)) if len(ew) > 0 else 0
        n_nodes = seq[-1]["node_features"].shape[0]
        density = n_edges / max(1, n_nodes * (n_nodes - 1) / 2)
        # Max degree
        if seq[-1]["edge_index"].shape[1] > 0:
            ei = seq[-1]["edge_index"]
            degrees = np.bincount(ei[0], minlength=n_nodes)
            max_deg = float(np.max(degrees))
        else:
            max_deg = 0.0
        feat_parts.append(np.array([n_edges, avg_ew, density, max_deg]))

        # Macro features from last snapshot (6)
        macro = seq[-1].get("macro", {})
        feat_parts.append(np.array([
            macro.get("vix_proxy", 0),
            macro.get("yield_spread", 0),
            macro.get("corr_regime", 0),
            macro.get("momentum_regime", 0.5),
            macro.get("vol_regime", 0),
            macro.get("vol_x_corr", 0),
        ]))

        features.append(np.concatenate(feat_parts))
    return np.array(features)


# ── Main experiment ──────────────────────────────────────────────────

def main():
    import torch
    from scr_financial.ml.gnn_predictor import GNNPredictor, NODE_FEATURE_NAMES
    from scr_financial.ml.walk_forward import walk_forward_evaluate
    from scr_financial.ml.baselines import GradientBoostingBaseline, extract_flat_features

    t_start = time.time()

    # --- Step 1: Build regime-augmented snapshots ---
    logger.info("=" * 70)
    logger.info("EXPERIMENT: Macro Regime Features for GNN Spectral Prediction")
    logger.info("=" * 70)

    snapshots = build_regime_snapshots(
        universe="eu_50",
        lookback_years=5,
        corr_window=60,
        min_corr=0.6,
        stride=3,
    )
    if len(snapshots) < 40:
        logger.error("Insufficient snapshots (%d), aborting", len(snapshots))
        return

    n_snaps = len(snapshots)
    n_features = snapshots[0]["node_features"].shape[1]
    n_banks = snapshots[0]["node_features"].shape[0]
    logger.info("Snapshots: %d, Banks: %d, Node features: %d", n_snaps, n_banks, n_features)

    # Print macro feature summary
    macro_keys = ["vix_proxy", "yield_spread", "corr_regime", "momentum_regime", "vol_regime", "vol_x_corr"]
    for key in macro_keys:
        vals = [s["macro"][key] for s in snapshots]
        logger.info("  %s: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                     key, np.mean(vals), np.std(vals), np.min(vals), np.max(vals))

    results = {
        "experiment": "regime_features",
        "description": "Macro regime indicators as GNN node features",
        "n_snapshots": n_snaps,
        "n_banks": n_banks,
        "n_node_features": n_features,
        "feature_names": {
            "base": ["vol_30d", "mean_return_30d", "log_price", "beta_proxy", "momentum_20d"],
            "macro": ["vix_proxy", "yield_spread", "corr_regime", "momentum_regime", "vol_regime"],
            "interaction": ["vol_x_corr"],
        },
        "config": {
            "universe": "eu_50",
            "lookback_years": 5,
            "corr_window": 60,
            "min_corr": 0.6,
            "stride": 3,
        },
        "macro_summary": {},
    }
    for key in macro_keys:
        vals = [s["macro"][key] for s in snapshots]
        results["macro_summary"][key] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }

    # --- Step 2: Walk-forward CV for GNN with regime features ---
    logger.info("\n--- GNN with regime features (hidden=8, layers=1, heads=2) ---")

    # Monkey-patch NODE_FEATURE_NAMES so GNNPredictor uses the right count
    import scr_financial.ml.gnn_predictor as gnn_mod
    original_names = gnn_mod.NODE_FEATURE_NAMES
    gnn_mod.NODE_FEATURE_NAMES = [
        "vol_30d", "mean_return_30d", "log_price", "beta_proxy", "momentum_20d",
        "vix_proxy", "yield_spread", "corr_regime", "momentum_regime", "vol_regime",
        "vol_x_corr",
    ]

    # CV parameters -- adjusted for data size
    # With seq_len=10, we have n_snaps - 10 usable sequences
    n_seqs = n_snaps - 10
    logger.info("Usable sequences for CV: %d", n_seqs)

    # Compute CV sizes that actually fit the data
    # WalkForwardCV needs: min_train + gap + test + (n_splits-1)*step <= n_seqs
    test_sz = max(5, n_seqs // 8)
    gap_sz = 2
    min_train = max(15, n_seqs // 4)
    # Verify there's room for at least 3 folds
    available = n_seqs - min_train - gap_sz - test_sz
    if available <= 0:
        # Shrink to fit
        min_train = max(12, n_seqs // 3)
        test_sz = max(4, n_seqs // 10)
        gap_sz = 1
    n_cv_splits = min(5, max(2, available // max(1, test_sz // 2)))
    n_cv_splits = max(2, min(n_cv_splits, 5))

    cv_kwargs = dict(
        n_splits=n_cv_splits,
        min_train_size=min_train,
        test_size=test_sz,
        gap_size=gap_sz,
    )
    logger.info("CV config: %s", cv_kwargs)

    gnn_config = dict(
        seq_len=10,
        hidden_dim=8,
        num_gat_layers=1,
        num_lstm_layers=1,
        heads=2,
        dropout=0.1,
    )

    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    gnn_result = walk_forward_evaluate(
        GNNPredictor, gnn_config, snapshots,
        epochs=200, lr=3e-3, patience=30, seed=GLOBAL_SEED,
        **cv_kwargs,
    )
    results["gnn_regime"] = gnn_result

    gnn_r2 = gnn_result["aggregate"]["test_r2_mean"]
    gnn_r2_std = gnn_result["aggregate"]["test_r2_std"]
    logger.info("GNN (regime features) walk-forward R2: %.4f +/- %.4f", gnn_r2, gnn_r2_std)

    # Per-fold details
    for fold in gnn_result["per_fold"]:
        logger.info("  Fold %d: R2=%.4f, MSE=%.6f, time=%.1fs",
                     fold["fold"], fold["test_r2"], fold["test_mse"], fold["training_time_s"])

    # --- Step 3: GradientBoosting on extended features ---
    logger.info("\n--- GradientBoosting on 36 graph + 6 macro = 42 features ---")

    seq_len = 10
    TARGET_NAMES = ["lambda_2", "spectral_gap", "spectral_radius"]
    sequences, targets = [], []
    for i in range(len(snapshots) - seq_len):
        sequences.append(snapshots[i: i + seq_len])
        tgt = snapshots[i + seq_len]
        targets.append([tgt["targets"][k] for k in TARGET_NAMES])
    targets = np.array(targets)

    # Walk-forward CV for GB too
    from scr_financial.ml.walk_forward import WalkForwardCV
    cv = WalkForwardCV(**cv_kwargs)
    gb_fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(len(sequences))):
        train_seqs = [sequences[i] for i in train_idx]
        test_seqs = [sequences[i] for i in test_idx]
        y_train = targets[train_idx]
        y_test = targets[test_idx]

        X_train = extract_extended_flat_features(train_seqs)
        X_test = extract_extended_flat_features(test_seqs)

        gb = GradientBoostingBaseline(n_estimators=200, seed=GLOBAL_SEED + fold_i)
        t0 = time.time()
        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_test)
        train_time = time.time() - t0

        # Metrics
        ss_res = np.sum((y_test - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_test - np.mean(y_test, axis=0)) ** 2, axis=0)
        r2_per = {}
        for ti, tn in enumerate(TARGET_NAMES):
            r2_per[tn] = float(1 - ss_res[ti] / ss_tot[ti]) if ss_tot[ti] > 1e-8 else 0.0

        mse = float(np.mean((y_test - y_pred) ** 2))
        r2_avg = float(np.mean(list(r2_per.values())))

        gb_fold_results.append({
            "fold": fold_i,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "test_r2": round(r2_avg, 4),
            "test_mse": round(mse, 6),
            "test_r2_per_target": {k: round(v, 4) for k, v in r2_per.items()},
            "training_time_s": round(train_time, 1),
            "n_features": X_train.shape[1],
        })
        logger.info("  GB Fold %d: R2=%.4f, MSE=%.6f, features=%d, time=%.1fs",
                     fold_i, r2_avg, mse, X_train.shape[1], train_time)

    gb_r2s = [f["test_r2"] for f in gb_fold_results]
    gb_mses = [f["test_mse"] for f in gb_fold_results]
    gb_aggregate = {
        "test_r2_mean": round(float(np.mean(gb_r2s)), 4),
        "test_r2_std": round(float(np.std(gb_r2s)), 4),
        "test_r2_median": round(float(np.median(gb_r2s)), 4),
        "test_mse_mean": round(float(np.mean(gb_mses)), 6),
        "test_mse_std": round(float(np.std(gb_mses)), 6),
        "n_folds_completed": len(gb_fold_results),
    }

    results["gb_regime"] = {
        "per_fold": gb_fold_results,
        "aggregate": gb_aggregate,
        "cv_config": cv_kwargs,
    }

    gb_r2 = gb_aggregate["test_r2_mean"]
    gb_r2_std = gb_aggregate["test_r2_std"]
    logger.info("GB (regime features) walk-forward R2: %.4f +/- %.4f", gb_r2, gb_r2_std)

    # --- Step 4: Also run baseline GNN without macro features for comparison ---
    logger.info("\n--- Baseline GNN (no macro features, 5 node features) ---")

    # Build baseline snapshots (strip macro features)
    baseline_snapshots = []
    for s in snapshots:
        bs = dict(s)
        bs["node_features"] = s["node_features"][:, :5].copy()
        baseline_snapshots.append(bs)

    gnn_mod.NODE_FEATURE_NAMES = original_names  # restore original 5 features

    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    baseline_result = walk_forward_evaluate(
        GNNPredictor, gnn_config, baseline_snapshots,
        epochs=200, lr=3e-3, patience=30, seed=GLOBAL_SEED,
        **cv_kwargs,
    )
    results["gnn_baseline"] = baseline_result

    baseline_r2 = baseline_result["aggregate"]["test_r2_mean"]
    baseline_r2_std = baseline_result["aggregate"]["test_r2_std"]
    logger.info("GNN (baseline, no macro) walk-forward R2: %.4f +/- %.4f", baseline_r2, baseline_r2_std)

    # Restore patched names for regime model record
    gnn_mod.NODE_FEATURE_NAMES = original_names

    # --- Step 5: Also run baseline GB without macro features ---
    logger.info("\n--- Baseline GB (graph features only, no macro) ---")

    from scr_financial.ml.baselines import extract_flat_features as extract_basic_flat

    gb_base_folds = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(len(sequences))):
        train_seqs = [sequences[i] for i in train_idx]
        test_seqs = [sequences[i] for i in test_idx]
        y_train = targets[train_idx]
        y_test = targets[test_idx]

        X_train = extract_basic_flat(train_seqs)
        X_test = extract_basic_flat(test_seqs)

        gb = GradientBoostingBaseline(n_estimators=200, seed=GLOBAL_SEED + fold_i)
        t0 = time.time()
        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_test)
        train_time = time.time() - t0

        ss_res = np.sum((y_test - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_test - np.mean(y_test, axis=0)) ** 2, axis=0)
        r2_per = {}
        for ti, tn in enumerate(TARGET_NAMES):
            r2_per[tn] = float(1 - ss_res[ti] / ss_tot[ti]) if ss_tot[ti] > 1e-8 else 0.0
        r2_avg = float(np.mean(list(r2_per.values())))
        mse = float(np.mean((y_test - y_pred) ** 2))

        gb_base_folds.append({
            "fold": fold_i,
            "test_r2": round(r2_avg, 4),
            "test_mse": round(mse, 6),
            "n_features": X_train.shape[1],
        })
        logger.info("  GB Base Fold %d: R2=%.4f, features=%d", fold_i, r2_avg, X_train.shape[1])

    gb_base_r2s = [f["test_r2"] for f in gb_base_folds]
    gb_base_agg = {
        "test_r2_mean": round(float(np.mean(gb_base_r2s)), 4),
        "test_r2_std": round(float(np.std(gb_base_r2s)), 4),
        "n_folds_completed": len(gb_base_folds),
    }
    results["gb_baseline"] = {
        "per_fold": gb_base_folds,
        "aggregate": gb_base_agg,
    }

    gb_base_r2_mean = gb_base_agg["test_r2_mean"]
    logger.info("GB (baseline, no macro) walk-forward R2: %.4f +/- %.4f",
                gb_base_r2_mean, gb_base_agg["test_r2_std"])

    # --- Summary ---
    elapsed = time.time() - t_start
    results["timing_total_s"] = round(elapsed, 1)

    summary = {
        "gnn_regime_r2": f"{gnn_r2:.4f} +/- {gnn_r2_std:.4f}",
        "gnn_baseline_r2": f"{baseline_r2:.4f} +/- {baseline_r2_std:.4f}",
        "gnn_improvement": f"{gnn_r2 - baseline_r2:+.4f}",
        "gb_regime_r2": f"{gb_r2:.4f} +/- {gb_r2_std:.4f}",
        "gb_baseline_r2": f"{gb_base_r2_mean:.4f} +/- {gb_base_agg['test_r2_std']:.4f}",
        "gb_improvement": f"{gb_r2 - gb_base_r2_mean:+.4f}",
    }
    results["summary"] = summary

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("GNN (regime features):   R2 = %s", summary["gnn_regime_r2"])
    logger.info("GNN (baseline):          R2 = %s", summary["gnn_baseline_r2"])
    logger.info("GNN improvement:         %s", summary["gnn_improvement"])
    logger.info("")
    logger.info("GB  (regime features):   R2 = %s", summary["gb_regime_r2"])
    logger.info("GB  (baseline):          R2 = %s", summary["gb_baseline_r2"])
    logger.info("GB  improvement:         %s", summary["gb_improvement"])
    logger.info("")
    logger.info("Total time: %.1fs", elapsed)

    # --- Save ---
    out_path = RESULTS_DIR / "exp_regime_features.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
