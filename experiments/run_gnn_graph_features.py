#!/usr/bin/env python3
"""
Experiment: GNN with broadcast graph-level features.

Key insight: GradientBoosting found corr_std has ~80% feature importance for
predicting lambda_2. We broadcast 10 graph-level features to ALL nodes so
the GNN sees both node-level AND graph-level information.

Graph-level features (broadcast to every node):
  corr_std, corr_mean, density, avg_clustering, n_components,
  avg_degree, edge_change_abs, lam2_change, avg_vol, vol_dispersion

Node-level features (existing 5):
  volatility, return, log_price, beta_proxy, momentum

Total: 15 features per node.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("gnn_graph_features")

RESULTS_DIR = Path("experiments/results/publication")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ── JSON-safe helper ──────────────────────────────────────────────────────────

def _json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Step 1: Fetch eu_50 data via yfinance (with retry) ───────────────────────

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


# ── Step 2-3: Build snapshots with graph-level features ───────────────────────

def build_enriched_snapshots(
    prices, ids, corr_window=60, min_corr=0.6, stride=3,
):
    """Build snapshots with 10 graph-level features broadcast to all nodes.

    Returns list of snapshot dicts with node_features shape [N, 15].
    """
    import networkx as nx
    from scr_financial.network.spectral import (
        compute_laplacian, eigendecomposition, find_spectral_gap,
        analyze_spectral_properties,
    )

    returns = prices.pct_change().dropna()
    n_banks = len(ids)
    logger.info("Building enriched snapshots: %d days, %d banks, corr_window=%d, "
                "threshold=%.1f, stride=%d",
                len(returns), n_banks, corr_window, min_corr, stride)

    snapshots = []
    valid_dates = returns.index[corr_window:]
    prev_adj = None
    prev_lam2 = None

    for count, date_idx in enumerate(range(0, len(valid_dates), stride)):
        date = valid_dates[date_idx]
        window_end = corr_window + date_idx
        ret_window = returns.iloc[window_end - corr_window: window_end]

        # --- Correlation matrix ---
        corr = ret_window.reindex(columns=ids).corr()
        corr_vals = corr.reindex(index=ids, columns=ids).values
        corr_vals = np.nan_to_num(corr_vals, nan=0.0)
        np.fill_diagonal(corr_vals, 1.0)

        # Upper-triangle correlations (excluding diagonal)
        upper = corr_vals[np.triu_indices(n_banks, k=1)]

        # --- Adjacency matrix (threshold) ---
        adj = np.zeros((n_banks, n_banks), dtype=np.float32)
        for i in range(n_banks):
            for j in range(i + 1, n_banks):
                w = corr_vals[i, j]
                if w >= min_corr:
                    adj[i, j] = w
                    adj[j, i] = w

        # --- Node features (5 existing) ---
        node_feats_base = np.zeros((n_banks, 5), dtype=np.float32)
        for i, bid in enumerate(ids):
            if bid in ret_window.columns:
                rets_i = ret_window[bid].values
                node_feats_base[i, 0] = float(np.std(rets_i) * np.sqrt(252))  # vol
                node_feats_base[i, 1] = float(np.mean(rets_i) * 252)  # return
                cum_ret = (1 + ret_window[bid]).prod()
                node_feats_base[i, 2] = float(np.log(max(cum_ret, 0.01)))  # log price
                mkt = ret_window.mean(axis=1).values
                cov = np.cov(rets_i, mkt)[0, 1] if len(rets_i) > 2 else 0
                var_mkt = np.var(mkt) if np.var(mkt) > 1e-10 else 1.0
                node_feats_base[i, 3] = float(cov / var_mkt)  # beta
                if len(rets_i) >= 20:
                    node_feats_base[i, 4] = float(np.sum(rets_i[-20:]))  # momentum

        # --- 10 graph-level features ---
        # 1. corr_std
        corr_std = float(np.std(upper))
        # 2. corr_mean
        corr_mean = float(np.mean(upper))
        # 3. density
        n_edges = np.sum(adj > 0) / 2
        max_edges = n_banks * (n_banks - 1) / 2
        density = float(n_edges / max_edges) if max_edges > 0 else 0.0
        # 4. avg_clustering
        G = nx.from_numpy_array(adj)
        avg_clustering = float(nx.average_clustering(G, weight='weight'))
        # 5. n_components
        n_components = float(nx.number_connected_components(G))
        # 6. avg_degree
        degrees = np.sum(adj > 0, axis=1)
        avg_degree = float(np.mean(degrees))
        # 7. edge_change_abs (vs previous snapshot)
        if prev_adj is not None:
            edge_change_abs = float(np.sum(np.abs((adj > 0).astype(float) -
                                                   (prev_adj > 0).astype(float))) / 2)
        else:
            edge_change_abs = 0.0
        # 8. Spectral targets (also used for lam2_change)
        adj_sym = (adj + adj.T) / 2.0
        L = compute_laplacian(adj_sym, normalized=True)
        eigenvalues, eigenvectors = eigendecomposition(L)
        gap_idx, gap_size = find_spectral_gap(eigenvalues)
        props = analyze_spectral_properties(eigenvalues, eigenvectors)
        lam2 = float(props["algebraic_connectivity"])
        gap = float(gap_size)
        radius = float(props["spectral_radius"])

        if prev_lam2 is not None:
            lam2_change = lam2 - prev_lam2
        else:
            lam2_change = 0.0
        # 9. avg_vol (average node volatility)
        avg_vol = float(np.mean(node_feats_base[:, 0]))
        # 10. vol_dispersion (std of node volatilities)
        vol_dispersion = float(np.std(node_feats_base[:, 0]))

        # --- BROADCAST graph features to all nodes ---
        graph_feats = np.array([
            corr_std, corr_mean, density, avg_clustering, n_components,
            avg_degree, edge_change_abs, lam2_change, avg_vol, vol_dispersion
        ], dtype=np.float32)
        # Tile to [N, 10]
        graph_feats_broadcast = np.tile(graph_feats, (n_banks, 1))

        # CONCATENATE: [N, 5] + [N, 10] = [N, 15]
        node_feats_full = np.concatenate([node_feats_base, graph_feats_broadcast], axis=1)

        # Edge index / weight
        rows, cols = np.nonzero(adj)
        edge_index = (np.stack([rows, cols], axis=0).astype(np.int64)
                      if len(rows) > 0 else np.zeros((2, 0), dtype=np.int64))
        edge_weight = adj[rows, cols] if len(rows) > 0 else np.zeros(0, dtype=np.float32)

        snapshots.append({
            "node_features": node_feats_full,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "targets": {"lambda_2": lam2, "spectral_gap": gap, "spectral_radius": radius},
            "lambda_2": lam2,
            "spectral_gap": gap,
            "spectral_radius": radius,
            "time": count,
            "date": str(date.date()) if hasattr(date, 'date') else str(date),
            # Store graph-level features for analysis
            "_graph_features": {
                "corr_std": corr_std, "corr_mean": corr_mean, "density": density,
                "avg_clustering": avg_clustering, "n_components": n_components,
                "avg_degree": avg_degree, "edge_change_abs": edge_change_abs,
                "lam2_change": lam2_change, "avg_vol": avg_vol, "vol_dispersion": vol_dispersion,
            },
        })

        prev_adj = adj.copy()
        prev_lam2 = lam2

        if (count + 1) % 50 == 0:
            logger.info("  snapshot %d / ~%d  (date=%s, lam2=%.4f, corr_std=%.4f, density=%.3f)",
                        count + 1, len(range(0, len(valid_dates), stride)),
                        date.date() if hasattr(date, 'date') else date,
                        lam2, corr_std, density)

    logger.info("Built %d enriched snapshots (15 features/node)", len(snapshots))
    return snapshots


# ── Main experiment ───────────────────────────────────────────────────────────

def main():
    import torch
    torch.manual_seed(SEED)

    # --- Load universe ---
    from scr_financial.config.loader import load_universe
    universe = load_universe("eu_50")
    ids = universe.ids
    tickers = universe.tickers
    logger.info("Universe: %s (%d banks)", universe.name, len(ids))

    # --- Step 1: Fetch prices ---
    prices = fetch_prices_with_retry(tickers, period="4y", max_attempts=3)
    # Map ticker columns back to bank IDs
    rev = {b.ticker: b.id for b in universe.banks}
    prices.columns = [rev.get(str(c), str(c)) for c in prices.columns]
    # Keep only columns that are bank IDs
    valid_cols = [c for c in prices.columns if c in ids]
    prices = prices[valid_cols]
    # Drop banks with too many NaNs (>30%)
    drop_thresh = 0.3
    nan_frac = prices.isna().mean()
    keep_cols = nan_frac[nan_frac < drop_thresh].index.tolist()
    dropped = [c for c in valid_cols if c not in keep_cols]
    if dropped:
        logger.info("Dropping %d banks with >%.0f%% NaN: %s", len(dropped), drop_thresh*100, dropped)
    prices = prices[keep_cols].dropna()
    ids = keep_cols
    logger.info("Final: %d banks, %d trading days", len(ids), len(prices))

    # --- Steps 2-3: Build enriched snapshots ---
    snapshots = build_enriched_snapshots(
        prices, ids, corr_window=60, min_corr=0.6, stride=3,
    )
    logger.info("Total snapshots: %d", len(snapshots))
    if len(snapshots) < 50:
        raise RuntimeError(f"Too few snapshots ({len(snapshots)}). Need >= 50.")

    # Verify feature shape
    assert snapshots[0]["node_features"].shape[1] == 15, \
        f"Expected 15 features, got {snapshots[0]['node_features'].shape[1]}"
    logger.info("Node features shape: %s (15 = 5 node + 10 graph-level broadcast)",
                snapshots[0]["node_features"].shape)

    # --- Step 5: Override NODE_FEATURE_NAMES ---
    import scr_financial.ml.gnn_predictor as gnn_mod
    gnn_mod.NODE_FEATURE_NAMES = [
        # 5 node-level
        "volatility", "return", "log_price", "beta_proxy", "momentum",
        # 10 graph-level (broadcast)
        "corr_std", "corr_mean", "density", "avg_clustering", "n_components",
        "avg_degree", "edge_change_abs", "lam2_change", "avg_vol", "vol_dispersion",
    ]
    logger.info("NODE_FEATURE_NAMES overridden to %d features", len(gnn_mod.NODE_FEATURE_NAMES))

    # --- Step 6: GNN config (tiny) ---
    gnn_kwargs = {
        "seq_len": 10,
        "hidden_dim": 8,
        "num_gat_layers": 1,
        "num_lstm_layers": 1,
        "heads": 2,
        "dropout": 0.3,
    }
    logger.info("GNN config: %s", gnn_kwargs)

    # --- Step 7: Walk-forward CV ---
    from scr_financial.ml.walk_forward import walk_forward_evaluate
    from scr_financial.ml.gnn_predictor import GNNPredictor

    logger.info("=== Walk-forward CV (4 folds) ===")
    t0 = time.time()
    wf_results = walk_forward_evaluate(
        predictor_class=GNNPredictor,
        predictor_kwargs=gnn_kwargs,
        snapshots=snapshots,
        n_splits=4,
        min_train_size=80,
        test_size=30,
        gap_size=15,
        epochs=200,
        lr=3e-3,
        patience=30,
        seed=SEED,
    )
    wf_time = time.time() - t0
    logger.info("Walk-forward completed in %.1fs", wf_time)

    wf_r2 = wf_results["aggregate"]["test_r2_mean"]
    wf_r2_std = wf_results["aggregate"]["test_r2_std"]
    logger.info("Walk-forward R2: %.4f +/- %.4f", wf_r2, wf_r2_std)
    for fold in wf_results["per_fold"]:
        logger.info("  Fold %d: test_r2=%.4f, test_mse=%.6f (train=%d, test=%d, epochs=%d)",
                     fold["fold"], fold["test_r2"], fold["test_mse"],
                     fold["train_size"], fold["test_size"], fold["epochs_trained"])

    # --- Step 8: Simple train/test split ---
    logger.info("=== Simple train/test split ===")
    t0 = time.time()
    predictor = GNNPredictor(**gnn_kwargs)
    predictor.train(snapshots, epochs=200, lr=3e-3, test_fraction=0.2, patience=30)
    split_time = time.time() - t0

    split_train_r2 = predictor.train_metrics.get("r2", 0)
    split_test_r2 = predictor.test_metrics.get("r2", 0)
    split_test_mse = predictor.test_metrics.get("mse", 0)
    split_r2_per = predictor.test_metrics.get("r2_per_target", {})
    logger.info("Simple split: train_r2=%.4f, test_r2=%.4f, test_mse=%.6f (%.1fs)",
                split_train_r2, split_test_r2, split_test_mse, split_time)
    logger.info("  Per-target R2: %s", {k: f"{v:.4f}" for k, v in split_r2_per.items()})

    # --- Step 9: Save results ---
    n_params = predictor.model.count_parameters() if predictor.model else 0

    results = {
        "experiment": "gnn_graph_features",
        "description": (
            "GNN with 10 graph-level features broadcast to all nodes. "
            "15 features/node = 5 node-level + 10 graph-level. "
            "Key insight: corr_std has ~80% feature importance in GradientBoosting."
        ),
        "config": {
            "universe": "eu_50",
            "n_banks": len(ids),
            "threshold": 0.6,
            "corr_window": 60,
            "stride": 3,
            "n_snapshots": len(snapshots),
            "n_node_features": 15,
            "node_feature_names": gnn_mod.NODE_FEATURE_NAMES,
            "gnn": gnn_kwargs,
            "n_parameters": n_params,
        },
        "walk_forward": {
            "n_splits": 4,
            "min_train_size": 80,
            "test_size": 30,
            "gap_size": 15,
            "aggregate": wf_results["aggregate"],
            "per_fold": wf_results["per_fold"],
            "total_time_s": round(wf_time, 1),
        },
        "simple_split": {
            "train_r2": split_train_r2,
            "test_r2": split_test_r2,
            "test_mse": split_test_mse,
            "r2_per_target": split_r2_per,
            "total_time_s": round(split_time, 1),
        },
        "graph_feature_stats": {
            feat: {
                "mean": float(np.mean([s["_graph_features"][feat] for s in snapshots])),
                "std": float(np.std([s["_graph_features"][feat] for s in snapshots])),
                "min": float(np.min([s["_graph_features"][feat] for s in snapshots])),
                "max": float(np.max([s["_graph_features"][feat] for s in snapshots])),
            }
            for feat in ["corr_std", "corr_mean", "density", "avg_clustering",
                         "n_components", "avg_degree", "edge_change_abs",
                         "lam2_change", "avg_vol", "vol_dispersion"]
        },
    }

    out_path = RESULTS_DIR / "exp_gnn_graph_features.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)
    logger.info("Saved results to %s", out_path)

    # --- Step 10: Print summary ---
    print("\n" + "=" * 70)
    print("EXPERIMENT: GNN with Broadcast Graph-Level Features")
    print("=" * 70)
    print(f"Universe: eu_50 ({len(ids)} banks after filtering)")
    print(f"Snapshots: {len(snapshots)} (corr_window=60, threshold=0.6, stride=3)")
    print(f"Node features: 15 (5 node-level + 10 graph-level broadcast)")
    print(f"GNN: hidden={gnn_kwargs['hidden_dim']}, layers={gnn_kwargs['num_gat_layers']}, "
          f"heads={gnn_kwargs['heads']}, params={n_params}")
    print()
    print(f"Walk-forward CV (4 folds):")
    print(f"  R2 = {wf_r2:.4f} +/- {wf_r2_std:.4f}")
    for fold in wf_results["per_fold"]:
        print(f"    Fold {fold['fold']}: R2={fold['test_r2']:.4f}, "
              f"MSE={fold['test_mse']:.6f}, epochs={fold['epochs_trained']}")
    print()
    print(f"Simple train/test split:")
    print(f"  Train R2 = {split_train_r2:.4f}")
    print(f"  Test  R2 = {split_test_r2:.4f}")
    print(f"  Per-target: {', '.join(f'{k}={v:.4f}' for k, v in split_r2_per.items())}")
    print("=" * 70)

    if wf_r2 > 0:
        print("\n>>> POSITIVE walk-forward R2 -- graph features are informative! <<<")
    else:
        print("\n>>> Negative walk-forward R2 -- graph features alone not sufficient. <<<")


if __name__ == "__main__":
    main()
