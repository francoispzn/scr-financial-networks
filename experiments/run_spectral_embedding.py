#!/usr/bin/env python3
"""
Experiment: Spectral Embedding Features for GNN Predictor.

Gives each node its spectral embedding coordinates (eigenvector components)
as features, directly encoding each bank's position in the spectral structure.

Steps:
  1. Fetch eu_50 data (with retry), build adjacency at threshold=0.6, stride=3
  2. For each snapshot, compute Laplacian eigenvectors and add:
     - First 3 non-trivial eigenvector components per node
     - Node contribution to lambda_2: |u_1(i)|^2
     - Spectral distance to centroid
  3. Concatenate with original 5 features -> 10 features per node
  4. Test seq_len = 5, 10, 20, 30
  5. Train GNNPredictor (hidden=8, layers=1, heads=2, dropout=0.3) for each
  6. Walk-forward CV with best seq_len
  7. Save to experiments/results/publication/exp_spectral_embedding.json
"""

import sys
sys.path.insert(0, '.')

import json
import logging
import time
import copy
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("spectral_embedding")

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


# ── Step 1: Fetch snapshots ───────────────────────────────────────────

def fetch_snapshots():
    """Fetch eu_50 snapshots with retry, threshold=0.6, stride=3.

    Uses a custom build that drops banks with missing price data so that
    delisted / failing tickers don't cause all rows to be dropped.
    """
    from scr_financial.config.loader import load_universe
    from scr_financial.network.spectral import (
        compute_laplacian, eigendecomposition, find_spectral_gap,
        analyze_spectral_properties,
    )
    from dashboard.data_api import _fetch_prices, BANK_TICKERS

    # Load eu_50 universe and patch tickers
    univ = load_universe("eu_50")
    ids = univ.ids
    for b in univ.banks:
        if b.id not in BANK_TICKERS:
            BANK_TICKERS[b.id] = b.ticker

    corr_window = 60
    min_corr = 0.6
    stride = 3
    lookback_years = 3

    for attempt in range(3):
        try:
            prices = _fetch_prices(ids, period=f"{lookback_years * 365 + 60}d")
            if prices.empty:
                logger.warning("Empty prices (attempt %d)", attempt + 1)
                time.sleep(5)
                continue

            # Drop columns (banks) with >20% missing data
            thresh = int(len(prices) * 0.8)
            prices = prices.dropna(axis=1, thresh=thresh)
            # Forward-fill remaining gaps, then drop leading NaNs
            prices = prices.ffill().dropna()
            valid_ids = [c for c in prices.columns if c in ids]
            prices = prices[valid_ids]
            n_banks = len(valid_ids)

            if n_banks < 5:
                logger.warning("Only %d banks with data (attempt %d)", n_banks, attempt + 1)
                time.sleep(5)
                continue

            returns = prices.pct_change().iloc[1:]  # drop first NaN row
            logger.info("Building snapshots: %d trading days, %d banks (of %d requested)",
                        len(returns), n_banks, len(ids))

            if len(returns) < corr_window + 30:
                logger.warning("Insufficient history: %d rows (attempt %d)", len(returns), attempt + 1)
                time.sleep(5)
                continue

            snapshots = []
            valid_dates = returns.index[corr_window:]
            for count, date_idx in enumerate(range(0, len(valid_dates), stride)):
                date = valid_dates[date_idx]
                window_end = corr_window + date_idx
                ret_window = returns.iloc[window_end - corr_window: window_end]

                corr = ret_window.corr()
                n = n_banks
                adj = np.zeros((n, n), dtype=np.float32)
                for i, src in enumerate(valid_ids):
                    for j, tgt in enumerate(valid_ids):
                        if i >= j:
                            continue
                        w = float(corr.loc[src, tgt]) if (src in corr.index and tgt in corr.index) else 0.0
                        if np.isnan(w):
                            w = 0.0
                        if w >= min_corr:
                            adj[i, j] = w
                            adj[j, i] = w

                # Node features: [N, 5]
                node_feats = np.zeros((n, 5), dtype=np.float32)
                for i, bid in enumerate(valid_ids):
                    if bid in ret_window.columns:
                        rets_i = ret_window[bid].values
                        node_feats[i, 0] = float(np.nanstd(rets_i) * np.sqrt(252))
                        node_feats[i, 1] = float(np.nanmean(rets_i) * 252)
                        cum_ret = np.nanprod(1 + rets_i)
                        node_feats[i, 2] = float(np.log(max(cum_ret, 0.01)))
                        mkt = ret_window.mean(axis=1).values
                        cov = np.cov(rets_i, mkt)[0, 1] if len(rets_i) > 2 else 0
                        var_mkt = np.var(mkt) if np.var(mkt) > 1e-10 else 1.0
                        node_feats[i, 3] = float(cov / var_mkt)
                        if len(rets_i) >= 20:
                            node_feats[i, 4] = float(np.sum(rets_i[-20:]))

                rows, cols = np.nonzero(adj)
                edge_index = np.stack([rows, cols], axis=0).astype(np.int64) if len(rows) > 0 \
                    else np.zeros((2, 0), dtype=np.int64)
                edge_weight = adj[rows, cols] if len(rows) > 0 else np.zeros(0, dtype=np.float32)

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

            logger.info("Built %d snapshots (attempt %d)", len(snapshots), attempt + 1)
            if len(snapshots) > 50:
                return snapshots
            logger.warning("Only %d snapshots, retrying...", len(snapshots))
            time.sleep(5)
        except Exception as e:
            logger.warning("Fetch failed (attempt %d): %s", attempt + 1, e)
            import traceback
            traceback.print_exc()
            time.sleep(5)
    logger.error("Could not fetch sufficient data after 3 attempts.")
    return []


# ── Step 2: Add spectral embedding features ──────────────────────────

def add_spectral_embedding_features(snapshots):
    """
    For each snapshot, compute spectral embedding features and concatenate.

    New features per node (5 additional):
      - eigvec_1(i), eigvec_2(i), eigvec_3(i): first 3 non-trivial eigenvector components
      - |u_1(i)|^2: node's contribution to lambda_2 (Fiedler vector)
      - spectral_distance_to_centroid: Euclidean distance in spectral space to centroid
    """
    from scr_financial.network.spectral import compute_laplacian, eigendecomposition

    augmented = []
    n_original_feats = snapshots[0]["node_features"].shape[1]  # should be 5
    logger.info("Original feature dim: %d", n_original_feats)

    for idx, snap in enumerate(snapshots):
        n_nodes = snap["node_features"].shape[0]

        # Rebuild adjacency from edge_index and edge_weight
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        ei = snap["edge_index"]
        ew = snap["edge_weight"]
        if ei.shape[1] > 0:
            for k in range(ei.shape[1]):
                adj[ei[0, k], ei[1, k]] = ew[k]

        # Compute Laplacian and eigenvectors
        adj_sym = (adj + adj.T) / 2.0
        L = compute_laplacian(adj_sym, normalized=True)
        eigenvalues, eigenvectors = eigendecomposition(L)

        # Non-trivial eigenvectors: skip first (constant, eigenvalue ~ 0)
        # Take eigenvectors 1, 2, 3 (0-indexed)
        spectral_feats = np.zeros((n_nodes, 5), dtype=np.float32)

        n_evecs = eigenvectors.shape[1]
        for k in range(min(3, n_evecs - 1)):
            evec = eigenvectors[:, k + 1]  # skip trivial eigenvector 0
            spectral_feats[:, k] = evec.astype(np.float32)

        # Node contribution to lambda_2: |u_1(i)|^2
        if n_evecs > 1:
            fiedler = eigenvectors[:, 1]
            spectral_feats[:, 3] = (fiedler ** 2).astype(np.float32)

        # Spectral distance to centroid (using first 3 non-trivial eigenvectors)
        if n_evecs > 3:
            coords = eigenvectors[:, 1:4]  # [n_nodes, 3]
            centroid = coords.mean(axis=0)
            dists = np.sqrt(np.sum((coords - centroid) ** 2, axis=1))
            spectral_feats[:, 4] = dists.astype(np.float32)
        elif n_evecs > 1:
            coords = eigenvectors[:, 1:min(4, n_evecs)]
            centroid = coords.mean(axis=0)
            dists = np.sqrt(np.sum((coords - centroid) ** 2, axis=1))
            spectral_feats[:, 4] = dists.astype(np.float32)

        # Concatenate: original 5 + spectral 5 = 10
        new_feats = np.concatenate([snap["node_features"], spectral_feats], axis=1)

        new_snap = copy.copy(snap)
        new_snap["node_features"] = new_feats
        augmented.append(new_snap)

        if (idx + 1) % 50 == 0:
            logger.info("Augmented %d/%d snapshots with spectral features", idx + 1, len(snapshots))

    logger.info("All %d snapshots augmented: %d -> %d features per node",
                len(augmented), n_original_feats, augmented[0]["node_features"].shape[1])
    return augmented


# ── Step 3-6: Train with different seq_len ────────────────────────────

def train_with_seq_len(snapshots, seq_len, epochs=200, patience=30):
    """Train GNNPredictor with given seq_len, return metrics."""
    import torch
    from scr_financial.ml.gnn_predictor import GNNPredictor

    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    n_feats = snapshots[0]["node_features"].shape[1]

    predictor = GNNPredictor(
        seq_len=seq_len,
        hidden_dim=8,
        num_gat_layers=1,
        num_lstm_layers=1,
        heads=2,
        dropout=0.3,
    )

    # Patch NODE_FEATURE_NAMES to match the augmented feature count
    import scr_financial.ml.gnn_predictor as gnn_mod
    orig_names = gnn_mod.NODE_FEATURE_NAMES
    gnn_mod.NODE_FEATURE_NAMES = [f"feat_{i}" for i in range(n_feats)]

    t0 = time.time()
    try:
        predictor.train(
            snapshots,
            epochs=epochs,
            lr=3e-3,
            test_fraction=0.2,
            patience=patience,
        )
        train_time = time.time() - t0

        result = {
            "seq_len": seq_len,
            "n_features": n_feats,
            "train_r2": predictor.train_metrics.get("r2", 0),
            "test_r2": predictor.test_metrics.get("r2", 0),
            "test_mse": predictor.test_metrics.get("mse", 0),
            "test_r2_per_target": predictor.test_metrics.get("r2_per_target", {}),
            "train_mse": predictor.train_metrics.get("mse", 0),
            "training_time_s": round(train_time, 1),
            "n_snapshots": len(snapshots),
            "n_sequences": len(snapshots) - seq_len,
            "epochs_trained": predictor.training_history[-1]["epoch"] if predictor.training_history else 0,
            "n_params": predictor.model.count_parameters() if predictor.model else 0,
        }
    except Exception as e:
        logger.warning("Training failed for seq_len=%d: %s", seq_len, e)
        train_time = time.time() - t0
        result = {
            "seq_len": seq_len,
            "error": str(e),
            "training_time_s": round(train_time, 1),
        }
    finally:
        gnn_mod.NODE_FEATURE_NAMES = orig_names

    return result


def train_baseline(snapshots_original, seq_len, epochs=200, patience=30):
    """Train with original 5 features only (no spectral embedding) for comparison."""
    import torch
    from scr_financial.ml.gnn_predictor import GNNPredictor

    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    predictor = GNNPredictor(
        seq_len=seq_len,
        hidden_dim=8,
        num_gat_layers=1,
        num_lstm_layers=1,
        heads=2,
        dropout=0.3,
    )

    t0 = time.time()
    try:
        predictor.train(
            snapshots_original,
            epochs=epochs,
            lr=3e-3,
            test_fraction=0.2,
            patience=patience,
        )
        train_time = time.time() - t0

        result = {
            "seq_len": seq_len,
            "n_features": 5,
            "train_r2": predictor.train_metrics.get("r2", 0),
            "test_r2": predictor.test_metrics.get("r2", 0),
            "test_mse": predictor.test_metrics.get("mse", 0),
            "test_r2_per_target": predictor.test_metrics.get("r2_per_target", {}),
            "train_mse": predictor.train_metrics.get("mse", 0),
            "training_time_s": round(train_time, 1),
            "epochs_trained": predictor.training_history[-1]["epoch"] if predictor.training_history else 0,
            "n_params": predictor.model.count_parameters() if predictor.model else 0,
        }
    except Exception as e:
        logger.warning("Baseline training failed: %s", e)
        result = {"seq_len": seq_len, "error": str(e)}

    return result


# ── Step 7: Walk-forward CV ───────────────────────────────────────────

def run_walk_forward(snapshots, seq_len):
    """Run walk-forward CV with best seq_len."""
    import torch
    from scr_financial.ml.gnn_predictor import GNNPredictor
    from scr_financial.ml.walk_forward import walk_forward_evaluate
    import scr_financial.ml.gnn_predictor as gnn_mod

    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    n_feats = snapshots[0]["node_features"].shape[1]
    orig_names = gnn_mod.NODE_FEATURE_NAMES
    gnn_mod.NODE_FEATURE_NAMES = [f"feat_{i}" for i in range(n_feats)]

    n_total = len(snapshots)
    # Adjust CV params based on available data
    min_train = max(seq_len + 15, 60)
    test_size = max(20, n_total // 10)
    gap_size = 5

    try:
        cv_result = walk_forward_evaluate(
            predictor_class=GNNPredictor,
            predictor_kwargs={
                "seq_len": seq_len,
                "hidden_dim": 8,
                "num_gat_layers": 1,
                "num_lstm_layers": 1,
                "heads": 2,
                "dropout": 0.3,
            },
            snapshots=snapshots,
            n_splits=3,
            min_train_size=min_train,
            test_size=test_size,
            gap_size=gap_size,
            epochs=200,
            lr=3e-3,
            patience=30,
            seed=GLOBAL_SEED,
        )
    except Exception as e:
        logger.warning("Walk-forward CV failed: %s", e)
        cv_result = {"error": str(e)}
    finally:
        gnn_mod.NODE_FEATURE_NAMES = orig_names

    return cv_result


# ── Main ──────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # Step 1: Fetch
    logger.info("=== Step 1: Fetching eu_50 data (threshold=0.6, stride=3) ===")
    snapshots_original = fetch_snapshots()
    if not snapshots_original:
        logger.error("No data fetched. Aborting.")
        return
    logger.info("Got %d snapshots, %d nodes, %d original features",
                len(snapshots_original),
                snapshots_original[0]["node_features"].shape[0],
                snapshots_original[0]["node_features"].shape[1])

    # Step 2: Add spectral embedding features
    logger.info("=== Step 2: Computing spectral embedding features ===")
    snapshots_augmented = add_spectral_embedding_features(snapshots_original)
    logger.info("Augmented feature dim: %d", snapshots_augmented[0]["node_features"].shape[1])

    # Step 4-5: Test different seq_len values
    seq_lens = [5, 10, 20, 30]
    results_by_seq_len = {}

    for sl in seq_lens:
        if len(snapshots_augmented) < sl + 10:
            logger.warning("Not enough snapshots (%d) for seq_len=%d, skipping",
                          len(snapshots_augmented), sl)
            continue

        logger.info("=== Training with spectral embedding, seq_len=%d ===", sl)
        result = train_with_seq_len(snapshots_augmented, sl)
        results_by_seq_len[str(sl)] = result
        if "error" not in result:
            logger.info("  seq_len=%d: test_r2=%.4f, test_mse=%.6f, time=%.1fs, epochs=%d",
                       sl, result["test_r2"], result["test_mse"],
                       result["training_time_s"], result.get("epochs_trained", 0))
        else:
            logger.info("  seq_len=%d: FAILED - %s", sl, result["error"])

    # Find best seq_len
    best_sl = None
    best_r2 = -999
    for sl_str, res in results_by_seq_len.items():
        if "error" not in res and res["test_r2"] > best_r2:
            best_r2 = res["test_r2"]
            best_sl = int(sl_str)

    logger.info("=== Best seq_len: %s (test_r2=%.4f) ===", best_sl, best_r2)

    # Step 6: Baseline comparison (original 5 features, best seq_len)
    baseline_result = None
    if best_sl is not None:
        logger.info("=== Baseline (5 features, seq_len=%d) ===", best_sl)
        baseline_result = train_baseline(snapshots_original, best_sl)
        if "error" not in baseline_result:
            logger.info("  Baseline: test_r2=%.4f, test_mse=%.6f",
                       baseline_result["test_r2"], baseline_result["test_mse"])

    # Step 7: Walk-forward CV with best seq_len
    wf_result = None
    if best_sl is not None:
        logger.info("=== Walk-forward CV with best seq_len=%d ===", best_sl)
        wf_result = run_walk_forward(snapshots_augmented, best_sl)
        if "error" not in wf_result:
            agg = wf_result.get("aggregate", {})
            logger.info("  WF-CV: mean_r2=%.4f +/- %.4f, mean_mse=%.6f",
                       agg.get("test_r2_mean", 0),
                       agg.get("test_r2_std", 0),
                       agg.get("test_mse_mean", 0))

    # Also run walk-forward on baseline for comparison
    wf_baseline = None
    if best_sl is not None:
        logger.info("=== Walk-forward CV baseline (5 features, seq_len=%d) ===", best_sl)
        import torch
        from scr_financial.ml.gnn_predictor import GNNPredictor
        from scr_financial.ml.walk_forward import walk_forward_evaluate

        np.random.seed(GLOBAL_SEED)
        torch.manual_seed(GLOBAL_SEED)

        n_total = len(snapshots_original)
        min_train = max(best_sl + 15, 60)
        test_size = max(20, n_total // 10)

        try:
            wf_baseline = walk_forward_evaluate(
                predictor_class=GNNPredictor,
                predictor_kwargs={
                    "seq_len": best_sl,
                    "hidden_dim": 8,
                    "num_gat_layers": 1,
                    "num_lstm_layers": 1,
                    "heads": 2,
                    "dropout": 0.3,
                },
                snapshots=snapshots_original,
                n_splits=3,
                min_train_size=min_train,
                test_size=test_size,
                gap_size=5,
                epochs=200,
                lr=3e-3,
                patience=30,
                seed=GLOBAL_SEED,
            )
        except Exception as e:
            logger.warning("Baseline walk-forward failed: %s", e)
            wf_baseline = {"error": str(e)}

    # ── Compile results ───────────────────────────────────────────────
    total_time = time.time() - t_start

    output = {
        "experiment": "spectral_embedding_features",
        "description": (
            "Spectral embedding coordinates (eigenvector components) as node features. "
            "For each snapshot: first 3 non-trivial eigenvector components, "
            "node contribution to lambda_2, spectral distance to centroid. "
            "Original 5 features + 5 spectral = 10 features per node."
        ),
        "data_config": {
            "universe": "eu_50",
            "threshold": 0.6,
            "stride": 3,
            "corr_window": 60,
            "lookback_years": 3,
            "n_snapshots": len(snapshots_original),
            "n_nodes": int(snapshots_original[0]["node_features"].shape[0]),
        },
        "model_config": {
            "hidden_dim": 8,
            "num_gat_layers": 1,
            "num_lstm_layers": 1,
            "heads": 2,
            "dropout": 0.3,
            "epochs": 200,
            "patience": 30,
            "lr": 3e-3,
        },
        "spectral_features": [
            "eigvec_1(i): Fiedler vector component",
            "eigvec_2(i): 2nd non-trivial eigenvector component",
            "eigvec_3(i): 3rd non-trivial eigenvector component",
            "|u_1(i)|^2: node contribution to algebraic connectivity",
            "spectral_distance_to_centroid: Euclidean distance in spectral space",
        ],
        "seq_len_comparison": results_by_seq_len,
        "best_seq_len": best_sl,
        "best_test_r2": best_r2,
        "baseline_5_features": baseline_result,
        "walk_forward_cv_spectral": wf_result,
        "walk_forward_cv_baseline": wf_baseline,
        "total_time_s": round(total_time, 1),
    }

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SPECTRAL EMBEDDING EXPERIMENT RESULTS")
    print("=" * 70)

    print(f"\nData: {output['data_config']['n_snapshots']} snapshots, "
          f"{output['data_config']['n_nodes']} nodes")

    print("\n--- Sequence Length Comparison (10 spectral features) ---")
    print(f"{'seq_len':>8}  {'test_r2':>10}  {'test_mse':>10}  {'time(s)':>8}  {'epochs':>6}")
    for sl_str, res in sorted(results_by_seq_len.items(), key=lambda x: int(x[0])):
        if "error" not in res:
            print(f"{res['seq_len']:>8}  {res['test_r2']:>10.4f}  {res['test_mse']:>10.6f}  "
                  f"{res['training_time_s']:>8.1f}  {res.get('epochs_trained', '?'):>6}")
        else:
            print(f"{sl_str:>8}  {'ERROR':>10}  {res['error'][:40]}")

    print(f"\nBest seq_len: {best_sl} (test_r2 = {best_r2:.4f})")

    if baseline_result and "error" not in baseline_result:
        print(f"\n--- Baseline (5 original features, seq_len={best_sl}) ---")
        print(f"  test_r2 = {baseline_result['test_r2']:.4f}")
        print(f"  test_mse = {baseline_result['test_mse']:.6f}")
        delta = best_r2 - baseline_result['test_r2']
        direction = "BETTER" if delta > 0 else "WORSE"
        print(f"\n  Spectral embedding vs baseline: {delta:+.4f} R2 ({direction})")

    if wf_result and "error" not in wf_result:
        agg = wf_result.get("aggregate", {})
        print(f"\n--- Walk-Forward CV (spectral, seq_len={best_sl}) ---")
        print(f"  mean_r2 = {agg.get('test_r2_mean', 0):.4f} +/- {agg.get('test_r2_std', 0):.4f}")
        print(f"  mean_mse = {agg.get('test_mse_mean', 0):.6f}")
        print(f"  folds completed: {agg.get('n_folds_completed', 0)}")

    if wf_baseline and "error" not in wf_baseline:
        agg_bl = wf_baseline.get("aggregate", {})
        print(f"\n--- Walk-Forward CV (baseline 5 features, seq_len={best_sl}) ---")
        print(f"  mean_r2 = {agg_bl.get('test_r2_mean', 0):.4f} +/- {agg_bl.get('test_r2_std', 0):.4f}")
        print(f"  mean_mse = {agg_bl.get('test_mse_mean', 0):.6f}")

        if wf_result and "error" not in wf_result:
            agg_sp = wf_result.get("aggregate", {})
            delta_wf = agg_sp.get("test_r2_mean", 0) - agg_bl.get("test_r2_mean", 0)
            direction_wf = "BETTER" if delta_wf > 0 else "WORSE"
            print(f"\n  WF-CV spectral vs baseline: {delta_wf:+.4f} mean R2 ({direction_wf})")

    # Per-target R2 for best spectral
    if best_sl and str(best_sl) in results_by_seq_len:
        res = results_by_seq_len[str(best_sl)]
        if "test_r2_per_target" in res and res["test_r2_per_target"]:
            print(f"\n--- Per-Target R2 (spectral, seq_len={best_sl}) ---")
            for target, r2 in res["test_r2_per_target"].items():
                print(f"  {target}: {r2:.4f}")

    print(f"\nTotal time: {total_time:.1f}s")
    print("=" * 70)

    # ── Save ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "exp_spectral_embedding.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_safe)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
