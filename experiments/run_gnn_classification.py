#!/usr/bin/env python3
"""GNN-as-feature-extractor for binary classification.

Tests whether a GNN trained on spectral targets learns useful REPRESENTATIONS
for downstream classification (high_vol_regime prediction).

Approach:
  1. Fetch eu_50 data, build snapshots (threshold=0.6, stride=3, corr_window=60)
  2. Compute binary labels: high_vol_regime = (avg_vol in next 5 steps > 75th pct)
  3. Train GNNPredictor on spectral targets (lambda_2, gap, rho)
  4. Extract graph-level embeddings from the trained GNN encoder
  5. Use embeddings alone / graph features alone / combined for classification
  6. Walk-forward CV (3 folds), compare AUC

Run with:
    cd /Users/francoispetizon/scr-financial-networks
    /opt/anaconda3/envs/systemic_risk/bin/python experiments/run_gnn_classification.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, '.')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("gnn_classification")
warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_DIR = Path("experiments/results/publication")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 42
HORIZON = 5
SEQ_LEN = 10  # GNN sequence length


def _json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Data fetching (reuse from run_binary_classification) ────────────────────

def fetch_snapshots():
    """Fetch eu_50 snapshots with threshold=0.6, stride=3, corr_window=60."""
    from dashboard.data_api import build_daily_graph_snapshots

    for universe in ["eu_50", "eu_10"]:
        for attempt in range(3):
            try:
                snaps = build_daily_graph_snapshots(
                    lookback_years=3,
                    corr_window=60,
                    min_corr=0.6,
                    stride=3,
                    universe=universe,
                )
                if len(snaps) > 50:
                    logger.info("Fetched %d snapshots from %s (attempt %d)",
                                len(snaps), universe, attempt + 1)
                    return snaps, universe
                logger.warning("Only %d snapshots (attempt %d), retrying...",
                               len(snaps), attempt + 1)
                time.sleep(5)
            except Exception as e:
                logger.warning("Fetch %s failed (attempt %d): %s", universe, attempt + 1, e)
                time.sleep(5)
        logger.warning("Could not fetch from %s after 3 attempts", universe)

    logger.error("Could not fetch data from any universe.")
    return [], "none"


# ── 36 graph-level features (reuse from run_binary_classification) ──────────

def compute_graph_features(snapshots: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """Extract 36 graph-level features from each snapshot."""
    from scr_financial.network.spectral import (
        compute_laplacian, eigendecomposition, find_spectral_gap,
    )
    from scipy.stats import skew, kurtosis

    feature_names = [
        "lam2", "gap", "rho", "scg_risk", "ev_entropy", "spectral_norm_ratio",
        "avg_degree", "std_degree", "max_degree", "min_degree", "degree_skew",
        "density", "n_edges", "clustering_coeff",
        "corr_mean", "corr_std", "corr_skew", "corr_kurt",
        "corr_q25", "corr_q50", "corr_q75", "corr_max",
        "avg_vol", "vol_std", "vol_skew",
        "lam2_change", "rho_change", "density_change", "edge_change_abs", "vol_change",
        "modularity_proxy", "assortativity", "second_largest_component",
        "effective_resistance_approx", "laplacian_energy", "algebraic_connectivity_ratio",
    ]
    assert len(feature_names) == 36

    all_features = []
    prev_lam2, prev_rho, prev_density, prev_n_edges, prev_vol = None, None, None, None, None

    for i, snap in enumerate(snapshots):
        edge_index = snap["edge_index"]
        n = snap["node_features"].shape[0]

        adj = np.zeros((n, n))
        if edge_index.shape[1] > 0:
            weights = snap.get("edge_weight", np.ones(edge_index.shape[1]))
            for k in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, k]), int(edge_index[1, k])
                adj[src, dst] = float(weights[k])

        L = compute_laplacian(adj)
        evals, evecs = eigendecomposition(L)
        gap_idx, gap_size = find_spectral_gap(evals, adjacency_matrix=adj)

        lam2 = float(evals[1]) if len(evals) > 1 else 0.0
        rho = float(evals[-1]) if len(evals) > 0 else 0.0
        scg_risk = float(1.0 - lam2 / rho) if rho > 1e-8 else 1.0

        evals_pos = evals[evals > 1e-10]
        if len(evals_pos) > 0:
            p = evals_pos / evals_pos.sum()
            ev_entropy = float(-np.sum(p * np.log(p + 1e-15)))
        else:
            ev_entropy = 0.0
        spectral_norm_ratio = float(lam2 / rho) if rho > 1e-8 else 0.0

        degrees = np.sum(adj > 0, axis=1).astype(float)
        avg_degree = float(np.mean(degrees))
        std_degree = float(np.std(degrees))
        max_degree = float(np.max(degrees)) if n > 0 else 0.0
        min_degree = float(np.min(degrees)) if n > 0 else 0.0
        degree_skew = float(skew(degrees)) if n > 2 else 0.0

        n_edges_val = int(np.count_nonzero(adj) // 2)
        max_edges = n * (n - 1) / 2
        density_val = n_edges_val / max_edges if max_edges > 0 else 0.0

        clustering = 0.0
        if avg_degree > 1:
            try:
                import networkx as nx
                G = nx.from_numpy_array(adj)
                clustering = float(nx.average_clustering(G, weight="weight"))
            except Exception:
                pass

        upper_vals = adj[np.triu_indices(n, k=1)]
        nonzero_corrs = upper_vals[upper_vals > 0]
        if len(nonzero_corrs) > 2:
            corr_mean = float(np.mean(nonzero_corrs))
            corr_std = float(np.std(nonzero_corrs))
            corr_skew_val = float(skew(nonzero_corrs))
            corr_kurt_val = float(kurtosis(nonzero_corrs))
            corr_q25 = float(np.percentile(nonzero_corrs, 25))
            corr_q50 = float(np.percentile(nonzero_corrs, 50))
            corr_q75 = float(np.percentile(nonzero_corrs, 75))
            corr_max_val = float(np.max(nonzero_corrs))
        else:
            corr_mean = corr_std = corr_skew_val = corr_kurt_val = 0.0
            corr_q25 = corr_q50 = corr_q75 = corr_max_val = 0.0

        node_feats = snap["node_features"]
        vols = node_feats[:, 0]
        avg_vol = float(np.mean(vols))
        vol_std_val = float(np.std(vols))
        vol_skew_val = float(skew(vols)) if n > 2 else 0.0

        lam2_change = float(lam2 - prev_lam2) if prev_lam2 is not None else 0.0
        rho_change = float(rho - prev_rho) if prev_rho is not None else 0.0
        density_change = float(density_val - prev_density) if prev_density is not None else 0.0
        edge_change_abs = float(abs(n_edges_val - prev_n_edges)) if prev_n_edges is not None else 0.0
        vol_change = float(avg_vol - prev_vol) if prev_vol is not None else 0.0

        prev_lam2, prev_rho, prev_density = lam2, rho, density_val
        prev_n_edges, prev_vol = n_edges_val, avg_vol

        modularity_proxy = float(gap_size) if gap_size else 0.0

        try:
            import networkx as nx
            G = nx.from_numpy_array(adj)
            assortativity = float(nx.degree_assortativity_coefficient(G, weight="weight"))
        except Exception:
            assortativity = 0.0

        try:
            import networkx as nx
            G = nx.from_numpy_array(adj)
            components = sorted(nx.connected_components(G), key=len, reverse=True)
            second_comp = len(components[1]) / n if len(components) > 1 else 0.0
        except Exception:
            second_comp = 0.0

        try:
            L_pinv = np.linalg.pinv(L)
            eff_resistance = float(np.trace(L_pinv)) / n if n > 0 else 0.0
        except Exception:
            eff_resistance = 0.0

        mean_deg = 2 * n_edges_val / n if n > 0 else 0.0
        laplacian_energy = float(np.sum(np.abs(evals - mean_deg)))
        alg_conn_ratio = spectral_norm_ratio

        feat_vec = [
            lam2, gap_size, rho, scg_risk, ev_entropy, spectral_norm_ratio,
            avg_degree, std_degree, max_degree, min_degree, degree_skew,
            density_val, n_edges_val, clustering,
            corr_mean, corr_std, corr_skew_val, corr_kurt_val,
            corr_q25, corr_q50, corr_q75, corr_max_val,
            avg_vol, vol_std_val, vol_skew_val,
            lam2_change, rho_change, density_change, edge_change_abs, vol_change,
            modularity_proxy, assortativity, second_comp,
            eff_resistance, laplacian_energy, alg_conn_ratio,
        ]
        all_features.append(feat_vec)

    X = np.array(all_features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feature_names


# ── Binary labels ───────────────────────────────────────────────────────────

def compute_high_vol_labels(snapshots: List[Dict], horizon: int = 5) -> np.ndarray:
    """high_vol_regime: avg volatility in next `horizon` steps > 75th pct of training vol."""
    n = len(snapshots)
    vol_series = np.zeros(n)
    for i, snap in enumerate(snapshots):
        vol_series[i] = float(np.mean(snap["node_features"][:, 0]))

    # Use first 80% as "training" for threshold computation (same as binary_classification)
    train_end = int(0.8 * n)
    vol_75th = np.percentile(vol_series[:train_end], 75)

    y = np.zeros(n, dtype=int)
    for i in range(n - horizon):
        future_avg = np.mean(vol_series[i + 1: i + 1 + horizon])
        if future_avg > vol_75th:
            y[i] = 1
    return y, vol_75th


# ── GNN training + embedding extraction ─────────────────────────────────────

def train_gnn_and_extract_embeddings(
    snapshots: List[Dict],
    seq_len: int = 10,
    hidden_dim: int = 8,
    epochs: int = 200,
    lr: float = 3e-3,
    patience: int = 30,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Train GNN on spectral targets, then extract graph embeddings for each snapshot.

    Returns:
        embeddings: array [n_snapshots, hidden_dim] - graph embeddings for each snapshot
        train_info: dict with training metrics
    """
    import torch
    from torch_geometric.data import Data, Batch
    from scr_financial.ml.gnn_predictor import GNNPredictor, TARGET_NAMES

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train GNN on spectral targets
    logger.info("Training GNN on spectral targets (hidden=%d, epochs=%d)...", hidden_dim, epochs)
    predictor = GNNPredictor(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_gat_layers=1,
        num_lstm_layers=1,
        heads=2,
        dropout=0.3,
    )
    final_loss = predictor.train(
        snapshots, epochs=epochs, lr=lr, test_fraction=0.2, patience=patience,
    )

    train_info = {
        "final_loss": float(final_loss),
        "train_r2": predictor.train_metrics.get("r2", 0),
        "test_r2": predictor.test_metrics.get("r2", 0),
        "test_mse": predictor.test_metrics.get("mse", 0),
        "r2_per_target": predictor.test_metrics.get("r2_per_target", {}),
        "n_params": predictor.model.count_parameters() if predictor.model else 0,
        "epochs_trained": predictor.training_history[-1]["epoch"] if predictor.training_history else 0,
    }
    logger.info("GNN trained: test_r2=%.4f, test_mse=%.6f, %d params",
                train_info["test_r2"], train_info["test_mse"], train_info["n_params"])

    # Extract embeddings: run each snapshot through the GNN encoder
    model = predictor.model
    model.eval()

    embeddings = np.zeros((len(snapshots), hidden_dim))

    with torch.no_grad():
        for i, snap in enumerate(snapshots):
            data = predictor._snapshot_to_data(snap)
            # Run through GNN encoder (GAT + global_mean_pool)
            emb = model.gnn(data.x, data.edge_index, data.edge_weight, batch=None)
            embeddings[i] = emb.numpy().flatten()

    logger.info("Extracted embeddings: shape %s", embeddings.shape)
    return embeddings, train_info


# ── Walk-forward classification ─────────────────────────────────────────────

def walkforward_classify(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Walk-forward CV for classification. Returns per-classifier metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone

    usable = len(y) - HORIZON
    X_use = X[:usable]
    y_use = y[:usable]
    n = len(y_use)
    pos_rate = float(np.mean(y_use))

    if pos_rate == 0.0 or pos_rate == 1.0:
        return {"skipped": True, "reason": "no class variation", "n_samples": n, "positive_rate": pos_rate}

    classifiers = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=seed, class_weight="balanced", C=1.0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=seed,
            class_weight="balanced", n_jobs=-1,
        ),
    }

    # Walk-forward splits
    fold_size = n // (n_folds + 1)
    wf_results = {name: {"AUC": [], "F1": [], "precision": [], "recall": []}
                  for name in classifiers}

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = min(train_end + fold_size, n)

        if test_end <= test_start or train_end < 20:
            continue

        X_tr = X_use[:train_end]
        y_tr = y_use[:train_end]
        X_te = X_use[test_start:test_end]
        y_te = y_use[test_start:test_end]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        for name, clf_template in classifiers.items():
            try:
                clf = clone(clf_template)
                clf.fit(X_tr_sc, y_tr)
                y_pred = clf.predict(X_te_sc)
                y_prob = (clf.predict_proba(X_te_sc)[:, 1] if hasattr(clf, "predict_proba")
                          else clf.decision_function(X_te_sc))

                auc = float(roc_auc_score(y_te, y_prob))
                f1 = float(f1_score(y_te, y_pred, zero_division=0))
                prec = float(precision_score(y_te, y_pred, zero_division=0))
                rec = float(recall_score(y_te, y_pred, zero_division=0))

                wf_results[name]["AUC"].append(auc)
                wf_results[name]["F1"].append(f1)
                wf_results[name]["precision"].append(prec)
                wf_results[name]["recall"].append(rec)
            except Exception as e:
                logger.warning("  WF fold %d %s failed: %s", fold, name, e)

    results = {"n_samples": n, "positive_rate": round(pos_rate, 4), "classifiers": {}}

    for name in classifiers:
        aucs = wf_results[name]["AUC"]
        f1s = wf_results[name]["F1"]
        if aucs:
            results["classifiers"][name] = {
                "AUC_mean": round(float(np.mean(aucs)), 4),
                "AUC_std": round(float(np.std(aucs)), 4),
                "F1_mean": round(float(np.mean(f1s)), 4),
                "F1_std": round(float(np.std(f1s)), 4),
                "precision_mean": round(float(np.mean(wf_results[name]["precision"])), 4),
                "recall_mean": round(float(np.mean(wf_results[name]["recall"])), 4),
                "n_folds": len(aucs),
                "per_fold_AUC": [round(a, 4) for a in aucs],
            }
        else:
            results["classifiers"][name] = {"n_folds": 0, "error": "no valid folds"}

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import torch
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    t0 = time.time()

    # ── Step 1: Fetch data ──
    logger.info("=== Step 1: Fetching data (threshold=0.6, stride=3, corr_window=60) ===")
    snapshots, universe_name = fetch_snapshots()
    if not snapshots:
        logger.error("No data. Exiting.")
        return

    n_banks = snapshots[0]["node_features"].shape[0]
    logger.info("Got %d snapshots, %d banks from %s", len(snapshots), n_banks, universe_name)

    # ── Step 2: Compute graph features (36-dim) ──
    logger.info("=== Step 2: Computing 36 graph-level features ===")
    X_graph, feature_names = compute_graph_features(snapshots)
    logger.info("Graph features: %s", X_graph.shape)

    # ── Step 3: Compute binary labels ──
    logger.info("=== Step 3: Computing high_vol_regime labels (horizon=%d) ===", HORIZON)
    y, vol_75th = compute_high_vol_labels(snapshots, horizon=HORIZON)
    pos_rate = float(np.mean(y[:len(y) - HORIZON]))
    logger.info("Labels: %d samples, %.1f%% positive, vol_75th=%.4f",
                len(y), pos_rate * 100, vol_75th)

    # ── Step 4: Train GNN on spectral targets + extract embeddings ──
    logger.info("=== Step 4: Training GNN on spectral targets ===")
    hidden_dim = 8
    embeddings, gnn_train_info = train_gnn_and_extract_embeddings(
        snapshots,
        seq_len=SEQ_LEN,
        hidden_dim=hidden_dim,
        epochs=200,
        lr=3e-3,
        patience=30,
        seed=GLOBAL_SEED,
    )

    # ── Step 5: Build feature matrices for 3 comparisons ──
    logger.info("=== Step 5: Walk-forward classification (3 folds) ===")

    # A) GNN embeddings alone (8-dim)
    logger.info("--- (A) GNN embeddings alone (%d-dim) ---", embeddings.shape[1])
    res_emb = walkforward_classify(embeddings, y, n_folds=3, seed=GLOBAL_SEED)

    # B) Graph features alone (36-dim)
    logger.info("--- (B) Graph features alone (%d-dim) ---", X_graph.shape[1])
    res_graph = walkforward_classify(X_graph, y, n_folds=3, seed=GLOBAL_SEED)

    # C) Combined: embeddings + graph features (44-dim)
    X_combined = np.concatenate([embeddings, X_graph], axis=1)
    logger.info("--- (C) Combined: embeddings + graph features (%d-dim) ---", X_combined.shape[1])
    res_combined = walkforward_classify(X_combined, y, n_folds=3, seed=GLOBAL_SEED)

    elapsed = time.time() - t0

    # ── Step 6: Compile and save results ──
    logger.info("=== Step 6: Saving results ===")

    # Extract best AUC from each approach
    def best_auc(res):
        best = 0.0
        best_clf = ""
        for name, m in res.get("classifiers", {}).items():
            if isinstance(m, dict) and "AUC_mean" in m and m["AUC_mean"] > best:
                best = m["AUC_mean"]
                best_clf = name
        return best, best_clf

    auc_emb, clf_emb = best_auc(res_emb)
    auc_graph, clf_graph = best_auc(res_graph)
    auc_combined, clf_combined = best_auc(res_combined)

    results = {
        "experiment": "gnn_classification",
        "description": (
            "GNN as feature extractor for binary classification. "
            "Train GNN on spectral targets (lambda_2, gap, rho) to learn graph representations, "
            "then use graph-level embeddings for downstream high_vol_regime classification. "
            "Compares: GNN embeddings alone vs 36 graph features alone vs combined."
        ),
        "config": {
            "universe": universe_name,
            "n_banks": int(n_banks),
            "n_snapshots": len(snapshots),
            "threshold": 0.6,
            "corr_window": 60,
            "stride": 3,
            "horizon": HORIZON,
            "seq_len": SEQ_LEN,
            "gnn_hidden_dim": hidden_dim,
            "gnn_config": {
                "num_gat_layers": 1,
                "num_lstm_layers": 1,
                "heads": 2,
                "dropout": 0.3,
            },
            "walk_forward_folds": 3,
            "vol_75th_threshold": round(vol_75th, 6),
        },
        "gnn_spectral_training": gnn_train_info,
        "classification_results": {
            "gnn_embeddings_only": {
                "feature_dim": int(embeddings.shape[1]),
                "description": f"GNN encoder embeddings ({hidden_dim}-dim)",
                **res_emb,
            },
            "graph_features_only": {
                "feature_dim": int(X_graph.shape[1]),
                "description": "36 hand-crafted graph features",
                **res_graph,
            },
            "combined": {
                "feature_dim": int(X_combined.shape[1]),
                "description": f"GNN embeddings ({hidden_dim}) + graph features (36) = {X_combined.shape[1]}-dim",
                **res_combined,
            },
        },
        "comparison_summary": {
            "gnn_embeddings_best_AUC": round(auc_emb, 4),
            "gnn_embeddings_best_clf": clf_emb,
            "graph_features_best_AUC": round(auc_graph, 4),
            "graph_features_best_clf": clf_graph,
            "combined_best_AUC": round(auc_combined, 4),
            "combined_best_clf": clf_combined,
            "gnn_adds_value": auc_combined > auc_graph + 0.005,
            "gnn_representations_useful": auc_emb > 0.6,
            "previous_baseline_RF_AUC": 0.9831,
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    out_path = RESULTS_DIR / "exp_gnn_classification.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)
    logger.info("Saved to %s", out_path)

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("GNN-AS-FEATURE-EXTRACTOR CLASSIFICATION RESULTS")
    print("=" * 70)
    print(f"Universe: {universe_name} ({n_banks} banks), {len(snapshots)} snapshots")
    print(f"Label: high_vol_regime (positive rate: {pos_rate:.1%})")
    print(f"GNN spectral training: test_r2={gnn_train_info['test_r2']:.4f}, "
          f"params={gnn_train_info['n_params']}")
    print()

    for approach_name, approach_res in results["classification_results"].items():
        print(f"  {approach_name} ({approach_res['feature_dim']}-dim):")
        for clf_name, metrics in approach_res.get("classifiers", {}).items():
            if isinstance(metrics, dict) and "AUC_mean" in metrics:
                print(f"    {clf_name}: AUC={metrics['AUC_mean']:.4f}+/-{metrics['AUC_std']:.4f}, "
                      f"F1={metrics['F1_mean']:.4f}, P={metrics['precision_mean']:.4f}, "
                      f"R={metrics['recall_mean']:.4f}  (folds: {metrics.get('per_fold_AUC', [])})")
        print()

    print("COMPARISON:")
    print(f"  GNN embeddings alone:  AUC = {auc_emb:.4f} ({clf_emb})")
    print(f"  Graph features alone:  AUC = {auc_graph:.4f} ({clf_graph})")
    print(f"  Combined:              AUC = {auc_combined:.4f} ({clf_combined})")
    print(f"  Previous baseline (RF on graph feats): AUC = 0.9831")
    print()

    if auc_combined > auc_graph + 0.005:
        print(">>> YES: GNN embeddings ADD value beyond hand-crafted graph features.")
    elif auc_emb > 0.6:
        print(">>> PARTIAL: GNN embeddings are informative but don't improve over graph features.")
    else:
        print(">>> NO: GNN embeddings are not useful for this classification task.")

    print(f"\nElapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
