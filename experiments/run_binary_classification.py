#!/usr/bin/env python3
"""Binary classification experiment: can we predict network stress events?

Instead of regressing lambda_2 values, we predict binary labels:
will network connectivity deteriorate significantly in the next k steps?

Labels:
  - connectivity_drop_1std:  lambda_2 drops by >1 std within next 5 steps
  - connectivity_drop_05std: lambda_2 drops by >0.5 std within next 5 steps
  - density_drop:            network density drops by >10% within next 5 steps
  - high_vol_regime:         avg volatility exceeds 75th percentile within next 5 steps

Run with:
    cd /Users/francoispetizon/scr-financial-networks
    /opt/anaconda3/envs/systemic_risk/bin/python experiments/run_binary_classification.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("binary_classification")
warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_DIR = Path("experiments/results/publication")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SEED = 42
HORIZON = 5  # look-ahead window (steps)


# ── JSON-safe conversion ──────────────────────────────────────────

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


# ── Data fetching with retry ─────────────────────────────────────

def fetch_snapshots():
    """Fetch eu_50 snapshots with threshold=0.6 and stride=3.

    Falls back to eu_10 if eu_50 fails (some tickers may be delisted or
    blocked by TLS errors).
    """
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
                    return snaps
                logger.warning("Only %d snapshots from %s (attempt %d), retrying...",
                               len(snaps), universe, attempt + 1)
                time.sleep(5)
            except Exception as e:
                logger.warning("Fetch %s failed (attempt %d): %s", universe, attempt + 1, e)
                time.sleep(5)
        logger.warning("Could not fetch from %s after 3 attempts, trying next universe...", universe)

    logger.error("Could not fetch sufficient data from any universe.")
    return []


# ── 36 graph-level features ──────────────────────────────────────

def compute_graph_features(snapshots: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """Extract 36 graph-level features from each snapshot.

    Features per snapshot:
      Spectral (6): lam2, gap, rho, scg_risk, ev_entropy, spectral_norm_ratio
      Degree (5):   avg_degree, std_degree, max_degree, min_degree, degree_skew
      Density (3):  density, n_edges, clustering_coeff
      Correlation (8): corr_mean, corr_std, corr_skew, corr_kurt, corr_q25, corr_q50, corr_q75, corr_max
      Volatility (3): avg_vol, vol_std, vol_skew
      Change/momentum (5): lam2_change, rho_change, density_change, edge_change_abs, vol_change
      Higher-order (6): modularity_proxy, assortativity, second_largest_component,
                         effective_resistance_approx, laplacian_energy, algebraic_connectivity_ratio
    """
    from scr_financial.network.spectral import (
        compute_laplacian, eigendecomposition, find_spectral_gap,
    )
    from scipy.stats import skew, kurtosis

    feature_names = [
        # Spectral (6)
        "lam2", "gap", "rho", "scg_risk", "ev_entropy", "spectral_norm_ratio",
        # Degree (5)
        "avg_degree", "std_degree", "max_degree", "min_degree", "degree_skew",
        # Density (3)
        "density", "n_edges", "clustering_coeff",
        # Correlation (8)
        "corr_mean", "corr_std", "corr_skew", "corr_kurt",
        "corr_q25", "corr_q50", "corr_q75", "corr_max",
        # Volatility (3)
        "avg_vol", "vol_std", "vol_skew",
        # Change/momentum (5)
        "lam2_change", "rho_change", "density_change", "edge_change_abs", "vol_change",
        # Higher-order (6)
        "modularity_proxy", "assortativity", "second_largest_component",
        "effective_resistance_approx", "laplacian_energy", "algebraic_connectivity_ratio",
    ]

    assert len(feature_names) == 36, f"Expected 36 features, got {len(feature_names)}"

    all_features = []
    prev_lam2, prev_rho, prev_density, prev_n_edges, prev_vol = None, None, None, None, None

    for i, snap in enumerate(snapshots):
        edge_index = snap["edge_index"]
        n = snap["node_features"].shape[0]

        # Build adjacency matrix
        adj = np.zeros((n, n))
        if edge_index.shape[1] > 0:
            weights = snap.get("edge_weight", np.ones(edge_index.shape[1]))
            for k in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, k]), int(edge_index[1, k])
                adj[src, dst] = float(weights[k])

        # Spectral decomposition
        L = compute_laplacian(adj)
        evals, evecs = eigendecomposition(L)
        gap_idx, gap_size = find_spectral_gap(evals, adjacency_matrix=adj)

        lam2 = float(evals[1]) if len(evals) > 1 else 0.0
        rho = float(evals[-1]) if len(evals) > 0 else 0.0
        scg_risk = float(1.0 - lam2 / rho) if rho > 1e-8 else 1.0

        # Eigenvalue entropy
        evals_pos = evals[evals > 1e-10]
        if len(evals_pos) > 0:
            p = evals_pos / evals_pos.sum()
            ev_entropy = float(-np.sum(p * np.log(p + 1e-15)))
        else:
            ev_entropy = 0.0

        spectral_norm_ratio = float(lam2 / rho) if rho > 1e-8 else 0.0

        # Degree statistics
        degrees = np.sum(adj > 0, axis=1).astype(float)
        avg_degree = float(np.mean(degrees))
        std_degree = float(np.std(degrees))
        max_degree = float(np.max(degrees)) if n > 0 else 0.0
        min_degree = float(np.min(degrees)) if n > 0 else 0.0
        degree_skew = float(skew(degrees)) if n > 2 else 0.0

        # Density
        n_edges_val = int(np.count_nonzero(adj) // 2)
        max_edges = n * (n - 1) / 2
        density_val = n_edges_val / max_edges if max_edges > 0 else 0.0

        # Clustering coefficient (average local)
        clustering = 0.0
        if avg_degree > 1:
            try:
                import networkx as nx
                G = nx.from_numpy_array(adj)
                clustering = float(nx.average_clustering(G, weight="weight"))
            except Exception:
                clustering = 0.0

        # Correlation statistics (from upper triangle of adj as proxy for correlations)
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

        # Volatility features (node_features[:,0] = volatility_30d)
        node_feats = snap["node_features"]
        vols = node_feats[:, 0]
        avg_vol = float(np.mean(vols))
        vol_std_val = float(np.std(vols))
        vol_skew_val = float(skew(vols)) if n > 2 else 0.0

        # Change features (vs previous snapshot)
        lam2_change = float(lam2 - prev_lam2) if prev_lam2 is not None else 0.0
        rho_change = float(rho - prev_rho) if prev_rho is not None else 0.0
        density_change = float(density_val - prev_density) if prev_density is not None else 0.0
        edge_change_abs = float(abs(n_edges_val - prev_n_edges)) if prev_n_edges is not None else 0.0
        vol_change = float(avg_vol - prev_vol) if prev_vol is not None else 0.0

        prev_lam2, prev_rho, prev_density = lam2, rho, density_val
        prev_n_edges, prev_vol = n_edges_val, avg_vol

        # Higher-order features
        # Modularity proxy: ratio of within-cluster to between-cluster edges
        # Use spectral gap as modularity proxy
        modularity_proxy = float(gap_size) if gap_size else 0.0

        # Assortativity (degree correlation)
        try:
            import networkx as nx
            G = nx.from_numpy_array(adj)
            assortativity = float(nx.degree_assortativity_coefficient(G, weight="weight"))
        except Exception:
            assortativity = 0.0

        # Second largest connected component size
        try:
            import networkx as nx
            G = nx.from_numpy_array(adj)
            components = sorted(nx.connected_components(G), key=len, reverse=True)
            second_comp = len(components[1]) / n if len(components) > 1 else 0.0
        except Exception:
            second_comp = 0.0

        # Effective resistance approximation (trace of pseudoinverse of L)
        try:
            L_pinv = np.linalg.pinv(L)
            eff_resistance = float(np.trace(L_pinv)) / n if n > 0 else 0.0
        except Exception:
            eff_resistance = 0.0

        # Laplacian energy: sum of |eigenvalues - mean_degree|
        mean_deg = 2 * n_edges_val / n if n > 0 else 0.0
        laplacian_energy = float(np.sum(np.abs(evals - mean_deg)))

        # Algebraic connectivity ratio: lambda_2 / lambda_n
        alg_conn_ratio = spectral_norm_ratio  # same as lam2/rho

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
    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feature_names


# ── Stress label construction ────────────────────────────────────

def compute_stress_labels(snapshots: List[Dict], X: np.ndarray, horizon: int = 5) -> Dict[str, np.ndarray]:
    """Compute binary stress labels for each snapshot.

    Returns dict of label_name -> binary array (1=stress, 0=normal).
    Arrays are same length as snapshots, with trailing entries set to 0 (unknown future).
    """
    from scr_financial.network.spectral import compute_laplacian, eigendecomposition

    n = len(snapshots)

    # Extract time series
    lambda2_series = np.zeros(n)
    density_series = np.zeros(n)
    vol_series = np.zeros(n)

    for i, snap in enumerate(snapshots):
        edge_index = snap["edge_index"]
        n_nodes = snap["node_features"].shape[0]

        # lambda_2
        adj = np.zeros((n_nodes, n_nodes))
        if edge_index.shape[1] > 0:
            weights = snap.get("edge_weight", np.ones(edge_index.shape[1]))
            for k in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, k]), int(edge_index[1, k])
                adj[src, dst] = float(weights[k])

        L = compute_laplacian(adj)
        evals, _ = eigendecomposition(L)
        lambda2_series[i] = float(evals[1]) if len(evals) > 1 else 0.0

        # density
        n_edges = int(np.count_nonzero(adj) // 2)
        max_edges = n_nodes * (n_nodes - 1) / 2
        density_series[i] = n_edges / max_edges if max_edges > 0 else 0.0

        # volatility
        vol_series[i] = float(np.mean(snap["node_features"][:, 0]))

    # Compute statistics for thresholds
    lam2_std = np.std(lambda2_series)
    vol_75th = np.percentile(vol_series, 75)

    labels = {}

    # Label 1: connectivity_drop_1std - lambda_2 drops by >1 std within next `horizon` steps
    y1 = np.zeros(n, dtype=int)
    for i in range(n - horizon):
        future_min = np.min(lambda2_series[i + 1: i + 1 + horizon])
        if lambda2_series[i] - future_min > lam2_std:
            y1[i] = 1
    labels["connectivity_drop_1std"] = y1

    # Label 2: connectivity_drop_05std - lambda_2 drops by >0.5 std
    y2 = np.zeros(n, dtype=int)
    for i in range(n - horizon):
        future_min = np.min(lambda2_series[i + 1: i + 1 + horizon])
        if lambda2_series[i] - future_min > 0.5 * lam2_std:
            y2[i] = 1
    labels["connectivity_drop_05std"] = y2

    # Label 3: density_drop - density drops by >10% within next `horizon` steps
    y3 = np.zeros(n, dtype=int)
    for i in range(n - horizon):
        future_min = np.min(density_series[i + 1: i + 1 + horizon])
        if density_series[i] > 1e-8:
            pct_drop = (density_series[i] - future_min) / density_series[i]
            if pct_drop > 0.10:
                y3[i] = 1
    labels["density_drop"] = y3

    # Label 4: high_vol_regime - avg volatility exceeds 75th pct within next `horizon` steps
    y4 = np.zeros(n, dtype=int)
    for i in range(n - horizon):
        future_max = np.max(vol_series[i + 1: i + 1 + horizon])
        if future_max > vol_75th:
            y4[i] = 1
    labels["high_vol_regime"] = y4

    return labels


# ── Classification pipeline ──────────────────────────────────────

def run_classification(X: np.ndarray, y: np.ndarray, label_name: str, seed: int = GLOBAL_SEED) -> Dict:
    """Train classifiers on 80/20 split + walk-forward CV, return metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler

    # Trim to usable range (exclude last HORIZON samples with unknown future)
    usable = len(y) - HORIZON
    X_use = X[:usable]
    y_use = y[:usable]

    n = len(y_use)
    pos_rate = float(np.mean(y_use))
    logger.info("  Label '%s': %d samples, %.1f%% positive", label_name, n, pos_rate * 100)

    if pos_rate == 0.0 or pos_rate == 1.0:
        logger.warning("  Skipping '%s': no class variation", label_name)
        return {
            "n_samples": n, "positive_rate": pos_rate,
            "skipped": True, "reason": "no class variation",
        }

    classifiers = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=seed, class_weight="balanced", C=1.0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=seed,
            class_weight="balanced", n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=seed, subsample=0.8,
        ),
    }

    results = {
        "n_samples": n,
        "positive_rate": round(pos_rate, 4),
        "holdout": {},
        "walk_forward": {},
    }

    # ── 80/20 temporal split (no shuffle -- temporal data) ──
    split = int(0.8 * n)
    X_train, X_test = X_use[:split], X_use[split:]
    y_train, y_test = y_use[:split], y_use[split:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    for clf_name, clf in classifiers.items():
        try:
            clf.fit(X_train_sc, y_train)
            y_pred = clf.predict(X_test_sc)

            # Probabilities for AUC
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test_sc)[:, 1]
            else:
                y_prob = clf.decision_function(X_test_sc)

            # Handle edge case: only one class in test set
            if len(np.unique(y_test)) < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y_test, y_prob))

            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))

            results["holdout"][clf_name] = {
                "AUC": round(auc, 4) if not np.isnan(auc) else None,
                "F1": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
            }
            logger.info("    %s holdout: AUC=%.4f, F1=%.4f, P=%.4f, R=%.4f",
                        clf_name, auc if not np.isnan(auc) else 0, f1, prec, rec)

        except Exception as e:
            logger.warning("    %s holdout failed: %s", clf_name, e)
            results["holdout"][clf_name] = {"error": str(e)}

    # ── Walk-forward CV (3 folds) ──
    n_folds = 3
    fold_size = n // (n_folds + 1)  # reserve first fold_size for initial train
    wf_results = {clf_name: {"AUC": [], "F1": [], "precision": [], "recall": []}
                  for clf_name in classifiers}

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

        for clf_name, clf_template in classifiers.items():
            try:
                # Clone classifier
                from sklearn.base import clone
                clf = clone(clf_template)
                clf.fit(X_tr_sc, y_tr)
                y_pred = clf.predict(X_te_sc)
                y_prob = clf.predict_proba(X_te_sc)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_te_sc)

                auc = float(roc_auc_score(y_te, y_prob))
                f1 = float(f1_score(y_te, y_pred, zero_division=0))
                prec = float(precision_score(y_te, y_pred, zero_division=0))
                rec = float(recall_score(y_te, y_pred, zero_division=0))

                wf_results[clf_name]["AUC"].append(auc)
                wf_results[clf_name]["F1"].append(f1)
                wf_results[clf_name]["precision"].append(prec)
                wf_results[clf_name]["recall"].append(rec)
            except Exception as e:
                logger.warning("    WF fold %d %s failed: %s", fold, clf_name, e)

    for clf_name in classifiers:
        aucs = wf_results[clf_name]["AUC"]
        f1s = wf_results[clf_name]["F1"]
        if aucs:
            results["walk_forward"][clf_name] = {
                "AUC_mean": round(float(np.mean(aucs)), 4),
                "AUC_std": round(float(np.std(aucs)), 4),
                "F1_mean": round(float(np.mean(f1s)), 4),
                "F1_std": round(float(np.std(f1s)), 4),
                "precision_mean": round(float(np.mean(wf_results[clf_name]["precision"])), 4),
                "recall_mean": round(float(np.mean(wf_results[clf_name]["recall"])), 4),
                "n_folds": len(aucs),
            }
        else:
            results["walk_forward"][clf_name] = {"n_folds": 0, "error": "no valid folds"}

    return results


# ── Main ─────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # Step 1: Fetch data
    logger.info("=== Step 1: Fetching eu_50 data (threshold=0.6, stride=3) ===")
    snapshots = fetch_snapshots()
    if not snapshots:
        logger.error("No data fetched. Exiting.")
        return

    logger.info("Got %d snapshots, %d banks", len(snapshots), snapshots[0]["node_features"].shape[0])

    # Step 2: Compute 36 graph-level features
    logger.info("=== Step 2: Computing 36 graph-level features ===")
    X, feature_names = compute_graph_features(snapshots)
    logger.info("Feature matrix shape: %s", X.shape)

    # Step 3: Define stress labels
    logger.info("=== Step 3: Computing stress labels (horizon=%d) ===", HORIZON)
    labels = compute_stress_labels(snapshots, X, horizon=HORIZON)

    # Step 4: Run classification for each label
    logger.info("=== Step 4: Running classification experiments ===")
    all_results = {
        "metadata": {
            "n_snapshots": len(snapshots),
            "n_banks": int(snapshots[0]["node_features"].shape[0]),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "horizon": HORIZON,
            "threshold": 0.6,
            "stride": 3,
            "universe": "eu_50",
        },
        "label_results": {},
    }

    best_auc = 0.0
    best_combo = ""

    for label_name, y in labels.items():
        logger.info("\n--- Label: %s ---", label_name)
        result = run_classification(X, y, label_name)
        all_results["label_results"][label_name] = result

        # Track best AUC
        for section in ["holdout", "walk_forward"]:
            if section in result:
                for clf_name, metrics in result[section].items():
                    auc_key = "AUC" if section == "holdout" else "AUC_mean"
                    if isinstance(metrics, dict) and auc_key in metrics and metrics[auc_key] is not None:
                        auc_val = metrics[auc_key]
                        if auc_val > best_auc:
                            best_auc = auc_val
                            best_combo = f"{label_name} + {clf_name} ({section})"

    # Summary
    above_065 = []
    for label_name, result in all_results["label_results"].items():
        for section in ["holdout", "walk_forward"]:
            if section in result:
                for clf_name, metrics in result[section].items():
                    auc_key = "AUC" if section == "holdout" else "AUC_mean"
                    if isinstance(metrics, dict) and auc_key in metrics and metrics[auc_key] is not None:
                        if metrics[auc_key] > 0.65:
                            above_065.append({
                                "label": label_name,
                                "model": clf_name,
                                "eval": section,
                                "AUC": metrics[auc_key],
                            })

    all_results["summary"] = {
        "best_auc": round(best_auc, 4),
        "best_combination": best_combo,
        "combos_above_065_AUC": above_065,
        "any_above_065": len(above_065) > 0,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # Step 5: Save
    out_path = RESULTS_DIR / "exp_binary_classification.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_safe)
    logger.info("Saved results to %s", out_path)

    # Print summary
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION RESULTS")
    print("=" * 70)

    for label_name, result in all_results["label_results"].items():
        print(f"\nLabel: {label_name}")
        print(f"  Samples: {result['n_samples']}, Positive rate: {result.get('positive_rate', 'N/A')}")
        if result.get("skipped"):
            print(f"  SKIPPED: {result.get('reason')}")
            continue
        print("  Holdout:")
        for clf_name, metrics in result.get("holdout", {}).items():
            if isinstance(metrics, dict) and "AUC" in metrics:
                print(f"    {clf_name}: AUC={metrics['AUC']}, F1={metrics['F1']}, P={metrics['precision']}, R={metrics['recall']}")
        print("  Walk-forward:")
        for clf_name, metrics in result.get("walk_forward", {}).items():
            if isinstance(metrics, dict) and "AUC_mean" in metrics:
                print(f"    {clf_name}: AUC={metrics['AUC_mean']}+/-{metrics['AUC_std']}, F1={metrics['F1_mean']}+/-{metrics['F1_std']}")

    print(f"\nBest AUC: {best_auc:.4f} ({best_combo})")
    print(f"Combinations exceeding 0.65 AUC: {len(above_065)}")
    for combo in above_065:
        print(f"  - {combo['label']} + {combo['model']} ({combo['eval']}): AUC={combo['AUC']}")

    if above_065:
        print("\n>>> YES: Some combinations exceed 0.65 AUC -- useful for early warning.")
    else:
        print("\n>>> NO: No combination exceeds 0.65 AUC -- limited early-warning value.")

    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
