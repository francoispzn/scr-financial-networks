#!/usr/bin/env python
"""Walk-forward CV for GradientBoosting on 36 graph-level features.

Tests whether GB on rich graph features genuinely beats AR(1)/persistence
across multiple forecast horizons using proper walk-forward validation.
"""
import sys
sys.path.insert(0, '.')

import json
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

# ── 1. Fetch snapshots with retry ──────────────────────────────────

def fetch_snapshots():
    from dashboard.data_api import build_daily_graph_snapshots
    for attempt in range(3):
        try:
            snaps = build_daily_graph_snapshots(
                lookback_years=3, corr_window=60,
                min_corr=0.3, stride=5,
            )
            if len(snaps) > 50:
                logger.info("Fetched %d snapshots (attempt %d)", len(snaps), attempt + 1)
                return snaps
            logger.warning("Only %d snapshots (attempt %d), retrying...", len(snaps), attempt + 1)
            time.sleep(5)
        except Exception as e:
            logger.warning("Fetch failed (attempt %d): %s", attempt + 1, e)
            time.sleep(5)
    raise RuntimeError("Could not fetch sufficient data after 3 attempts")


# ── 2. Build 36 graph-level features ──────────────────────────────

def build_graph_features(snapshots):
    """Extract 36 graph-level features per snapshot.

    Feature groups:
      Network topology (8):  n_edges, density, avg_degree, std_degree, max_degree,
                              clustering_coeff, assortativity, n_components
      Correlation stats (6): corr_mean, corr_std, corr_q25, corr_q75, corr_skew, corr_kurt
      Market aggregates (6): avg_vol, std_vol, avg_return, std_return, avg_beta, avg_momentum
      Eigenvalue dist (7):   lam2, gap, rho, ev_entropy, ev_ratio_12, ev_sum, ev_std
      SCG / derived (3):     scg_risk, spectral_resilience, fiedler_ratio
      Change features (6):   lam2_change, gap_change, rho_change,
                              edge_change_abs, vol_change, corr_mean_change
    Total: 36
    """
    feature_names = [
        # Topology
        "n_edges", "density", "avg_degree", "std_degree", "max_degree",
        "clustering_coeff", "assortativity", "n_components",
        # Correlation
        "corr_mean", "corr_std", "corr_q25", "corr_q75", "corr_skew", "corr_kurt",
        # Market
        "avg_vol", "std_vol", "avg_return", "std_return", "avg_beta", "avg_momentum",
        # Eigenvalue distribution
        "lam2", "gap", "rho", "ev_entropy", "ev_ratio_12", "ev_sum", "ev_std",
        # SCG / derived
        "scg_risk", "spectral_resilience", "fiedler_ratio",
        # Change features (diff from prev snapshot)
        "lam2_change", "gap_change", "rho_change",
        "edge_change_abs", "vol_change", "corr_mean_change",
    ]
    assert len(feature_names) == 36

    rows = []
    prev = None
    for s in snapshots:
        adj = _adj_from_snapshot(s)
        n = adj.shape[0]
        n_possible = n * (n - 1) / 2

        # --- Topology ---
        n_edges = s["edge_index"].shape[1] // 2
        density = n_edges / n_possible if n_possible > 0 else 0
        degrees = np.sum(adj > 0, axis=1).astype(float)
        avg_deg = float(np.mean(degrees))
        std_deg = float(np.std(degrees))
        max_deg = float(np.max(degrees))

        # Clustering coefficient (local average)
        clustering = _avg_clustering(adj)
        # Assortativity (degree correlation)
        assort = _assortativity(adj, degrees)
        # Connected components
        n_comp = _n_components(adj)

        # --- Correlation stats (from edge weights) ---
        weights = adj[adj > 0]
        if len(weights) == 0:
            weights = np.array([0.0])
        corr_mean = float(np.mean(weights))
        corr_std = float(np.std(weights))
        corr_q25 = float(np.percentile(weights, 25))
        corr_q75 = float(np.percentile(weights, 75))
        corr_skew = float(sp_stats.skew(weights)) if len(weights) > 2 else 0.0
        corr_kurt = float(sp_stats.kurtosis(weights)) if len(weights) > 2 else 0.0

        # --- Market aggregates ---
        nf = s["node_features"]  # [N, 5]: vol, return, log_price, beta, momentum
        avg_vol = float(np.mean(nf[:, 0]))
        std_vol = float(np.std(nf[:, 0]))
        avg_ret = float(np.mean(nf[:, 1]))
        std_ret = float(np.std(nf[:, 1]))
        avg_beta = float(np.mean(nf[:, 3]))
        avg_mom = float(np.mean(nf[:, 4]))

        # --- Eigenvalue distribution ---
        lam2 = s["targets"]["lambda_2"]
        gap = s["targets"]["spectral_gap"]
        rho = s["targets"]["spectral_radius"]

        # Compute Laplacian eigenvalues for distribution stats
        from scr_financial.network.spectral import compute_laplacian, eigendecomposition
        L = compute_laplacian(adj, normalized=True)
        evals, _ = eigendecomposition(L)
        evals_pos = evals[evals > 1e-10]
        if len(evals_pos) > 0:
            ev_norm = evals_pos / np.sum(evals_pos)
            ev_entropy = float(-np.sum(ev_norm * np.log(ev_norm + 1e-15)))
        else:
            ev_entropy = 0.0
        ev_ratio_12 = float(evals[1] / evals[2]) if len(evals) > 2 and evals[2] > 1e-10 else 0.0
        ev_sum = float(np.sum(evals))
        ev_std = float(np.std(evals))

        # --- SCG / derived ---
        scg_risk = 1 - lam2 / rho if rho > 1e-10 else 1.0
        spectral_resilience = lam2 * gap if gap > 0 else 0.0
        fiedler_ratio = lam2 / ev_sum if ev_sum > 1e-10 else 0.0

        # --- Change features ---
        if prev is not None:
            lam2_change = lam2 - prev["lam2"]
            gap_change = gap - prev["gap"]
            rho_change = rho - prev["rho"]
            edge_change_abs = abs(n_edges - prev["n_edges"])
            vol_change = avg_vol - prev["avg_vol"]
            corr_mean_change = corr_mean - prev["corr_mean"]
        else:
            lam2_change = gap_change = rho_change = 0.0
            edge_change_abs = vol_change = corr_mean_change = 0.0

        prev = {"lam2": lam2, "gap": gap, "rho": rho,
                "n_edges": n_edges, "avg_vol": avg_vol, "corr_mean": corr_mean}

        row = [
            n_edges, density, avg_deg, std_deg, max_deg,
            clustering, assort, n_comp,
            corr_mean, corr_std, corr_q25, corr_q75, corr_skew, corr_kurt,
            avg_vol, std_vol, avg_ret, std_ret, avg_beta, avg_mom,
            lam2, gap, rho, ev_entropy, ev_ratio_12, ev_sum, ev_std,
            scg_risk, spectral_resilience, fiedler_ratio,
            lam2_change, gap_change, rho_change,
            edge_change_abs, vol_change, corr_mean_change,
        ]
        rows.append(row)

    X = np.array(rows, dtype=np.float64)
    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feature_names


def _adj_from_snapshot(s):
    """Reconstruct adjacency matrix from edge_index and edge_weight."""
    ei = s["edge_index"]
    ew = s.get("edge_weight", None)
    n = s["node_features"].shape[0]
    adj = np.zeros((n, n), dtype=np.float32)
    if ei.shape[1] > 0:
        for k in range(ei.shape[1]):
            i, j = ei[0, k], ei[1, k]
            w = float(ew[k]) if ew is not None and len(ew) > k else 1.0
            adj[i, j] = w
    return adj


def _avg_clustering(adj):
    """Average local clustering coefficient."""
    n = adj.shape[0]
    binary = (adj > 0).astype(float)
    coeffs = []
    for i in range(n):
        neighbors = np.where(binary[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            coeffs.append(0.0)
            continue
        # Count triangles
        sub = binary[np.ix_(neighbors, neighbors)]
        triangles = np.sum(sub) / 2
        coeffs.append(2 * triangles / (k * (k - 1)))
    return float(np.mean(coeffs))


def _assortativity(adj, degrees):
    """Degree assortativity (Pearson correlation of degrees at edge endpoints)."""
    rows, cols = np.where(adj > 0)
    if len(rows) < 4:
        return 0.0
    d_src = degrees[rows]
    d_dst = degrees[cols]
    if np.std(d_src) < 1e-10 or np.std(d_dst) < 1e-10:
        return 0.0
    return float(np.corrcoef(d_src, d_dst)[0, 1])


def _n_components(adj):
    """Number of connected components via BFS."""
    n = adj.shape[0]
    binary = (adj > 0)
    visited = np.zeros(n, dtype=bool)
    comps = 0
    for start in range(n):
        if visited[start]:
            continue
        comps += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            for nb in range(n):
                if binary[node, nb] and not visited[nb]:
                    stack.append(nb)
    return comps


# ── 3. Build multi-horizon targets ────────────────────────────────

def build_targets(snapshots, horizon):
    """For each snapshot t, target = average of next h values of [lam2, gap, rho].

    Returns targets array of shape [T - horizon, 3].
    """
    lam2 = np.array([s["targets"]["lambda_2"] for s in snapshots])
    gap = np.array([s["targets"]["spectral_gap"] for s in snapshots])
    rho = np.array([s["targets"]["spectral_radius"] for s in snapshots])

    targets = []
    for t in range(len(snapshots) - horizon):
        targets.append([
            np.mean(lam2[t + 1: t + 1 + horizon]),
            np.mean(gap[t + 1: t + 1 + horizon]),
            np.mean(rho[t + 1: t + 1 + horizon]),
        ])
    return np.array(targets)


# ── 4. Walk-forward CV ────────────────────────────────────────────

class WalkForwardCV:
    def __init__(self, n_splits=5, min_train_size=100, test_size=40, gap_size=10):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.gap_size = gap_size

    def split(self, n_samples):
        """Yield (train_idx, test_idx) for each fold."""
        # Total needed per fold = min_train + gap + test
        # Space folds so they use all data
        total_per_fold = self.min_train_size + self.gap_size + self.test_size
        # Expanding window: each fold adds more training data
        available = n_samples - self.gap_size - self.test_size
        if available < self.min_train_size:
            raise ValueError(f"Not enough samples ({n_samples}) for walk-forward CV")

        # Evenly space test windows across available range
        max_test_start = n_samples - self.test_size
        min_test_start = self.min_train_size + self.gap_size
        if max_test_start <= min_test_start:
            raise ValueError("Not enough samples for requested CV config")

        test_starts = np.linspace(min_test_start, max_test_start,
                                  self.n_splits, dtype=int)

        for test_start in test_starts:
            test_end = test_start + self.test_size
            train_end = test_start - self.gap_size
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, min(test_end, n_samples))
            if len(train_idx) >= self.min_train_size and len(test_idx) > 0:
                yield train_idx, test_idx


def persistence_predict(X_train, y_train, X_test, target_col_in_X):
    """Persistence baseline: predict y_{t+h} = current value (last known).

    The current spectral values are columns 20, 21, 22 in the 36-feature vector
    (lam2, gap, rho).
    """
    # lam2 is feature index 20, gap=21, rho=22
    return X_test[:, [20, 21, 22]]


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0


def per_target_r2(y_true, y_pred, names=None):
    if names is None:
        names = ["lambda_2", "spectral_gap", "spectral_radius"]
    out = {}
    for i, name in enumerate(names):
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        out[name] = float(1 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
    return out


# ── 5. Main experiment ────────────────────────────────────────────

def main():
    from scr_financial.stats.hypothesis_tests import diebold_mariano_test

    logger.info("=" * 70)
    logger.info("Walk-Forward CV: GradientBoosting on 36 Graph Features")
    logger.info("=" * 70)

    # Fetch data
    t0 = time.time()
    snapshots = fetch_snapshots()
    logger.info("Data fetch took %.1fs", time.time() - t0)

    # Build features
    logger.info("Building 36 graph-level features...")
    X_all, feature_names = build_graph_features(snapshots)
    logger.info("Feature matrix: %s", X_all.shape)

    horizons = [1, 5, 10]
    target_names = ["lambda_2", "spectral_gap", "spectral_radius"]
    cv = WalkForwardCV(n_splits=5, min_train_size=100, test_size=40, gap_size=10)

    results = {
        "n_snapshots": len(snapshots),
        "n_features": 36,
        "feature_names": feature_names,
        "cv_config": {"n_splits": 5, "min_train_size": 100, "test_size": 40, "gap_size": 10},
        "horizons": {},
    }

    for h in horizons:
        logger.info("\n" + "─" * 50)
        logger.info("HORIZON h=%d", h)
        logger.info("─" * 50)

        # Build targets for this horizon
        y_all = build_targets(snapshots, h)
        # Align: X has len(snapshots) rows, y has len(snapshots) - h rows
        # Use X[0:len(y)] aligned with y
        n = min(len(X_all), len(y_all))
        X = X_all[:n]
        y = y_all[:n]
        logger.info("Samples after alignment: %d", n)

        gb_fold_r2 = []
        persist_fold_r2 = []
        gb_per_target_r2 = {t: [] for t in target_names}
        persist_per_target_r2 = {t: [] for t in target_names}
        dm_errors_gb = []
        dm_errors_persist = []

        fold_details = []

        for fold_i, (train_idx, test_idx) in enumerate(cv.split(n)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            logger.info("  Fold %d: train=%d, test=%d", fold_i + 1, len(train_idx), len(test_idx))

            # --- GradientBoosting ---
            gb = MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=200, max_depth=4, random_state=SEED,
                    learning_rate=0.1, subsample=0.8,
                )
            )
            gb.fit(X_train, y_train)
            y_pred_gb = gb.predict(X_test)

            # --- Persistence baseline ---
            y_pred_persist = persistence_predict(X_train, y_train, X_test, None)

            # R2 scores
            r2_gb = r2_score(y_test, y_pred_gb)
            r2_persist = r2_score(y_test, y_pred_persist)
            gb_fold_r2.append(r2_gb)
            persist_fold_r2.append(r2_persist)

            # Per-target R2
            pt_gb = per_target_r2(y_test, y_pred_gb)
            pt_persist = per_target_r2(y_test, y_pred_persist)
            for t in target_names:
                gb_per_target_r2[t].append(pt_gb[t])
                persist_per_target_r2[t].append(pt_persist[t])

            # Collect errors for DM test
            e_gb = (y_test - y_pred_gb).flatten()
            e_persist = (y_test - y_pred_persist).flatten()
            dm_errors_gb.append(e_gb)
            dm_errors_persist.append(e_persist)

            fold_detail = {
                "fold": fold_i + 1,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "gb_r2": round(r2_gb, 4),
                "persist_r2": round(r2_persist, 4),
                "gb_per_target": {t: round(v, 4) for t, v in pt_gb.items()},
                "persist_per_target": {t: round(v, 4) for t, v in pt_persist.items()},
            }
            fold_details.append(fold_detail)
            logger.info("    GB R²=%.4f  Persist R²=%.4f", r2_gb, r2_persist)

        # Aggregate
        gb_r2_mean = float(np.mean(gb_fold_r2))
        gb_r2_std = float(np.std(gb_fold_r2))
        persist_r2_mean = float(np.mean(persist_fold_r2))
        persist_r2_std = float(np.std(persist_fold_r2))

        # Diebold-Mariano test (concatenated errors across folds)
        all_e_gb = np.concatenate(dm_errors_gb)
        all_e_persist = np.concatenate(dm_errors_persist)
        dm_result = diebold_mariano_test(all_e_persist, all_e_gb, horizon=h, loss='mse')

        # Per-target DM tests
        dm_per_target = {}
        for ti, t in enumerate(target_names):
            e_gb_t = np.concatenate([e.reshape(-1, 3)[:, ti] for e in dm_errors_gb])
            e_p_t = np.concatenate([e.reshape(-1, 3)[:, ti] for e in dm_errors_persist])
            dm_t = diebold_mariano_test(e_p_t, e_gb_t, horizon=h, loss='mse')
            dm_per_target[t] = {
                "dm_stat": round(dm_t["dm_stat"], 4),
                "p_value": round(dm_t["p_value"], 4),
                "significant_0.05": dm_t["p_value"] < 0.05,
            }

        horizon_result = {
            "gb": {
                "r2_mean": round(gb_r2_mean, 4),
                "r2_std": round(gb_r2_std, 4),
                "per_target_r2_mean": {t: round(float(np.mean(gb_per_target_r2[t])), 4) for t in target_names},
                "per_target_r2_std": {t: round(float(np.std(gb_per_target_r2[t])), 4) for t in target_names},
            },
            "persistence": {
                "r2_mean": round(persist_r2_mean, 4),
                "r2_std": round(persist_r2_std, 4),
                "per_target_r2_mean": {t: round(float(np.mean(persist_per_target_r2[t])), 4) for t in target_names},
                "per_target_r2_std": {t: round(float(np.std(persist_per_target_r2[t])), 4) for t in target_names},
            },
            "diebold_mariano_aggregate": {
                "dm_stat": round(dm_result["dm_stat"], 4),
                "p_value": round(dm_result["p_value"], 4),
                "n_errors": dm_result["n"],
                "significant_0.05": dm_result["p_value"] < 0.05,
                "gb_significantly_better": dm_result["dm_stat"] > 0 and dm_result["p_value"] < 0.05,
            },
            "diebold_mariano_per_target": dm_per_target,
            "fold_details": fold_details,
        }

        results["horizons"][f"h={h}"] = horizon_result

        logger.info("\n  SUMMARY h=%d:", h)
        logger.info("    GB:          R²=%.4f ± %.4f", gb_r2_mean, gb_r2_std)
        logger.info("    Persistence: R²=%.4f ± %.4f", persist_r2_mean, persist_r2_std)
        logger.info("    DM test:     stat=%.4f, p=%.4f, sig=%s",
                     dm_result["dm_stat"], dm_result["p_value"],
                     dm_result["p_value"] < 0.05)
        for t in target_names:
            logger.info("    %s: GB=%.4f±%.4f  Persist=%.4f±%.4f  DM p=%.4f",
                         t,
                         np.mean(gb_per_target_r2[t]), np.std(gb_per_target_r2[t]),
                         np.mean(persist_per_target_r2[t]), np.std(persist_per_target_r2[t]),
                         dm_per_target[t]["p_value"])

    # Save results
    out_dir = Path("experiments/results/publication")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp_gb_walkforward.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nResults saved to %s", out_path)

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    for h_key, h_res in results["horizons"].items():
        gb = h_res["gb"]
        p = h_res["persistence"]
        dm = h_res["diebold_mariano_aggregate"]
        sig_str = "YES" if dm["gb_significantly_better"] else "NO"
        logger.info("%s:  GB R²=%.4f±%.4f  vs  Persist R²=%.4f±%.4f  |  DM p=%.4f  Sig: %s",
                     h_key, gb["r2_mean"], gb["r2_std"],
                     p["r2_mean"], p["r2_std"], dm["p_value"], sig_str)
        for t in ["lambda_2", "spectral_gap", "spectral_radius"]:
            dm_t = h_res["diebold_mariano_per_target"][t]
            logger.info("  %s: GB=%.4f  Persist=%.4f  DM p=%.4f %s",
                         t, gb["per_target_r2_mean"][t], p["per_target_r2_mean"][t],
                         dm_t["p_value"], "*" if dm_t["significant_0.05"] else "")


if __name__ == "__main__":
    main()
