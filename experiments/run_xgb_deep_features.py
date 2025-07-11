#!/usr/bin/env python3
"""
Deep feature importance analysis for spectral prediction.

Which graph features ACTUALLY predict spectral changes?
Uses XGBoost + SHAP + walk-forward CV + ablation.
"""
import sys
sys.path.insert(0, '.')

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("xgb_deep_features")

RESULTS_DIR = Path("experiments/results/publication")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GLOBAL_SEED = 42

# ── JSON-safe helper ─────────────────────────────────────────────────
def _json_safe(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
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
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── Step 1: Fetch eu_50 data with robust handling ────────────────────
def fetch_snapshots():
    """Fetch eu_50 data directly, handling partial ticker failures robustly."""
    import yfinance as yf
    from scr_financial.config.loader import load_universe
    from scr_financial.network.spectral import (
        compute_laplacian, eigendecomposition, find_spectral_gap,
        analyze_spectral_properties,
    )

    univ = load_universe("eu_50")
    all_tickers = {b.id: b.ticker for b in univ.banks}
    all_ids = univ.ids

    corr_window = 60
    min_corr = 0.6
    stride = 3
    lookback_years = 3

    for attempt in range(3):
        try:
            logger.info("Fetching prices (attempt %d)...", attempt + 1)

            # Download in small batches to avoid yfinance multi-ticker issues
            all_prices = {}
            ticker_to_id = {v: k for k, v in all_tickers.items()}
            ticker_list = list(all_tickers.values())
            batch_size = 8

            for batch_start in range(0, len(ticker_list), batch_size):
                batch = ticker_list[batch_start:batch_start + batch_size]
                try:
                    if len(batch) == 1:
                        raw = yf.download(batch[0], period="3y", progress=False, auto_adjust=True)
                        if not raw.empty and "Close" in raw.columns:
                            bid = ticker_to_id[batch[0]]
                            all_prices[bid] = raw["Close"]
                    else:
                        raw = yf.download(batch, period="3y", progress=False, auto_adjust=True)
                        if not raw.empty:
                            if raw.columns.nlevels == 2:
                                close = raw["Close"]
                            else:
                                close = raw
                            for col in close.columns:
                                ticker_str = str(col)
                                if ticker_str in ticker_to_id:
                                    series = close[col].dropna()
                                    if len(series) > 100:
                                        all_prices[ticker_to_id[ticker_str]] = series
                except Exception as batch_err:
                    logger.debug("Batch download failed for %s: %s", batch, batch_err)

            if not all_prices:
                logger.warning("No prices fetched, retrying...")
                time.sleep(5)
                continue

            prices = pd.DataFrame(all_prices)
            # Drop columns that are all NaN
            prices = prices.dropna(axis=1, how="all")
            # Forward-fill then drop remaining NaN rows from the start
            prices = prices.ffill().bfill().dropna()

            # Only keep bank IDs that successfully downloaded
            valid_ids = [bid for bid in all_ids if bid in prices.columns]
            prices = prices[valid_ids]
            n_banks = len(valid_ids)

            logger.info("Got prices for %d/%d banks, %d trading days",
                        n_banks, len(all_ids), len(prices))

            if n_banks < 10 or len(prices) < corr_window + 50:
                logger.warning("Insufficient data, retrying...")
                time.sleep(5)
                continue

            # Compute returns
            returns = prices.pct_change().dropna()
            logger.info("Computing snapshots from %d returns, %d banks, corr_window=%d, stride=%d",
                        len(returns), n_banks, corr_window, stride)

            # Build graph snapshots
            snapshots = []
            valid_dates = returns.index[corr_window:]

            for count, date_idx in enumerate(range(0, len(valid_dates), stride)):
                date = valid_dates[date_idx]
                window_end = corr_window + date_idx
                ret_window = returns.iloc[window_end - corr_window: window_end]

                # Correlation adjacency
                corr = ret_window.corr()
                n = n_banks
                adj = np.zeros((n, n), dtype=np.float32)
                for i, src in enumerate(valid_ids):
                    for j, tgt in enumerate(valid_ids):
                        if i >= j:
                            continue
                        w = float(corr.loc[src, tgt]) if (src in corr.index and tgt in corr.index) else 0
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
                        node_feats[i, 0] = float(np.std(rets_i) * np.sqrt(252))
                        node_feats[i, 1] = float(np.mean(rets_i) * 252)
                        cum_ret = (1 + ret_window[bid]).prod()
                        node_feats[i, 2] = float(np.log(max(cum_ret, 0.01)))
                        mkt = ret_window.mean(axis=1).values
                        cov_val = np.cov(rets_i, mkt)[0, 1] if len(rets_i) > 2 else 0
                        var_mkt = np.var(mkt) if np.var(mkt) > 1e-10 else 1.0
                        node_feats[i, 3] = float(cov_val / var_mkt)
                        if len(rets_i) >= 20:
                            node_feats[i, 4] = float(np.sum(rets_i[-20:]))

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
                    "lambda_2": lam2,
                    "spectral_gap": gap,
                    "spectral_radius": radius,
                    "time": count,
                    "date": str(date.date()) if hasattr(date, 'date') else str(date),
                    "n_banks": n_banks,
                })

            logger.info("Built %d snapshots (stride=%d)", len(snapshots), stride)
            if len(snapshots) >= 30:
                return snapshots

            logger.warning("Only %d snapshots, retrying...", len(snapshots))
            time.sleep(5)

        except Exception as e:
            logger.warning("Fetch failed (attempt %d): %s", attempt + 1, e)
            import traceback
            traceback.print_exc()
            time.sleep(5)

    logger.error("Could not fetch sufficient snapshots after 3 attempts.")
    return []


# ── Step 2: Compute exhaustive feature set ───────────────────────────
def compute_features(snapshots):
    """Build a DataFrame of 60+ features from graph snapshots."""
    from scr_financial.network.spectral import (
        compute_laplacian, eigendecomposition, find_spectral_gap,
    )

    records = []

    for t, snap in enumerate(snapshots):
        node_feats = snap["node_features"]  # [N, 5]: vol, ret, log_price, beta, momentum
        n = node_feats.shape[0]

        # Reconstruct adjacency from edge_index / edge_weight
        adj = np.zeros((n, n), dtype=np.float32)
        ei = snap["edge_index"]
        ew = snap["edge_weight"]
        if ei.shape[1] > 0:
            for k in range(ei.shape[1]):
                adj[ei[0, k], ei[1, k]] = ew[k]

        # --- Spectral targets (already computed) ---
        lam2 = snap["lambda_2"]
        gap = snap["spectral_gap"]
        rho = snap["spectral_radius"]

        # --- Full Laplacian for extra spectral features ---
        adj_sym = (adj + adj.T) / 2.0
        L = compute_laplacian(adj_sym, normalized=True)
        evals, evecs = eigendecomposition(L)

        # --- BASIC GRAPH FEATURES ---
        degrees = adj_sym.sum(axis=1)
        n_edges = (adj_sym > 0).sum() / 2
        density = float(2 * n_edges / (n * (n - 1))) if n > 1 else 0

        row = {}
        row["t"] = t
        row["date"] = snap.get("date", "")

        # Graph structure (12 features)
        row["density"] = density
        row["n_edges"] = float(n_edges)
        row["avg_degree"] = float(degrees.mean())
        row["max_degree"] = float(degrees.max())
        row["min_degree"] = float(degrees.min())
        row["std_degree"] = float(degrees.std())
        row["degree_skew"] = float(skew(degrees)) if len(degrees) > 2 else 0
        row["degree_kurtosis"] = float(kurtosis(degrees)) if len(degrees) > 2 else 0
        row["avg_weight"] = float(adj_sym[adj_sym > 0].mean()) if (adj_sym > 0).any() else 0
        row["std_weight"] = float(adj_sym[adj_sym > 0].std()) if (adj_sym > 0).any() else 0
        row["max_weight"] = float(adj_sym.max())
        row["weight_sum"] = float(adj_sym.sum() / 2)

        # Correlation distribution features (8 features)
        upper = adj_sym[np.triu_indices(n, k=1)]
        all_corr = upper[upper > 0] if (upper > 0).any() else upper
        row["corr_mean"] = float(all_corr.mean()) if len(all_corr) > 0 else 0
        row["corr_std"] = float(all_corr.std()) if len(all_corr) > 0 else 0
        row["corr_q25"] = float(np.percentile(all_corr, 25)) if len(all_corr) > 0 else 0
        row["corr_q50"] = float(np.percentile(all_corr, 50)) if len(all_corr) > 0 else 0
        row["corr_q75"] = float(np.percentile(all_corr, 75)) if len(all_corr) > 0 else 0
        row["corr_q90"] = float(np.percentile(all_corr, 90)) if len(all_corr) > 0 else 0
        row["corr_skew"] = float(skew(all_corr)) if len(all_corr) > 2 else 0
        row["corr_iqr"] = row["corr_q75"] - row["corr_q25"]

        # Spectral features (10 features)
        row["lam2"] = lam2
        row["gap"] = gap
        row["rho"] = rho
        row["lam3"] = float(evals[2]) if len(evals) > 2 else 0
        row["lam4"] = float(evals[3]) if len(evals) > 3 else 0
        row["lam_ratio_23"] = float(evals[2] / max(evals[1], 1e-10)) if len(evals) > 2 else 0
        row["ev_entropy"] = float(-np.sum(evals[evals > 1e-10] * np.log(evals[evals > 1e-10] + 1e-15))) if (evals > 1e-10).any() else 0
        row["ev_energy"] = float(np.sum(evals ** 2))
        row["ev_sum"] = float(np.sum(evals))
        row["fiedler_localization"] = float(np.sum(evecs[:, 1] ** 4)) if evecs.shape[1] > 1 else 0

        # Node feature aggregates (6 features)
        avg_vol = float(node_feats[:, 0].mean())
        avg_ret = float(node_feats[:, 1].mean())
        avg_beta = float(node_feats[:, 3].mean())
        avg_momentum = float(node_feats[:, 4].mean())
        row["avg_vol"] = avg_vol
        row["std_vol"] = float(node_feats[:, 0].std())
        row["avg_ret"] = avg_ret
        row["avg_beta"] = avg_beta
        row["avg_momentum"] = avg_momentum
        row["vol_dispersion"] = float(node_feats[:, 0].max() - node_feats[:, 0].min())

        # SCG-related features
        gap_idx_val, _ = find_spectral_gap(evals, adjacency_matrix=adj_sym)
        row["scg_k"] = int(gap_idx_val)
        row["scg_ratio"] = float(gap_idx_val / n) if n > 0 else 0

        # INTERACTION features (3 features)
        row["corr_std_x_avg_vol"] = row["corr_std"] * avg_vol
        row["density_x_avg_vol"] = density * avg_vol
        row["lam2_x_avg_vol"] = lam2 * avg_vol

        records.append(row)

    df = pd.DataFrame(records)
    df = df.sort_values("t").reset_index(drop=True)

    # --- LAGGED features ---
    for lag in [1, 2, 3]:
        df[f"lam2_lag{lag}"] = df["lam2"].shift(lag)
    df["gap_lag1"] = df["gap"].shift(1)
    df["rho_lag1"] = df["rho"].shift(1)

    # --- ROLLING features ---
    for col, windows in [("lam2", [5, 10]), ("gap", [5, 10]), ("density", [5, 10])]:
        for w in windows:
            df[f"{col}_rmean{w}"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_rstd{w}"] = df[col].rolling(w, min_periods=1).std().fillna(0)

    # --- DIFF features ---
    df["lam2_diff5"] = df["lam2"] - df["lam2"].shift(5)
    df["density_diff5"] = df["density"] - df["density"].shift(5)

    # --- Target: next-step lambda_2 ---
    df["target_lam2_next"] = df["lam2"].shift(-1)

    # Drop rows with NaN targets or NaN from lagging
    df = df.dropna(subset=["target_lam2_next"]).reset_index(drop=True)
    # Drop the first few rows where lags are NaN
    df = df.iloc[3:].reset_index(drop=True)

    # Count feature columns
    exclude = {"t", "date", "target_lam2_next"}
    feature_cols = [c for c in df.columns if c not in exclude]
    logger.info("Feature matrix: %d samples, %d features", len(df), len(feature_cols))
    return df


# ── Step 4-8: Train, evaluate, ablate ────────────────────────────────
def run_analysis(df):
    import xgboost as xgb
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.inspection import permutation_importance

    # Identify feature columns (exclude metadata and target)
    exclude = {"t", "date", "target_lam2_next"}
    feature_cols = [c for c in df.columns if c not in exclude]
    logger.info("Using %d features", len(feature_cols))

    X = df[feature_cols].values.astype(np.float64)
    y = df["target_lam2_next"].values.astype(np.float64)

    # Fill any remaining NaN with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Full model training (on 80/20 split for importance) ──────────
    n = len(X)
    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=GLOBAL_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    r2_xgb = r2_score(y_test, y_pred)
    rmse_xgb = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae_xgb = float(mean_absolute_error(y_test, y_pred))

    # Persistence baseline: lam2(t) as prediction for lam2(t+1)
    lam2_col_idx = feature_cols.index("lam2")
    y_persist = X_test[:, lam2_col_idx]
    r2_persist = r2_score(y_test, y_persist)
    rmse_persist = float(np.sqrt(mean_squared_error(y_test, y_persist)))

    # AR(1) baseline
    lam2_train = X_train[:, lam2_col_idx]
    from numpy.linalg import lstsq
    A_ar1 = np.column_stack([np.ones(len(lam2_train)), lam2_train])
    coefs, _, _, _ = lstsq(A_ar1, y_train, rcond=None)
    lam2_test = X_test[:, lam2_col_idx]
    y_ar1_pred = coefs[0] + coefs[1] * lam2_test
    r2_ar1 = r2_score(y_test, y_ar1_pred)
    rmse_ar1 = float(np.sqrt(mean_squared_error(y_test, y_ar1_pred)))

    logger.info("=== HOLD-OUT RESULTS ===")
    logger.info("  XGBoost R2=%.4f  RMSE=%.6f  MAE=%.6f", r2_xgb, rmse_xgb, mae_xgb)
    logger.info("  Persistence R2=%.4f  RMSE=%.6f", r2_persist, rmse_persist)
    logger.info("  AR(1) R2=%.4f  RMSE=%.6f", r2_ar1, rmse_ar1)

    # ── Built-in feature importance ──────────────────────────────────
    fi = model.feature_importances_
    fi_pairs = sorted(zip(feature_cols, fi), key=lambda x: -x[1])
    logger.info("\n=== TOP 15 FEATURES (built-in importance) ===")
    for i, (name, imp) in enumerate(fi_pairs[:15]):
        logger.info("  %2d. %-30s %.4f", i + 1, name, imp)

    # ── Permutation importance ───────────────────────────────────────
    perm = permutation_importance(model, X_test, y_test, n_repeats=10,
                                  random_state=GLOBAL_SEED, n_jobs=-1)
    perm_pairs = sorted(zip(feature_cols, perm.importances_mean),
                        key=lambda x: -x[1])
    logger.info("\n=== TOP 15 FEATURES (permutation importance) ===")
    for i, (name, imp) in enumerate(perm_pairs[:15]):
        logger.info("  %2d. %-30s %.6f", i + 1, name, imp)

    # ── SHAP values ──────────────────────────────────────────────────
    shap_results = None
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_pairs = sorted(zip(feature_cols, mean_abs_shap), key=lambda x: -x[1])
        logger.info("\n=== TOP 15 FEATURES (SHAP mean |value|) ===")
        for i, (name, val) in enumerate(shap_pairs[:15]):
            logger.info("  %2d. %-30s %.6f", i + 1, name, val)
        shap_results = {
            "top_features": [[name, round(float(val), 6)] for name, val in shap_pairs[:30]],
            "available": True,
        }
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        shap_results = {"available": False, "error": str(e)}

    # ── Walk-forward CV (5 folds) ────────────────────────────────────
    logger.info("\n=== WALK-FORWARD CV (5 folds) ===")
    n_folds = 5
    fold_size = n // (n_folds + 1)
    wf_results = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 2)
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_end <= test_start:
            continue

        X_tr = X[:train_end]
        y_tr = y[:train_end]
        X_te = X[test_start:test_end]
        y_te = y[test_start:test_end]

        m = xgb.XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=GLOBAL_SEED, n_jobs=-1,
        )
        m.fit(X_tr, y_tr, verbose=False)
        y_p = m.predict(X_te)

        r2_fold = r2_score(y_te, y_p)
        rmse_fold = float(np.sqrt(mean_squared_error(y_te, y_p)))

        # Persistence for this fold
        lam2_te = X_te[:, lam2_col_idx]
        r2_persist_fold = r2_score(y_te, lam2_te)

        # AR(1) for this fold
        lam2_tr = X_tr[:, lam2_col_idx]
        A_ar = np.column_stack([np.ones(len(lam2_tr)), lam2_tr])
        c_ar, _, _, _ = lstsq(A_ar, y_tr, rcond=None)
        y_ar_p = c_ar[0] + c_ar[1] * lam2_te
        r2_ar_fold = r2_score(y_te, y_ar_p)

        wf_results.append({
            "fold": fold,
            "train_size": len(X_tr),
            "test_size": len(X_te),
            "r2_xgb": round(r2_fold, 4),
            "r2_persistence": round(r2_persist_fold, 4),
            "r2_ar1": round(r2_ar_fold, 4),
            "rmse_xgb": round(rmse_fold, 6),
            "xgb_beats_persistence": bool(r2_fold > r2_persist_fold),
            "xgb_beats_ar1": bool(r2_fold > r2_ar_fold),
        })
        logger.info("  Fold %d: XGB R2=%.4f  Persist R2=%.4f  AR(1) R2=%.4f  [train=%d, test=%d]",
                     fold, r2_fold, r2_persist_fold, r2_ar_fold, len(X_tr), len(X_te))

    wf_r2_xgb = [f["r2_xgb"] for f in wf_results]
    wf_r2_persist = [f["r2_persistence"] for f in wf_results]
    wf_r2_ar1 = [f["r2_ar1"] for f in wf_results]
    mean_r2_xgb = float(np.mean(wf_r2_xgb))
    mean_r2_persist = float(np.mean(wf_r2_persist))
    mean_r2_ar1 = float(np.mean(wf_r2_ar1))
    logger.info("\n  Walk-forward mean: XGB=%.4f  Persist=%.4f  AR(1)=%.4f",
                mean_r2_xgb, mean_r2_persist, mean_r2_ar1)

    # ── Diebold-Mariano test: XGB vs AR(1) ───────────────────────────
    y_pred_xgb_full = model.predict(X_test)
    e_xgb = y_test - y_pred_xgb_full
    e_ar1 = y_test - y_ar1_pred
    from scipy.stats import t as t_dist
    d = e_xgb ** 2 - e_ar1 ** 2
    n_dm = len(d)
    d_bar = np.mean(d)
    var_d = np.var(d, ddof=1) / n_dm
    if var_d > 0:
        dm_stat = d_bar / np.sqrt(var_d)
        dm_pval = 2 * (1 - t_dist.cdf(abs(dm_stat), df=n_dm - 1))
    else:
        dm_stat, dm_pval = 0.0, 1.0
    logger.info("\n  Diebold-Mariano XGB vs AR(1): DM=%.3f  p=%.4f", dm_stat, dm_pval)
    significant = bool(dm_pval < 0.05)
    logger.info("  XGB significantly beats AR(1)? %s (p=%.4f)", significant, dm_pval)

    # ── ABLATION: remove top features one at a time ──────────────────
    logger.info("\n=== ABLATION: Remove top features one at a time ===")
    top_features_ordered = [name for name, _ in fi_pairs[:10]]
    ablation_results = []

    for remove_feat in top_features_ordered:
        ablation_cols = [c for c in feature_cols if c != remove_feat]
        col_idxs = [feature_cols.index(c) for c in ablation_cols]
        X_tr_abl = X_train[:, col_idxs]
        X_te_abl = X_test[:, col_idxs]

        m_abl = xgb.XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=GLOBAL_SEED, n_jobs=-1,
        )
        m_abl.fit(X_tr_abl, y_train, verbose=False)
        y_abl = m_abl.predict(X_te_abl)
        r2_abl = r2_score(y_test, y_abl)
        drop = r2_xgb - r2_abl

        ablation_results.append({
            "removed_feature": remove_feat,
            "r2_without": round(r2_abl, 4),
            "r2_drop": round(drop, 4),
        })
        logger.info("  Remove %-30s -> R2=%.4f (drop=%.4f)", remove_feat, r2_abl, drop)

    # ── Compile results ──────────────────────────────────────────────
    results = {
        "experiment": "xgb_deep_features",
        "universe": "eu_50",
        "threshold": 0.6,
        "stride": 3,
        "n_samples": int(n),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "holdout": {
            "r2_xgb": round(r2_xgb, 4),
            "r2_persistence": round(r2_persist, 4),
            "r2_ar1": round(r2_ar1, 4),
            "rmse_xgb": round(rmse_xgb, 6),
            "rmse_persistence": round(rmse_persist, 6),
            "rmse_ar1": round(rmse_ar1, 6),
            "mae_xgb": round(mae_xgb, 6),
        },
        "builtin_importance_top30": [[name, round(float(imp), 6)] for name, imp in fi_pairs[:30]],
        "permutation_importance_top30": [[name, round(float(imp), 6)] for name, imp in perm_pairs[:30]],
        "shap": shap_results,
        "walk_forward_cv": {
            "n_folds": n_folds,
            "folds": wf_results,
            "mean_r2_xgb": round(mean_r2_xgb, 4),
            "mean_r2_persistence": round(mean_r2_persist, 4),
            "mean_r2_ar1": round(mean_r2_ar1, 4),
        },
        "diebold_mariano_vs_ar1": {
            "dm_statistic": round(dm_stat, 4),
            "p_value": round(dm_pval, 4),
            "significant_at_005": significant,
        },
        "ablation": ablation_results,
        "ar1_coefficients": {
            "intercept": round(float(coefs[0]), 6),
            "slope": round(float(coefs[1]), 6),
        },
    }

    return results


# ── Main ──────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    logger.info("=== XGBoost Deep Feature Importance Experiment ===")

    # Step 1: Fetch data
    logger.info("\n--- Step 1: Fetching eu_50 data ---")
    snapshots = fetch_snapshots()
    if not snapshots:
        logger.error("No snapshots available. Aborting.")
        return

    # Step 2: Compute features
    logger.info("\n--- Step 2: Computing exhaustive features ---")
    df = compute_features(snapshots)
    logger.info("Final feature matrix: %d samples x %d columns", len(df), len(df.columns))

    # Steps 4-8: Train, importance, SHAP, walk-forward, ablation
    logger.info("\n--- Steps 4-8: XGBoost analysis ---")
    results = run_analysis(df)

    # Step 9: Save
    out_path = RESULTS_DIR / "exp_xgb_deep_features.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)
    logger.info("\nSaved results to %s", out_path)

    elapsed = time.time() - t0
    logger.info("\n=== DONE (%.1fs) ===", elapsed)

    # ── Final summary print ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DEEP FEATURE IMPORTANCE RESULTS")
    print("=" * 70)
    print(f"\nUniverse: eu_50 | Threshold: 0.6 | Stride: 3")
    print(f"Samples: {results['n_samples']} | Features: {results['n_features']}")

    print(f"\n--- HOLD-OUT (80/20 split) ---")
    h = results["holdout"]
    print(f"  XGBoost     R2={h['r2_xgb']:.4f}  RMSE={h['rmse_xgb']:.6f}")
    print(f"  Persistence R2={h['r2_persistence']:.4f}  RMSE={h['rmse_persistence']:.6f}")
    print(f"  AR(1)       R2={h['r2_ar1']:.4f}  RMSE={h['rmse_ar1']:.6f}")

    print(f"\n--- TOP 15 FEATURES (built-in importance) ---")
    for i, (name, imp) in enumerate(results["builtin_importance_top30"][:15]):
        print(f"  {i+1:2d}. {name:<30s} {imp:.4f}")

    print(f"\n--- WALK-FORWARD CV ({results['walk_forward_cv']['n_folds']} folds) ---")
    wf = results["walk_forward_cv"]
    print(f"  Mean XGBoost R2:     {wf['mean_r2_xgb']:.4f}")
    print(f"  Mean Persistence R2: {wf['mean_r2_persistence']:.4f}")
    print(f"  Mean AR(1) R2:       {wf['mean_r2_ar1']:.4f}")

    dm = results["diebold_mariano_vs_ar1"]
    print(f"\n--- DIEBOLD-MARIANO: XGB vs AR(1) ---")
    print(f"  DM statistic: {dm['dm_statistic']:.3f}  p-value: {dm['p_value']:.4f}")
    print(f"  Significant at 5%? {'YES' if dm['significant_at_005'] else 'NO'}")

    print(f"\n--- ABLATION (top-10 feature removal) ---")
    for a in results["ablation"]:
        print(f"  Remove {a['removed_feature']:<30s} -> R2={a['r2_without']:.4f} (drop={a['r2_drop']:+.4f})")

    if results["shap"]["available"]:
        print(f"\n--- TOP 10 FEATURES (SHAP) ---")
        for i, (name, val) in enumerate(results["shap"]["top_features"][:10]):
            print(f"  {i+1:2d}. {name:<30s} {val:.6f}")

    beats = dm["significant_at_005"]
    print(f"\n{'=' * 70}")
    print(f"CONCLUSION: XGB with {results['n_features']} features {'SIGNIFICANTLY' if beats else 'does NOT significantly'}")
    print(f"  beat AR(1) (DM p={dm['p_value']:.4f})")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
