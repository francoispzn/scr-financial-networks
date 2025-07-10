"""
Run all experiments for the paper and collect results as JSON.

Experiments:
  1. Spectral Coarse-Graining validation (preservation metrics)
  2. GAT temporal GNN training and evaluation
  3. ABM stress testing (1-channel vs 3-channel comparison)
  4. Backtesting comparison (SCG risk vs Basel/SREP indicators)

Output: experiments/results.json
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("experiments")

RESULTS = {}


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 1: Spectral Coarse-Graining Validation
# ──────────────────────────────────────────────────────────────────────────────

def run_scg_validation():
    logger.info("=== Experiment 1: SCG Validation ===")
    from scr_financial.abm.simulation import BankingSystemSimulation
    from scr_financial.network.spectral import compute_laplacian, eigendecomposition, find_spectral_gap
    from scr_financial.network.coarse_graining import SpectralCoarseGraining

    # Build a 10-bank system with realistic parameters
    rng = np.random.default_rng(42)
    bank_names = ["DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
                  "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA"]
    bank_data = {}
    for name in bank_names:
        bank_data[name] = {
            "CET1_ratio": rng.uniform(10, 16),
            "LCR": rng.uniform(110, 180),
            "NSFR": rng.uniform(100, 140),
            "total_assets": rng.uniform(5e11, 2.5e12),
            "cash": rng.uniform(2e10, 8e10),
            "interbank_assets": rng.uniform(5e9, 5e10),
            "interbank_liabilities": rng.uniform(5e9, 5e10),
        }

    # Correlation-based network
    network = {}
    for i, bid in enumerate(bank_names):
        network[bid] = {}
        for j, oid in enumerate(bank_names):
            if i != j:
                network[bid][oid] = rng.uniform(0.15, 0.65)

    sim = BankingSystemSimulation(bank_data, network, {"CISS": 0.3}, seed=42)
    sim.run_simulation(10)

    adj = sim.get_adjacency_matrix()
    adj_sym = (adj + adj.T) / 2.0

    # Full spectral analysis
    L = compute_laplacian(adj_sym, normalized=True)
    eigenvalues, eigenvectors = eigendecomposition(L)
    gap_idx, gap_size = find_spectral_gap(eigenvalues)

    # Coarse-grain
    scg = SpectralCoarseGraining.from_adjacency(adj_sym, bank_names)
    scg.coarse_grain()
    accuracy = scg.compute_reconstruction_accuracy(time_steps=15)

    # Coarse-grained eigenvalues
    import networkx as nx
    try:
        cg_graph = scg.create_coarse_grained_graph()
        cg_adj = nx.to_numpy_array(cg_graph)
        cg_adj_sym = (cg_adj + cg_adj.T) / 2.0
        if cg_adj_sym.shape[0] > 1:
            L_cg = compute_laplacian(cg_adj_sym, normalized=True)
            ev_cg, _ = eigendecomposition(L_cg)
        else:
            ev_cg = []
    except Exception as e:
        logger.warning("Could not create coarse-grained graph: %s", e)
        ev_cg = []
        cg_adj_sym = np.array([])

    n_clusters = len(set(scg.clusters)) if scg.clusters is not None else None
    cg_error = scg.compute_coarse_graining_error()

    results = {
        "n_banks": len(bank_names),
        "n_clusters": n_clusters,
        "coarse_graining_error": float(cg_error) if cg_error is not None else None,
        "eigenvalues_original": [float(e) for e in eigenvalues],
        "eigenvalues_coarsened": [float(e) for e in ev_cg] if len(ev_cg) > 0 else [],
        "spectral_gap_index": int(gap_idx),
        "spectral_gap_size": float(gap_size),
        "algebraic_connectivity": float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0,
        "spectral_radius": float(eigenvalues[-1]),
        "reconstruction_r2": accuracy.get("r2", None),
        "reconstruction_rmse": accuracy.get("rmse", None),
        "density_original": float(np.count_nonzero(adj_sym) / (adj_sym.shape[0] * (adj_sym.shape[0] - 1))),
    }
    logger.info("SCG results: gap=%.4f, R2=%s, clusters=%s",
                results["spectral_gap_size"],
                results["reconstruction_r2"],
                results["n_clusters"])
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 2: GAT Training and Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_gat_training():
    """Train GAT on real market data (yfinance) with dynamic adjacency.

    Uses build_daily_graph_snapshots() which produces ~200+ snapshots from
    3 years of daily stock data with rolling correlation adjacency — giving
    genuine temporal variation in both node features and spectral targets.

    Runs N_SEEDS independent training runs with different seeds to compute
    confidence intervals on test metrics.
    """
    logger.info("=== Experiment 2: GAT Training (yfinance data) ===")
    from dashboard.data_api import build_daily_graph_snapshots
    from scr_financial.ml.gnn_predictor import GNNPredictor

    # Fetch real market data: 3 years, weekly stride for ~200 snapshots
    logger.info("Fetching yfinance data (3 years, stride=5)...")
    snapshots = build_daily_graph_snapshots(
        lookback_years=3,
        corr_window=60,
        min_corr=0.3,
        stride=5,  # Weekly sampling
    )
    logger.info("Got %d snapshots from real market data", len(snapshots))

    # Check spectral target variance (the root cause of previous R²=0)
    targets_arr = np.array([[s["targets"][k] for k in ["lambda_2", "spectral_gap", "spectral_radius"]]
                            for s in snapshots])
    logger.info("Target variance: lambda_2=%.6f, gap=%.6f, radius=%.6f",
                np.var(targets_arr[:, 0]), np.var(targets_arr[:, 1]), np.var(targets_arr[:, 2]))

    # Multi-seed training for statistical robustness
    N_SEEDS = 5
    all_runs = []

    for seed_i in range(N_SEEDS):
        np.random.seed(seed_i * 42)
        import torch
        torch.manual_seed(seed_i * 42)

        # Use smaller model to avoid overparameterization
        # ~200 snapshots → ~190 sequences → hidden=32 keeps params reasonable
        predictor = GNNPredictor(
            seq_len=10, hidden_dim=32, num_gat_layers=2, heads=4, dropout=0.2,
        )
        t0 = time.time()
        final_loss = predictor.train(
            snapshots, epochs=200, lr=3e-3, patience=30,
        )
        train_time = time.time() - t0

        run_result = {
            "seed": seed_i,
            "n_params": predictor.model.count_parameters() if predictor.model else 0,
            "final_train_loss": float(final_loss),
            "train_mse": predictor.train_metrics.get("mse", 0),
            "train_r2": predictor.train_metrics.get("r2", 0),
            "test_mse": predictor.test_metrics.get("mse", 0),
            "test_r2": predictor.test_metrics.get("r2", 0),
            "test_r2_per_target": predictor.test_metrics.get("r2_per_target", {}),
            "training_time_seconds": round(train_time, 1),
            "epochs_trained": predictor.training_history[-1]["epoch"] if predictor.training_history else 0,
        }
        all_runs.append(run_result)
        logger.info("  Seed %d: test_r2=%.4f, test_mse=%.6f, epochs=%d, time=%.1fs",
                     seed_i, run_result["test_r2"], run_result["test_mse"],
                     run_result["epochs_trained"], train_time)

    # Aggregate across seeds
    test_r2s = [r["test_r2"] for r in all_runs]
    test_mses = [r["test_mse"] for r in all_runs]

    results = {
        "data_source": "yfinance (3yr, stride=5)",
        "n_snapshots": len(snapshots),
        "target_variance": {
            "lambda_2": float(np.var(targets_arr[:, 0])),
            "spectral_gap": float(np.var(targets_arr[:, 1])),
            "spectral_radius": float(np.var(targets_arr[:, 2])),
        },
        "model_config": {
            "seq_len": 10, "hidden_dim": 32, "num_gat_layers": 2,
            "heads": 4, "dropout": 0.2, "epochs": 200, "patience": 30,
        },
        "n_seeds": N_SEEDS,
        "per_seed_results": all_runs,
        "aggregate": {
            "test_r2_mean": float(np.mean(test_r2s)),
            "test_r2_std": float(np.std(test_r2s)),
            "test_r2_95ci": [float(np.percentile(test_r2s, 2.5)), float(np.percentile(test_r2s, 97.5))],
            "test_mse_mean": float(np.mean(test_mses)),
            "test_mse_std": float(np.std(test_mses)),
            "n_params": all_runs[0]["n_params"],
        },
    }

    logger.info("GAT aggregate: test_r2=%.4f +/- %.4f, test_mse=%.6f +/- %.6f",
                results["aggregate"]["test_r2_mean"], results["aggregate"]["test_r2_std"],
                results["aggregate"]["test_mse_mean"], results["aggregate"]["test_mse_std"])
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 3: ABM Stress Testing (1-channel vs 3-channel)
# ──────────────────────────────────────────────────────────────────────────────

def run_abm_stress_test():
    logger.info("=== Experiment 3: ABM Stress Testing ===")
    from scr_financial.abm.simulation import BankingSystemSimulation

    bank_names = ["DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
                  "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA"]

    def build_sim(seed):
        rng = np.random.default_rng(seed)
        bank_data = {}
        for name in bank_names:
            bank_data[name] = {
                "CET1_ratio": rng.uniform(10, 16),
                "LCR": rng.uniform(110, 180),
                "NSFR": rng.uniform(100, 140),
                "total_assets": rng.uniform(5e11, 2.5e12),
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
        return BankingSystemSimulation(bank_data, network, {"CISS": 0.5, "funding_stress": 0.3}, seed=seed)

    # Stress scenarios
    scenarios = {
        "liquidity_crisis": {"DE_DBK": {"CET1_ratio": -5.0, "LCR": -80.0, "cash": -3e10}},
        "capital_shock": {"FR_BNP": {"CET1_ratio": -12.0}, "IT_UCG": {"CET1_ratio": -10.0}},
        "market_stress": {bid: {"CET1_ratio": -3.0, "LCR": -15.0} for bid in bank_names[:5]},
    }

    n_runs = 50
    results_by_scenario = {}

    for scenario_name, shock in scenarios.items():
        logger.info("  Scenario: %s (%d runs)", scenario_name, n_runs)
        defaults_3ch = []
        defaults_1ch = []
        final_cet1s_3ch = []
        final_cet1s_1ch = []

        for run_i in range(n_runs):
            seed = 1000 + run_i

            # 3-channel run
            sim3 = build_sim(seed)
            sim3.run_simulation(5)
            sim3.apply_external_shock(shock)
            sim3.run_simulation(30)
            n_def_3 = sum(1 for b in sim3.banks.values() if b._defaulted)
            avg_cet1_3 = np.mean([b.state.get("CET1_ratio", 0) for b in sim3.banks.values()])
            defaults_3ch.append(n_def_3)
            final_cet1s_3ch.append(float(avg_cet1_3))

            # 1-channel run (disable channels 2 and 3 temporarily)
            sim1 = build_sim(seed)
            # Monkey-patch to skip channels 2 and 3
            sim1._propagate_funding_stress = lambda: None
            sim1._propagate_fire_sales = lambda: None
            sim1.run_simulation(5)
            sim1.apply_external_shock(shock)
            sim1.run_simulation(30)
            n_def_1 = sum(1 for b in sim1.banks.values() if b._defaulted)
            avg_cet1_1 = np.mean([b.state.get("CET1_ratio", 0) for b in sim1.banks.values()])
            defaults_1ch.append(n_def_1)
            final_cet1s_1ch.append(float(avg_cet1_1))

        results_by_scenario[scenario_name] = {
            "n_runs": n_runs,
            "three_channel": {
                "mean_defaults": float(np.mean(defaults_3ch)),
                "std_defaults": float(np.std(defaults_3ch)),
                "max_defaults": int(np.max(defaults_3ch)),
                "mean_final_cet1": float(np.mean(final_cet1s_3ch)),
                "default_rate": float(np.mean([d > 0 for d in defaults_3ch])),
            },
            "one_channel": {
                "mean_defaults": float(np.mean(defaults_1ch)),
                "std_defaults": float(np.std(defaults_1ch)),
                "max_defaults": int(np.max(defaults_1ch)),
                "mean_final_cet1": float(np.mean(final_cet1s_1ch)),
                "default_rate": float(np.mean([d > 0 for d in defaults_1ch])),
            },
        }

    logger.info("ABM stress test complete.")
    return results_by_scenario


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 4: Backtesting Comparison
# ──────────────────────────────────────────────────────────────────────────────

def run_backtesting():
    """Backtesting using real market data with dynamic adjacency.

    Uses yfinance snapshots (monthly sampling) so that spectral properties
    genuinely evolve over time. Compares SCG risk indicator vs simple
    volatility-based stress measures for predicting future realized volatility.
    """
    logger.info("=== Experiment 4: Backtesting (real market data) ===")
    from dashboard.data_api import build_daily_graph_snapshots
    from scr_financial.network.spectral import compute_laplacian, eigendecomposition, find_spectral_gap

    # Monthly snapshots for backtesting (~36 months from 3 years)
    logger.info("Fetching yfinance data for backtesting (3yr, monthly stride)...")
    snapshots = build_daily_graph_snapshots(
        lookback_years=3,
        corr_window=60,
        min_corr=0.3,
        stride=21,  # ~monthly (21 trading days)
    )
    logger.info("Got %d monthly snapshots", len(snapshots))

    # Compute SCG risk and volatility stress for each snapshot
    monthly_data = []
    for i, snap in enumerate(snapshots):
        targets = snap["targets"]
        lam2 = targets["lambda_2"]
        rho = targets["spectral_radius"]
        scg_risk = float(1.0 - lam2 / rho) if rho > 1e-8 else 1.0

        # Volatility-based stress: average node volatility feature
        vol = float(np.mean(snap["node_features"][:, 0]))  # Feature 0 = volatility_30d

        monthly_data.append({
            "month": i,
            "date": snap.get("date", f"month_{i}"),
            "lambda_2": float(lam2),
            "spectral_gap": float(targets["spectral_gap"]),
            "spectral_radius": float(rho),
            "scg_risk": scg_risk,
            "avg_volatility": vol,
            "n_edges": snap["edge_index"].shape[1] // 2 if snap["edge_index"].shape[1] > 0 else 0,
        })

    # Rolling backtest: does SCG risk at time t predict future volatility at t+h?
    window = 3  # 3-month lookback
    horizon = 1  # 1-month ahead prediction

    def safe_corr(x, y):
        if len(x) < 5 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    scg_risk_series = np.array([d["scg_risk"] for d in monthly_data])
    vol_series = np.array([d["avg_volatility"] for d in monthly_data])
    lam2_series = np.array([d["lambda_2"] for d in monthly_data])
    gap_series = np.array([d["spectral_gap"] for d in monthly_data])

    # Predictive correlations: current indicator → future volatility
    n = len(monthly_data)
    pairs_scg, pairs_vol, pairs_lam2 = [], [], []
    for t in range(window, n - horizon):
        future_vol = vol_series[t + horizon]
        # Rolling average of current indicators
        current_scg = np.mean(scg_risk_series[t - window:t])
        current_vol = np.mean(vol_series[t - window:t])
        current_lam2 = np.mean(lam2_series[t - window:t])
        pairs_scg.append((current_scg, future_vol))
        pairs_vol.append((current_vol, future_vol))
        pairs_lam2.append((current_lam2, future_vol))

    pairs_scg = np.array(pairs_scg)
    pairs_vol = np.array(pairs_vol)
    pairs_lam2 = np.array(pairs_lam2)

    results = {
        "data_source": "yfinance (3yr, monthly stride)",
        "n_months": len(monthly_data),
        "rolling_window": window,
        "prediction_horizon": horizon,
        "scg_risk_stats": {
            "mean": float(np.mean(scg_risk_series)),
            "std": float(np.std(scg_risk_series)),
            "min": float(np.min(scg_risk_series)),
            "max": float(np.max(scg_risk_series)),
        },
        "lambda_2_stats": {
            "mean": float(np.mean(lam2_series)),
            "std": float(np.std(lam2_series)),
        },
        "predictive_correlations": {
            "scg_risk_to_future_vol": safe_corr(pairs_scg[:, 0], pairs_scg[:, 1]),
            "current_vol_to_future_vol": safe_corr(pairs_vol[:, 0], pairs_vol[:, 1]),
            "lambda2_to_future_vol": safe_corr(pairs_lam2[:, 0], pairs_lam2[:, 1]),
        },
        "monthly_sample": monthly_data[:5] + monthly_data[-5:],
    }

    logger.info("Backtest: SCG→vol=%.4f, vol→vol=%.4f, λ₂→vol=%.4f",
                results["predictive_correlations"]["scg_risk_to_future_vol"],
                results["predictive_correlations"]["current_vol_to_future_vol"],
                results["predictive_correlations"]["lambda2_to_future_vol"])
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()

    RESULTS["scg_validation"] = run_scg_validation()
    RESULTS["gat_training"] = run_gat_training()
    RESULTS["abm_stress_test"] = run_abm_stress_test()
    RESULTS["backtesting"] = run_backtesting()

    RESULTS["total_time_seconds"] = round(time.time() - t_start, 1)

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)

    logger.info("All experiments complete in %.1fs. Results saved to %s",
                RESULTS["total_time_seconds"], out_path)
