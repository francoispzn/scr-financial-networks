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
    logger.info("=== Experiment 2: GAT Training ===")
    from scr_financial.abm.simulation import BankingSystemSimulation
    from scr_financial.ml.gnn_predictor import GNNPredictor

    # Build simulation and generate snapshots
    rng = np.random.default_rng(123)
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
    network = {}
    for i, bid in enumerate(bank_names):
        network[bid] = {}
        for j, oid in enumerate(bank_names):
            if i != j:
                network[bid][oid] = rng.uniform(0.15, 0.65)

    sim = BankingSystemSimulation(bank_data, network, {"CISS": 0.3, "funding_stress": 0.1}, seed=123)

    # Generate 200 snapshots with occasional shocks for variance
    snapshots = []
    for step in range(200):
        sim.run_simulation(1)
        # Inject shocks every 40 steps for spectral variation
        if step > 0 and step % 40 == 0:
            target = bank_names[step % len(bank_names)]
            sim.apply_external_shock({target: {"CET1_ratio": -3.0, "LCR": -20.0}})
        snap = GNNPredictor.extract_graph_snapshot(sim)
        snapshots.append(snap)

    # Train GAT predictor
    predictor = GNNPredictor(seq_len=10, hidden_dim=64, num_gat_layers=3, heads=4, dropout=0.1)
    t0 = time.time()
    final_loss = predictor.train(snapshots, epochs=150, lr=3e-3)
    train_time = time.time() - t0

    results = {
        "n_snapshots": len(snapshots),
        "seq_len": predictor.seq_len,
        "hidden_dim": predictor.hidden_dim,
        "num_gat_layers": predictor.num_gat_layers,
        "heads": predictor.heads,
        "n_params": predictor.model.count_parameters() if predictor.model else 0,
        "epochs": 150,
        "final_train_loss": float(final_loss),
        "train_metrics": predictor.train_metrics,
        "test_metrics": predictor.test_metrics,
        "training_time_seconds": round(train_time, 1),
        "training_history": predictor.training_history[-10:],  # Last 10 checkpoints
    }

    # Prediction test
    preds = predictor.predict(snapshots, steps=20)
    results["prediction_sample"] = preds[:5]

    logger.info("GAT results: test_mse=%.6f, test_r2=%.4f, time=%.1fs, params=%d",
                predictor.test_metrics.get("mse", 0),
                predictor.test_metrics.get("r2", 0),
                train_time,
                results["n_params"])
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
    logger.info("=== Experiment 4: Backtesting ===")
    from scr_financial.abm.simulation import BankingSystemSimulation
    from scr_financial.network.spectral import compute_laplacian, eigendecomposition, find_spectral_gap

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
    network = {}
    for i, bid in enumerate(bank_names):
        network[bid] = {}
        for j, oid in enumerate(bank_names):
            if i != j:
                network[bid][oid] = rng.uniform(0.15, 0.65)

    sim = BankingSystemSimulation(bank_data, network, {"CISS": 0.3}, seed=42)

    # Generate 40 quarterly snapshots (10 years)
    quarterly_data = []
    for q in range(40):
        sim.run_simulation(5)
        # Periodic shocks
        if q in [8, 9, 20, 21, 30]:  # Crisis periods
            target = bank_names[q % len(bank_names)]
            sim.apply_external_shock({target: {"CET1_ratio": -5.0, "LCR": -30.0}})

        adj = sim.get_adjacency_matrix()
        adj_sym = (adj + adj.T) / 2.0
        L = compute_laplacian(adj_sym, normalized=True)
        eigenvalues, _ = eigendecomposition(L)
        gap_idx, gap_size = find_spectral_gap(eigenvalues)

        cet1s = [b.state.get("CET1_ratio", 10) for b in sim.banks.values()]
        n_stressed = sum(1 for c in cet1s if c < 8.0)

        quarterly_data.append({
            "quarter": q,
            "lambda_2": float(eigenvalues[1]) if len(eigenvalues) > 1 else 0,
            "spectral_gap": float(gap_size),
            "spectral_radius": float(eigenvalues[-1]),
            "scg_risk": float(1 - eigenvalues[1] / eigenvalues[-1]) if eigenvalues[-1] > 0 and len(eigenvalues) > 1 else 1.0,
            "avg_cet1": float(np.mean(cet1s)),
            "min_cet1": float(np.min(cet1s)),
            "n_stressed": n_stressed,
            "n_banks": len(bank_names),
            "basel_stress": float(n_stressed / len(bank_names)),
        })

    # Rolling backtest: does SCG risk at time t predict stress at t+1?
    window = 4
    correlations = {"scg_risk_vs_future_stress": [], "basel_vs_future_stress": []}
    for i in range(window, len(quarterly_data) - 1):
        current_scg = np.mean([quarterly_data[j]["scg_risk"] for j in range(i - window, i)])
        current_basel = np.mean([quarterly_data[j]["basel_stress"] for j in range(i - window, i)])
        future_stress = quarterly_data[i + 1]["n_stressed"]
        correlations["scg_risk_vs_future_stress"].append((current_scg, future_stress))
        correlations["basel_vs_future_stress"].append((current_basel, future_stress))

    # Compute Pearson correlations
    scg_pairs = np.array(correlations["scg_risk_vs_future_stress"])
    basel_pairs = np.array(correlations["basel_vs_future_stress"])

    def safe_corr(pairs):
        if len(pairs) < 3 or pairs[:, 0].std() < 1e-8 or pairs[:, 1].std() < 1e-8:
            return 0.0
        return float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])

    results = {
        "n_quarters": len(quarterly_data),
        "rolling_window": window,
        "scg_risk_correlation_to_future_stress": safe_corr(scg_pairs),
        "basel_stress_correlation_to_future_stress": safe_corr(basel_pairs),
        "quarterly_summary": quarterly_data[:5] + quarterly_data[-5:],  # First and last 5
        "crisis_quarters": [q["quarter"] for q in quarterly_data if q["n_stressed"] > 0],
    }

    logger.info("Backtest: SCG corr=%.4f, Basel corr=%.4f",
                results["scg_risk_correlation_to_future_stress"],
                results["basel_stress_correlation_to_future_stress"])
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
