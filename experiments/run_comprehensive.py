#!/usr/bin/env python3
"""
Comprehensive experiment suite for the SCR Financial Networks paper.

Runs multiple experiment configurations with proper statistical methodology:
  1. SCG Validation — preservation metrics across different network densities
  2. GAT Hyperparameter Sweep — hidden_dim, heads, layers, dropout
  3. ABM Multi-Scenario Stress Testing — varied shock magnitudes, 3ch vs 1ch
  4. Backtesting — rolling windows, multiple correlation thresholds
  5. Figure Generation — training curves, spectral evolution, attention heatmaps

All results saved to experiments/results/ as JSON + PNG figures.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("experiments")

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Publication-quality matplotlib defaults
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ══════════════════════════════════════════════════════════════════════
# Experiment 1: SCG Validation across network configurations
# ══════════════════════════════════════════════════════════════════════

def run_scg_experiments():
    logger.info("═══ Experiment 1: SCG Validation ═══")
    from scr_financial.network.spectral import compute_laplacian, eigendecomposition, find_spectral_gap
    from scr_financial.network.coarse_graining import SpectralCoarseGraining


    results = []
    rng = np.random.default_rng(42)

    # Test across different network configurations
    configs = [
        {"name": "dense_uniform", "n": 10, "low": 0.3, "high": 0.7, "desc": "Dense uniform (corr 0.3-0.7)"},
        {"name": "sparse_threshold", "n": 10, "low": 0.0, "high": 0.5, "threshold": 0.2, "desc": "Sparse (threshold 0.2)"},
        {"name": "clustered_3", "n": 12, "clusters": 3, "intra": 0.8, "inter": 0.1, "desc": "3 clusters (intra=0.8, inter=0.1)"},
        {"name": "clustered_2", "n": 10, "clusters": 2, "intra": 0.7, "inter": 0.15, "desc": "2 clusters (intra=0.7, inter=0.15)"},
        {"name": "star_topology", "n": 10, "star": True, "desc": "Star topology (1 hub)"},
    ]

    for cfg in configs:
        logger.info("  Config: %s", cfg["desc"])
        n = cfg["n"]

        if cfg.get("clusters"):
            # Block structure
            k = cfg["clusters"]
            size = n // k
            adj = np.zeros((n, n))
            for c in range(k):
                for i in range(size):
                    for j in range(size):
                        if i != j:
                            adj[c*size+i, c*size+j] = cfg["intra"] + rng.uniform(-0.1, 0.1)
            for c1 in range(k):
                for c2 in range(c1+1, k):
                    for i in range(size):
                        for j in range(size):
                            w = cfg["inter"] + rng.uniform(-0.05, 0.05)
                            if w > 0:
                                adj[c1*size+i, c2*size+j] = w
                                adj[c2*size+j, c1*size+i] = w
        elif cfg.get("star"):
            adj = np.zeros((n, n))
            for i in range(1, n):
                w = rng.uniform(0.3, 0.8)
                adj[0, i] = w
                adj[i, 0] = w
                # Some peripheral connections
                if rng.random() < 0.3:
                    j = rng.integers(1, n)
                    if j != i:
                        adj[i, j] = rng.uniform(0.05, 0.2)
                        adj[j, i] = adj[i, j]
        else:
            adj = rng.uniform(cfg["low"], cfg["high"], (n, n))
            adj = (adj + adj.T) / 2
            np.fill_diagonal(adj, 0)
            if cfg.get("threshold"):
                adj[adj < cfg["threshold"]] = 0

        # Ensure symmetric
        adj = (adj + adj.T) / 2.0
        np.fill_diagonal(adj, 0)

        # Run SCG
        bank_ids = [f"bank_{i}" for i in range(n)]
        scg = SpectralCoarseGraining.from_adjacency(adj, bank_ids)
        L_cg = scg.coarse_grain()
        scg.identify_clusters()

        # Spectral analysis (combinatorial Laplacian for Schmidt compatibility)
        L = compute_laplacian(adj, normalized=False)
        eigenvalues, eigenvectors = eigendecomposition(L)
        gap_idx, gap_size = find_spectral_gap(eigenvalues, adjacency_matrix=adj)

        # Reconstruction accuracy
        acc = scg.compute_reconstruction_accuracy(time_steps=20)
        diffusion_errors = scg.compare_diffusion_dynamics(time_steps=20)

        n_clusters = len(set(scg.clusters))
        density = np.count_nonzero(adj) / (n * (n - 1)) if n > 1 else 0

        result = {
            "config": cfg["name"],
            "description": cfg["desc"],
            "n_nodes": n,
            "n_clusters": n_clusters,
            "density": round(density, 3),
            "eigenvalues": [round(float(e), 4) for e in eigenvalues],
            "gap_index": int(gap_idx),
            "gap_size": round(float(gap_size), 4),
            "lambda_2": round(float(eigenvalues[1]), 4) if len(eigenvalues) > 1 else 0,
            "spectral_radius": round(float(eigenvalues[-1]), 4),
            "scg_risk": round(float(1 - eigenvalues[1]/eigenvalues[-1]), 4) if eigenvalues[-1] > 0 and len(eigenvalues) > 1 else 1.0,
            "reconstruction_r2_mean": round(float(np.mean(acc["r2"])), 4),
            "reconstruction_r2_min": round(float(np.min(acc["r2"])), 4),
            "reconstruction_rmse_mean": round(float(np.mean(acc["rmse"])), 6),
            "diffusion_error_mean": round(float(np.mean(diffusion_errors)), 6),
            "diffusion_error_max": round(float(np.max(diffusion_errors)), 6),
        }
        results.append(result)
        logger.info("    clusters=%d, gap=%.4f, R²=%.4f, density=%.3f",
                     n_clusters, gap_size, result["reconstruction_r2_mean"], density)

    # ── Figure: Eigenvalue spectra comparison ──
    fig, axes = plt.subplots(1, len(configs), figsize=(4*len(configs), 4), sharey=False)
    if len(configs) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        evs = r["eigenvalues"]
        ax.bar(range(len(evs)), evs, color="steelblue", alpha=0.8)
        ax.axhline(y=evs[r["gap_index"]], color="red", linestyle="--", alpha=0.5, label=f"gap at k={r['gap_index']}")
        ax.set_title(r["config"], fontsize=9)
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("λ")
        ax.legend(fontsize=7)
    plt.suptitle("Laplacian Eigenvalue Spectra Across Network Configurations", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "scg_eigenspectra.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure: Reconstruction accuracy ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        ax.bar(r["config"], r["reconstruction_r2_mean"], alpha=0.8, label=r["config"])
    ax.set_ylabel("Mean Reconstruction R²")
    ax.set_title("SCG Diffusion Reconstruction Accuracy")
    ax.set_ylim(0, 1.05)
    for i, r in enumerate(results):
        ax.text(i, r["reconstruction_r2_mean"] + 0.02, f'{r["reconstruction_r2_mean"]:.3f}', ha="center", fontsize=8)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "scg_reconstruction.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(RESULTS_DIR / "scg_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ══════════════════════════════════════════════════════════════════════
# Experiment 2: GAT Hyperparameter Sweep
# ══════════════════════════════════════════════════════════════════════

def run_gat_experiments():
    logger.info("═══ Experiment 2: GAT Hyperparameter Sweep ═══")

    import torch

    from dashboard.data_api import build_daily_graph_snapshots
    from scr_financial.ml.gnn_predictor import GNNPredictor

    # Fetch data once (expensive) — retry on transient yfinance failures
    logger.info("  Fetching yfinance data...")
    snapshots = []
    for attempt in range(3):
        try:
            snapshots = build_daily_graph_snapshots(lookback_years=3, corr_window=60, min_corr=0.3, stride=5)
            if len(snapshots) > 50:
                break
            logger.warning("  Only got %d snapshots (attempt %d), retrying...", len(snapshots), attempt+1)
            time.sleep(5)
        except Exception as e:
            logger.warning("  yfinance fetch failed (attempt %d): %s", attempt+1, e)
            time.sleep(5)
    if len(snapshots) < 50:
        logger.error("  Could not fetch sufficient data from yfinance. Skipping GAT experiments.")
        return {"error": "insufficient data", "n_snapshots": len(snapshots)}
    logger.info("  Got %d snapshots", len(snapshots))

    # Target variance check
    targets = np.array([[s["targets"][k] for k in ["lambda_2", "spectral_gap", "spectral_radius"]] for s in snapshots])
    target_stats = {
        "n_snapshots": len(snapshots),
        "lambda_2": {"mean": float(np.mean(targets[:,0])), "std": float(np.std(targets[:,0]))},
        "spectral_gap": {"mean": float(np.mean(targets[:,1])), "std": float(np.std(targets[:,1]))},
        "spectral_radius": {"mean": float(np.mean(targets[:,2])), "std": float(np.std(targets[:,2]))},
    }

    # Hyperparameter configurations to sweep
    configs = [
        {"name": "small", "hidden_dim": 16, "num_gat_layers": 2, "heads": 2, "dropout": 0.2},
        {"name": "medium", "hidden_dim": 32, "num_gat_layers": 2, "heads": 4, "dropout": 0.2},
        {"name": "large", "hidden_dim": 64, "num_gat_layers": 3, "heads": 4, "dropout": 0.1},
        {"name": "deep", "hidden_dim": 32, "num_gat_layers": 4, "heads": 4, "dropout": 0.3},
        {"name": "wide", "hidden_dim": 64, "num_gat_layers": 2, "heads": 8, "dropout": 0.2},
        {"name": "regularized", "hidden_dim": 32, "num_gat_layers": 2, "heads": 4, "dropout": 0.4},
    ]

    N_SEEDS = 5
    all_results = []

    for cfg in configs:
        logger.info("  Config: %s (h=%d, L=%d, H=%d, d=%.1f)",
                     cfg["name"], cfg["hidden_dim"], cfg["num_gat_layers"], cfg["heads"], cfg["dropout"])
        seed_results = []
        training_histories = []

        for seed in range(N_SEEDS):
            np.random.seed(seed * 42)
            torch.manual_seed(seed * 42)

            predictor = GNNPredictor(
                seq_len=10, hidden_dim=cfg["hidden_dim"],
                num_gat_layers=cfg["num_gat_layers"], heads=cfg["heads"],
                dropout=cfg["dropout"],
            )
            t0 = time.time()
            predictor.train(snapshots, epochs=300, lr=3e-3, patience=40)
            train_time = time.time() - t0

            seed_results.append({
                "seed": seed,
                "train_r2": predictor.train_metrics.get("r2", 0),
                "test_r2": predictor.test_metrics.get("r2", 0),
                "test_mse": predictor.test_metrics.get("mse", 0),
                "test_r2_per_target": predictor.test_metrics.get("r2_per_target", {}),
                "n_params": predictor.model.count_parameters() if predictor.model else 0,
                "epochs_trained": predictor.training_history[-1]["epoch"] if predictor.training_history else 0,
                "training_time_s": round(train_time, 1),
            })
            if seed == 0:
                training_histories = predictor.training_history

        test_r2s = [r["test_r2"] for r in seed_results]
        test_mses = [r["test_mse"] for r in seed_results]

        result = {
            "config": cfg,
            "n_seeds": N_SEEDS,
            "per_seed": seed_results,
            "aggregate": {
                "test_r2_mean": round(float(np.mean(test_r2s)), 4),
                "test_r2_std": round(float(np.std(test_r2s)), 4),
                "test_r2_median": round(float(np.median(test_r2s)), 4),
                "test_r2_95ci": [round(float(np.percentile(test_r2s, 2.5)), 4),
                                  round(float(np.percentile(test_r2s, 97.5)), 4)],
                "test_mse_mean": round(float(np.mean(test_mses)), 6),
                "test_mse_std": round(float(np.std(test_mses)), 6),
                "n_params": seed_results[0]["n_params"],
                "avg_epochs": round(float(np.mean([r["epochs_trained"] for r in seed_results])), 0),
            },
            "training_history_seed0": training_histories,
        }
        all_results.append(result)
        logger.info("    R²=%.4f±%.4f, MSE=%.6f, params=%d, epochs=%.0f",
                     result["aggregate"]["test_r2_mean"], result["aggregate"]["test_r2_std"],
                     result["aggregate"]["test_mse_mean"], result["aggregate"]["n_params"],
                     result["aggregate"]["avg_epochs"])

    # ── Figure: Hyperparameter comparison bar chart ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    names = [r["config"]["name"] for r in all_results]
    r2_means = [r["aggregate"]["test_r2_mean"] for r in all_results]
    r2_stds = [r["aggregate"]["test_r2_std"] for r in all_results]
    params = [r["aggregate"]["n_params"] for r in all_results]

    bars = ax1.bar(names, r2_means, yerr=r2_stds, capsize=5, alpha=0.8, color="steelblue")
    ax1.set_ylabel("Test R²")
    ax1.set_title("GAT Test R² by Configuration (5 seeds)")
    ax1.set_ylim(-0.1, max(r2_means) + 0.15)
    for bar, mean in zip(bars, r2_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{mean:.3f}", ha="center", fontsize=8)

    ax2.bar(names, params, alpha=0.8, color="coral")
    ax2.set_ylabel("Number of Parameters")
    ax2.set_title("Model Complexity")
    for i, p in enumerate(params):
        ax2.text(i, p + 500, f"{p:,}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gat_hyperparam_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure: Training curves (best config) ──
    best_idx = int(np.argmax(r2_means))
    best = all_results[best_idx]
    if best["training_history_seed0"]:
        hist = best["training_history_seed0"]
        epochs = [h["epoch"] for h in hist]
        train_losses = [h["train_loss"] for h in hist]
        test_losses = [h["test_loss"] for h in hist if h["test_loss"] is not None]
        test_epochs = [h["epoch"] for h in hist if h["test_loss"] is not None]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_losses, label="Train Loss", color="steelblue")
        if test_losses:
            ax.plot(test_epochs, test_losses, label="Test Loss", color="coral")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss (normalized)")
        ax.set_title(f"Training Curve — Best Config: {best['config']['name']}")
        ax.legend()
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "gat_training_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    output = {"target_stats": target_stats, "configs": all_results}
    with open(RESULTS_DIR / "gat_sweep.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    return output


# ══════════════════════════════════════════════════════════════════════
# Experiment 3: ABM Multi-Scenario Stress Testing
# ══════════════════════════════════════════════════════════════════════

def run_abm_experiments():
    logger.info("═══ Experiment 3: ABM Stress Testing ═══")


    from scr_financial.abm.simulation import BankingSystemSimulation

    bank_names = ["DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
                  "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA"]

    def build_sim(seed):
        rng = np.random.default_rng(seed)
        bank_data = {}
        for name in bank_names:
            bank_data[name] = {
                "CET1_ratio": rng.uniform(10, 16), "LCR": rng.uniform(110, 180),
                "NSFR": rng.uniform(100, 140), "total_assets": rng.uniform(5e11, 2.5e12),
                "cash": rng.uniform(2e10, 8e10), "interbank_assets": rng.uniform(5e9, 5e10),
                "interbank_liabilities": rng.uniform(5e9, 5e10),
            }
        network = {}
        for i, bid in enumerate(bank_names):
            network[bid] = {}
            for j, oid in enumerate(bank_names):
                if i != j:
                    network[bid][oid] = rng.uniform(0.15, 0.65)
        return BankingSystemSimulation(bank_data, network, {"CISS": 0.5, "funding_stress": 0.3}, seed=seed)

    # Scenarios with varying severity
    scenarios = {
        "mild_liquidity": {"DE_DBK": {"LCR": -30.0, "cash": -1e10}},
        "moderate_liquidity": {"DE_DBK": {"CET1_ratio": -3.0, "LCR": -60.0, "cash": -2e10}},
        "severe_liquidity": {"DE_DBK": {"CET1_ratio": -5.0, "LCR": -80.0, "cash": -3e10}},
        "single_capital_shock": {"FR_BNP": {"CET1_ratio": -12.0}},
        "dual_capital_shock": {"FR_BNP": {"CET1_ratio": -12.0}, "IT_UCG": {"CET1_ratio": -10.0}},
        "broad_market_stress": {bid: {"CET1_ratio": -3.0, "LCR": -15.0} for bid in bank_names[:5]},
        "systemic_crisis": {bid: {"CET1_ratio": -5.0, "LCR": -40.0, "cash": -1.5e10} for bid in bank_names[:7]},
    }

    N_RUNS = 100
    all_results = {}

    for scenario_name, shock in scenarios.items():
        logger.info("  Scenario: %s (%d runs)", scenario_name, N_RUNS)
        data_3ch = {"defaults": [], "final_cet1": [], "min_cet1": [], "cascade_depth": []}
        data_1ch = {"defaults": [], "final_cet1": [], "min_cet1": [], "cascade_depth": []}

        for run_i in range(N_RUNS):
            seed = 1000 + run_i

            for mode, data in [("3ch", data_3ch), ("1ch", data_1ch)]:
                sim = build_sim(seed)
                if mode == "1ch":
                    sim._propagate_funding_stress = lambda: None
                    sim._propagate_fire_sales = lambda: None

                sim.run_simulation(5)
                sim.apply_external_shock(shock)
                sim.run_simulation(30)

                n_def = sum(1 for b in sim.banks.values() if b._defaulted)
                cet1s = [b.state.get("CET1_ratio", 0) for b in sim.banks.values()]
                data["defaults"].append(n_def)
                data["final_cet1"].append(float(np.mean(cet1s)))
                data["min_cet1"].append(float(np.min(cet1s)))

        all_results[scenario_name] = {
            "n_runs": N_RUNS,
            "three_channel": {
                "mean_defaults": round(float(np.mean(data_3ch["defaults"])), 2),
                "std_defaults": round(float(np.std(data_3ch["defaults"])), 2),
                "max_defaults": int(np.max(data_3ch["defaults"])),
                "default_rate": round(float(np.mean([d > 0 for d in data_3ch["defaults"]])), 3),
                "mean_final_cet1": round(float(np.mean(data_3ch["final_cet1"])), 2),
                "mean_min_cet1": round(float(np.mean(data_3ch["min_cet1"])), 2),
            },
            "one_channel": {
                "mean_defaults": round(float(np.mean(data_1ch["defaults"])), 2),
                "std_defaults": round(float(np.std(data_1ch["defaults"])), 2),
                "max_defaults": int(np.max(data_1ch["defaults"])),
                "default_rate": round(float(np.mean([d > 0 for d in data_1ch["defaults"]])), 3),
                "mean_final_cet1": round(float(np.mean(data_1ch["final_cet1"])), 2),
                "mean_min_cet1": round(float(np.mean(data_1ch["min_cet1"])), 2),
            },
            "amplification_ratio": round(
                float(np.mean(data_3ch["defaults"])) / max(float(np.mean(data_1ch["defaults"])), 0.01), 2
            ),
        }
        logger.info("    3ch: %.1f defaults (rate=%.1f%%), 1ch: %.1f defaults, amplification=%.1fx",
                     all_results[scenario_name]["three_channel"]["mean_defaults"],
                     all_results[scenario_name]["three_channel"]["default_rate"] * 100,
                     all_results[scenario_name]["one_channel"]["mean_defaults"],
                     all_results[scenario_name]["amplification_ratio"])

    # ── Figure: Default comparison across scenarios ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    scenario_names = list(all_results.keys())
    x = np.arange(len(scenario_names))
    w = 0.35

    d3 = [all_results[s]["three_channel"]["mean_defaults"] for s in scenario_names]
    d1 = [all_results[s]["one_channel"]["mean_defaults"] for s in scenario_names]
    ax1.bar(x - w/2, d3, w, label="3-Channel", color="coral", alpha=0.8)
    ax1.bar(x + w/2, d1, w, label="1-Channel", color="steelblue", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace("_", "\n") for s in scenario_names], fontsize=7)
    ax1.set_ylabel("Mean Defaults")
    ax1.set_title("Default Count: 3-Channel vs 1-Channel ABM")
    ax1.legend()

    cet1_3 = [all_results[s]["three_channel"]["mean_min_cet1"] for s in scenario_names]
    cet1_1 = [all_results[s]["one_channel"]["mean_min_cet1"] for s in scenario_names]
    ax2.bar(x - w/2, cet1_3, w, label="3-Channel", color="coral", alpha=0.8)
    ax2.bar(x + w/2, cet1_1, w, label="1-Channel", color="steelblue", alpha=0.8)
    ax2.axhline(y=4.5, color="red", linestyle="--", alpha=0.5, label="CET1 min (4.5%)")
    ax2.axhline(y=8.0, color="orange", linestyle="--", alpha=0.5, label="CET1 stress (8%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace("_", "\n") for s in scenario_names], fontsize=7)
    ax2.set_ylabel("Mean Min CET1 (%)")
    ax2.set_title("Minimum CET1 Ratio Under Stress")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "abm_stress_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(RESULTS_DIR / "abm_stress.json", "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


# ══════════════════════════════════════════════════════════════════════
# Experiment 4: Backtesting with multiple configurations
# ══════════════════════════════════════════════════════════════════════

def run_backtesting_experiments():
    logger.info("═══ Experiment 4: Backtesting ═══")


    from dashboard.data_api import build_daily_graph_snapshots

    # Multiple correlation thresholds
    thresholds = [0.2, 0.3, 0.4, 0.5]
    all_results = {}

    for min_corr in thresholds:
        logger.info("  Correlation threshold: %.1f", min_corr)
        snapshots = []
        for attempt in range(3):
            try:
                snapshots = build_daily_graph_snapshots(
                    lookback_years=3, corr_window=60, min_corr=min_corr, stride=10,
                )
                if len(snapshots) > 20:
                    break
                time.sleep(3)
            except Exception as e:
                logger.warning("  yfinance failed for threshold %.1f (attempt %d): %s", min_corr, attempt+1, e)
                time.sleep(3)
        if len(snapshots) < 20:
            logger.warning("  Skipping threshold %.1f (insufficient data)", min_corr)
            continue
        logger.info("    Got %d snapshots", len(snapshots))

        monthly_data = []
        for i, snap in enumerate(snapshots):
            targets = snap["targets"]
            lam2 = targets["lambda_2"]
            rho = targets["spectral_radius"]
            scg_risk = float(1 - lam2/rho) if rho > 1e-8 else 1.0
            vol = float(np.mean(snap["node_features"][:, 0]))
            n_edges = snap["edge_index"].shape[1] // 2 if snap["edge_index"].shape[1] > 0 else 0

            monthly_data.append({
                "idx": i, "date": snap.get("date", ""),
                "lambda_2": float(lam2), "spectral_gap": float(targets["spectral_gap"]),
                "spectral_radius": float(rho), "scg_risk": scg_risk,
                "avg_volatility": vol, "n_edges": n_edges,
            })

        # Predictive correlations at multiple horizons
        scg_risk = np.array([d["scg_risk"] for d in monthly_data])
        vol = np.array([d["avg_volatility"] for d in monthly_data])
        lam2 = np.array([d["lambda_2"] for d in monthly_data])
        gap = np.array([d["spectral_gap"] for d in monthly_data])

        def safe_corr(x, y):
            if len(x) < 5 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        horizon_results = {}
        for h in [1, 2, 5, 10]:
            if h >= len(monthly_data) - 5:
                continue
            window = 3
            pairs = {"scg": [], "vol": [], "lam2": [], "gap": []}
            for t in range(window, len(monthly_data) - h):
                future_vol = vol[t + h]
                pairs["scg"].append((np.mean(scg_risk[t-window:t]), future_vol))
                pairs["vol"].append((np.mean(vol[t-window:t]), future_vol))
                pairs["lam2"].append((np.mean(lam2[t-window:t]), future_vol))
                pairs["gap"].append((np.mean(gap[t-window:t]), future_vol))

            horizon_results[f"h={h}"] = {
                "scg_risk": safe_corr(np.array(pairs["scg"])[:,0], np.array(pairs["scg"])[:,1]),
                "volatility": safe_corr(np.array(pairs["vol"])[:,0], np.array(pairs["vol"])[:,1]),
                "lambda_2": safe_corr(np.array(pairs["lam2"])[:,0], np.array(pairs["lam2"])[:,1]),
                "spectral_gap": safe_corr(np.array(pairs["gap"])[:,0], np.array(pairs["gap"])[:,1]),
            }

        all_results[f"corr_{min_corr}"] = {
            "min_corr": min_corr,
            "n_snapshots": len(snapshots),
            "scg_risk_stats": {"mean": round(float(np.mean(scg_risk)), 4), "std": round(float(np.std(scg_risk)), 4)},
            "lambda_2_stats": {"mean": round(float(np.mean(lam2)), 4), "std": round(float(np.std(lam2)), 4)},
            "horizons": horizon_results,
        }

    # ── Figure: Predictive correlation heatmap ──
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use the best threshold (0.3) for the time series plot
    best_key = "corr_0.3"
    if best_key in all_results:
        horizons = list(all_results[best_key]["horizons"].keys())
        indicators = ["scg_risk", "volatility", "lambda_2", "spectral_gap"]
        heatmap = np.zeros((len(indicators), len(horizons)))
        for j, h in enumerate(horizons):
            for i, ind in enumerate(indicators):
                heatmap[i, j] = all_results[best_key]["horizons"][h].get(ind, 0)

        im = ax.imshow(heatmap, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels(horizons)
        ax.set_yticks(range(len(indicators)))
        ax.set_yticklabels(["SCG Risk", "Volatility", "λ₂", "Spectral Gap"])
        ax.set_xlabel("Prediction Horizon")
        ax.set_title("Predictive Correlation: Indicator → Future Volatility (corr threshold=0.3)")
        for i in range(len(indicators)):
            for j in range(len(horizons)):
                ax.text(j, i, f"{heatmap[i,j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(heatmap[i,j]) > 0.3 else "black")
        plt.colorbar(im, ax=ax, label="Pearson r")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "backtesting_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(RESULTS_DIR / "backtesting.json", "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


# ══════════════════════════════════════════════════════════════════════
# Experiment 5: Spectral Evolution Time Series
# ══════════════════════════════════════════════════════════════════════

def run_spectral_evolution():
    logger.info("═══ Experiment 5: Spectral Evolution ═══")


    from dashboard.data_api import build_daily_graph_snapshots

    snapshots = []
    for attempt in range(3):
        try:
            snapshots = build_daily_graph_snapshots(lookback_years=3, corr_window=60, min_corr=0.3, stride=1)
            if len(snapshots) > 100:
                break
            time.sleep(3)
        except Exception as e:
            logger.warning("  yfinance failed (attempt %d): %s", attempt+1, e)
            time.sleep(3)
    if len(snapshots) < 100:
        logger.error("  Insufficient data for spectral evolution. Skipping.")
        return {"error": "insufficient data"}
    logger.info("  Got %d daily snapshots", len(snapshots))

    dates = [s.get("date", f"d{i}") for i, s in enumerate(snapshots)]
    lam2 = [s["targets"]["lambda_2"] for s in snapshots]
    gap = [s["targets"]["spectral_gap"] for s in snapshots]
    rho = [s["targets"]["spectral_radius"] for s in snapshots]
    scg_risk = [1 - l/r if r > 1e-8 else 1.0 for l, r in zip(lam2, rho)]
    vol = [float(np.mean(s["node_features"][:, 0])) for s in snapshots]
    n_edges = [s["edge_index"].shape[1] // 2 for s in snapshots]

    # ── Figure: Spectral properties time series ──
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(lam2, color="steelblue", linewidth=0.5)
    axes[0].set_ylabel("λ₂ (Algebraic\nConnectivity)")
    axes[0].set_title("Spectral Properties of the 10-Bank Interbank Network (2021-2024)")

    axes[1].plot(gap, color="darkorange", linewidth=0.5)
    axes[1].set_ylabel("Spectral Gap")

    axes[2].plot(scg_risk, color="crimson", linewidth=0.5)
    axes[2].set_ylabel("SCG Risk\n(1 - λ₂/ρ)")

    axes[3].plot(vol, color="purple", linewidth=0.5, alpha=0.7)
    axes[3].set_ylabel("Avg Volatility")
    axes[3].set_xlabel("Trading Day")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "spectral_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure: Network density evolution ──
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(n_edges, color="teal", linewidth=0.5)
    ax.set_ylabel("Number of Edges")
    ax.set_xlabel("Trading Day")
    ax.set_title("Network Density Evolution (edges with corr > 0.3)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "network_density_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()

    result = {
        "n_snapshots": len(snapshots),
        "lambda_2": {"mean": round(float(np.mean(lam2)), 4), "std": round(float(np.std(lam2)), 4),
                      "min": round(float(np.min(lam2)), 4), "max": round(float(np.max(lam2)), 4)},
        "spectral_gap": {"mean": round(float(np.mean(gap)), 4), "std": round(float(np.std(gap)), 4)},
        "scg_risk": {"mean": round(float(np.mean(scg_risk)), 4), "std": round(float(np.std(scg_risk)), 4),
                      "min": round(float(np.min(scg_risk)), 4), "max": round(float(np.max(scg_risk)), 4)},
        "volatility": {"mean": round(float(np.mean(vol)), 4), "std": round(float(np.std(vol)), 4)},
        "n_edges": {"mean": round(float(np.mean(n_edges)), 1), "std": round(float(np.std(n_edges)), 1)},
    }

    with open(RESULTS_DIR / "spectral_evolution.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ══════════════════════════════════════════════════════════════════════
# Architecture Diagram (text-based for LaTeX tikz)
# ══════════════════════════════════════════════════════════════════════

def generate_architecture_diagram():
    logger.info("═══ Generating Architecture Diagram ═══")

    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                               facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, weight="bold")

    def arrow(x1, y1, x2, y2, text=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.15, text, ha="center", fontsize=7, color="gray")

    # Data Layer
    box(0.5, 6.5, 2.5, 1.2, "yfinance\nDaily OHLCV", "#AED6F1", 8)
    box(3.5, 6.5, 2.5, 1.2, "ECB SDW\nSovereign Yields", "#AED6F1", 8)

    # Processing
    box(1.5, 4.8, 3.5, 1.2, "Rolling Correlation\nAdjacency Matrix", "#D5F5E3", 9)
    arrow(2.75, 6.5, 3.25, 6.0)
    arrow(4.75, 6.5, 3.25, 6.0)

    # SCG Pipeline
    box(0.3, 2.8, 3.0, 1.5, "SCG Pipeline\n─────────\nLaplacian → Eigendecomp\n→ ER Gap Test\n→ Coarse-Grain\n→ Rescale", "#FADBD8", 7)
    arrow(3.25, 4.8, 1.8, 4.3)

    # GNN Pipeline
    box(4.0, 2.8, 3.5, 1.5, "GAT Temporal GNN\n─────────\nGATConv (4 heads)\n→ BatchNorm → ELU\n→ global_mean_pool\n→ LSTM → FC → [λ₂, δ, ρ]", "#D6EAF8", 7)
    arrow(3.25, 4.8, 5.75, 4.3)

    # ABM Pipeline
    box(8.2, 2.8, 3.5, 1.5, "3-Channel ABM\n─────────\n1. Credit Contagion\n2. Funding Liquidity\n3. Fire-Sale Losses", "#E8DAEF", 7)
    arrow(3.25, 4.8, 9.95, 4.3)

    # Risk Metrics
    box(12.2, 2.8, 3.0, 1.5, "Risk Metrics\n─────────\nSRISK, MES, CoVaR\nVaR, Delta-CoVaR\nSCG Risk = 1-λ₂/ρ", "#FCF3CF", 7)
    arrow(1.8, 2.8, 13.7, 2.8)
    arrow(5.75, 2.8, 13.7, 2.8)
    arrow(9.95, 2.8, 13.7, 2.8)

    # Dashboard
    box(5.5, 0.5, 5.0, 1.5, "Interactive Dashboard (Dash + FastAPI)\n─────────\nNetwork · Simulate · Spectral · Evolve · AI/Data", "#F5CBA7", 9)
    arrow(13.7, 2.8, 8.0, 2.0)

    # Title
    ax.text(8, 7.8, "SCR Financial Networks — System Architecture", ha="center", fontsize=14, weight="bold")

    plt.savefig(FIGURES_DIR / "architecture_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Architecture diagram saved")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_start = time.time()

    # Run all experiments
    scg_results = run_scg_experiments()
    gat_results = run_gat_experiments()
    abm_results = run_abm_experiments()
    backtest_results = run_backtesting_experiments()
    spectral_results = run_spectral_evolution()
    generate_architecture_diagram()

    total_time = time.time() - t_start
    logger.info("═══ All experiments complete in %.1fs ═══", total_time)
    logger.info("Results: %s", RESULTS_DIR)
    logger.info("Figures: %s", FIGURES_DIR)

    # Summary
    summary = {
        "total_time_seconds": round(total_time, 1),
        "n_scg_configs": len(scg_results),
        "n_gat_configs": len(gat_results["configs"]),
        "n_abm_scenarios": len(abm_results),
        "n_backtest_thresholds": len(backtest_results),
        "n_spectral_snapshots": spectral_results["n_snapshots"],
        "figures_generated": len(list(FIGURES_DIR.glob("*.png"))),
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary: %s", json.dumps(summary, indent=2))
