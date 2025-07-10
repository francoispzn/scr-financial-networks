#!/usr/bin/env python3
"""
Additional experiments for expanded paper:
  1. GAT attention weight analysis — which bank pairs matter most
  2. Per-target R² breakdown across seeds and configs
  3. Correlation threshold sensitivity for GNN training
  4. SCG risk vs lambda_2 scatter (relationship analysis)
  5. Network topology statistics over time
"""

import json, logging, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("experiments.additional")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 12, "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# ══════════════════════════════════════════════════════════════════
# Exp A: Attention Weight Analysis
# ══════════════════════════════════════════════════════════════════

def run_attention_analysis():
    logger.info("═══ Attention Weight Analysis ═══")
    import torch
    from dashboard.data_api import build_daily_graph_snapshots
    from scr_financial.ml.gnn_predictor import GNNPredictor
    from torch_geometric.data import Data

    snapshots = []
    for attempt in range(3):
        try:
            snapshots = build_daily_graph_snapshots(lookback_years=3, corr_window=60, min_corr=0.3, stride=5)
            if len(snapshots) > 50: break
            time.sleep(3)
        except: time.sleep(3)
    if len(snapshots) < 50:
        return {"error": "insufficient data"}

    bank_names = ["DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
                  "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA"]

    # Train a model
    np.random.seed(42); torch.manual_seed(42)
    predictor = GNNPredictor(seq_len=10, hidden_dim=64, num_gat_layers=2, heads=8, dropout=0.2)
    predictor.train(snapshots, epochs=300, lr=3e-3, patience=40)

    # Extract attention weights on test data
    predictor.model.eval()
    n = len(bank_names)
    attn_accum = np.zeros((n, n))
    n_samples = 0

    test_start = int(len(snapshots) * 0.8)
    for i in range(test_start, len(snapshots)):
        snap = snapshots[i]
        data = predictor._snapshot_to_data(snap)
        with torch.no_grad():
            _ = predictor.model.gnn(data.x, data.edge_index, data.edge_weight)
            attn_w = predictor.model.gnn.get_attention_weights()
            if attn_w is not None:
                ei = data.edge_index.numpy()
                aw = attn_w.numpy().mean(axis=1)  # Average across heads
                for idx in range(ei.shape[1]):
                    src, tgt = ei[0, idx], ei[1, idx]
                    if src < n and tgt < n:
                        attn_accum[src, tgt] += aw[idx]
                n_samples += 1

    if n_samples > 0:
        attn_avg = attn_accum / n_samples
    else:
        attn_avg = attn_accum

    # Figure: Attention heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn_avg, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(bank_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(bank_names, fontsize=8)
    ax.set_title("Mean GAT Attention Weights Across Test Set")
    ax.set_xlabel("Target Bank"); ax.set_ylabel("Source Bank")
    for i in range(n):
        for j in range(n):
            if attn_avg[i, j] > 0.001:
                ax.text(j, i, f"{attn_avg[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if attn_avg[i,j] > attn_avg.max()*0.6 else "black")
    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "attention_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Top attention pairs
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and attn_avg[i, j] > 0:
                pairs.append((bank_names[i], bank_names[j], round(float(attn_avg[i, j]), 4)))
    pairs.sort(key=lambda x: -x[2])

    result = {
        "n_test_samples": n_samples,
        "attention_matrix": attn_avg.tolist(),
        "top_10_pairs": pairs[:10],
        "bank_names": bank_names,
        "row_sums": {bank_names[i]: round(float(attn_avg[i].sum()), 4) for i in range(n)},
        "col_sums": {bank_names[j]: round(float(attn_avg[:, j].sum()), 4) for j in range(n)},
    }
    logger.info("  Top 3 attention pairs: %s", pairs[:3])
    return result


# ══════════════════════════════════════════════════════════════════
# Exp B: Per-target R² analysis
# ══════════════════════════════════════════════════════════════════

def run_per_target_analysis():
    logger.info("═══ Per-Target R² Analysis ═══")
    with open(RESULTS_DIR / "gat_sweep.json") as f:
        gat = json.load(f)

    # Extract per-target R² for all configs and seeds
    results = {}
    for config in gat["configs"]:
        name = config["config"]["name"]
        per_target = {"lambda_2": [], "spectral_gap": [], "spectral_radius": []}
        for seed_r in config["per_seed"]:
            for target in per_target:
                per_target[target].append(seed_r["test_r2_per_target"].get(target, 0))

        results[name] = {
            t: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
            for t, v in per_target.items()
        }

    # Figure: Per-target R² grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    configs = list(results.keys())
    targets = ["lambda_2", "spectral_gap", "spectral_radius"]
    x = np.arange(len(configs))
    w = 0.25
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, target in enumerate(targets):
        means = [results[c][target]["mean"] for c in configs]
        stds = [results[c][target]["std"] for c in configs]
        label = {"lambda_2": "λ₂", "spectral_gap": "Spectral Gap", "spectral_radius": "Spectral Radius"}[target]
        ax.bar(x + i*w, means, w, yerr=stds, label=label, color=colors[i], alpha=0.8, capsize=3)

    ax.set_xticks(x + w); ax.set_xticklabels(configs)
    ax.set_ylabel("Test R²"); ax.set_title("Per-Target Test R² Across GAT Configurations (5 seeds)")
    ax.legend(); ax.set_ylim(-0.2, 0.8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "per_target_r2.png", dpi=300, bbox_inches="tight")
    plt.close()

    with open(RESULTS_DIR / "per_target_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ══════════════════════════════════════════════════════════════════
# Exp C: Correlation threshold sensitivity
# ══════════════════════════════════════════════════════════════════

def run_threshold_sensitivity():
    logger.info("═══ Correlation Threshold Sensitivity ═══")
    from dashboard.data_api import build_daily_graph_snapshots

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    results = {}

    for thr in thresholds:
        logger.info("  Threshold: %.1f", thr)
        snaps = []
        for attempt in range(3):
            try:
                snaps = build_daily_graph_snapshots(lookback_years=3, corr_window=60, min_corr=thr, stride=10)
                if len(snaps) > 20: break
                time.sleep(2)
            except: time.sleep(2)
        if len(snaps) < 20:
            continue

        lam2 = [s["targets"]["lambda_2"] for s in snaps]
        gap = [s["targets"]["spectral_gap"] for s in snaps]
        rho = [s["targets"]["spectral_radius"] for s in snaps]
        n_edges = [s["edge_index"].shape[1] // 2 for s in snaps]
        vol = [float(np.mean(s["node_features"][:, 0])) for s in snaps]

        results[str(thr)] = {
            "threshold": thr, "n_snapshots": len(snaps),
            "lambda_2": {"mean": round(float(np.mean(lam2)), 4), "std": round(float(np.std(lam2)), 4)},
            "spectral_gap": {"mean": round(float(np.mean(gap)), 4), "std": round(float(np.std(gap)), 4)},
            "spectral_radius": {"mean": round(float(np.mean(rho)), 4), "std": round(float(np.std(rho)), 4)},
            "n_edges": {"mean": round(float(np.mean(n_edges)), 1), "std": round(float(np.std(n_edges)), 1)},
            "density": round(float(np.mean(n_edges)) / 45, 3),  # 10 banks = 45 possible edges
        }
        logger.info("    edges=%.1f±%.1f, λ₂=%.3f±%.3f", np.mean(n_edges), np.std(n_edges), np.mean(lam2), np.std(lam2))

    # Figure: Threshold sensitivity
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    thrs = [float(t) for t in sorted(results.keys())]
    lam2_means = [results[str(t)]["lambda_2"]["mean"] for t in thrs]
    lam2_stds = [results[str(t)]["lambda_2"]["std"] for t in thrs]
    edges_means = [results[str(t)]["n_edges"]["mean"] for t in thrs]
    gap_means = [results[str(t)]["spectral_gap"]["mean"] for t in thrs]

    axes[0].errorbar(thrs, lam2_means, yerr=lam2_stds, marker="o", capsize=4, color="steelblue")
    axes[0].set_xlabel("Correlation Threshold"); axes[0].set_ylabel("λ₂")
    axes[0].set_title("Algebraic Connectivity vs Threshold")

    axes[1].plot(thrs, edges_means, marker="s", color="coral")
    axes[1].set_xlabel("Correlation Threshold"); axes[1].set_ylabel("Mean Edges")
    axes[1].set_title("Network Density vs Threshold")

    axes[2].plot(thrs, gap_means, marker="^", color="green")
    axes[2].set_xlabel("Correlation Threshold"); axes[2].set_ylabel("Spectral Gap")
    axes[2].set_title("Spectral Gap vs Threshold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()

    with open(RESULTS_DIR / "threshold_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ══════════════════════════════════════════════════════════════════
# Exp D: SCG risk decomposition scatter
# ══════════════════════════════════════════════════════════════════

def run_risk_decomposition():
    logger.info("═══ SCG Risk Decomposition ═══")
    from dashboard.data_api import build_daily_graph_snapshots

    snaps = []
    for attempt in range(3):
        try:
            snaps = build_daily_graph_snapshots(lookback_years=3, corr_window=60, min_corr=0.3, stride=1)
            if len(snaps) > 100: break
            time.sleep(3)
        except: time.sleep(3)
    if len(snaps) < 100:
        return {"error": "insufficient data"}

    lam2 = np.array([s["targets"]["lambda_2"] for s in snaps])
    rho = np.array([s["targets"]["spectral_radius"] for s in snaps])
    scg_risk = 1 - lam2 / rho
    vol = np.array([float(np.mean(s["node_features"][:, 0])) for s in snaps])
    n_edges = np.array([s["edge_index"].shape[1] // 2 for s in snaps])

    # Correlation matrix between spectral indicators and volatility
    data = np.column_stack([lam2, rho, scg_risk, vol, n_edges])
    corr = np.corrcoef(data.T)
    labels = ["λ₂", "ρ", "SCG Risk", "Volatility", "Edges"]

    # Figure: Correlation matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(labels)
    ax.set_yticks(range(5)); ax.set_yticklabels(labels)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if abs(corr[i,j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Cross-Correlation of Spectral Indicators and Market Volatility")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "indicator_correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure: λ₂ vs Volatility scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(lam2, vol, alpha=0.15, s=5, color="steelblue")
    axes[0].set_xlabel("λ₂ (Algebraic Connectivity)")
    axes[0].set_ylabel("Average Volatility")
    axes[0].set_title(f"λ₂ vs Volatility (r={np.corrcoef(lam2,vol)[0,1]:.3f})")

    axes[1].scatter(scg_risk, vol, alpha=0.15, s=5, color="crimson")
    axes[1].set_xlabel("SCG Risk (1 - λ₂/ρ)")
    axes[1].set_ylabel("Average Volatility")
    axes[1].set_title(f"SCG Risk vs Volatility (r={np.corrcoef(scg_risk,vol)[0,1]:.3f})")

    axes[2].scatter(n_edges, vol, alpha=0.15, s=5, color="teal")
    axes[2].set_xlabel("Number of Edges")
    axes[2].set_ylabel("Average Volatility")
    axes[2].set_title(f"Network Density vs Volatility (r={np.corrcoef(n_edges,vol)[0,1]:.3f})")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "spectral_vol_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    result = {
        "n_snapshots": len(snaps),
        "correlation_matrix": {f"{labels[i]}_vs_{labels[j]}": round(float(corr[i,j]), 4)
                                for i in range(5) for j in range(i+1, 5)},
        "scg_risk_stats": {"mean": round(float(np.mean(scg_risk)), 4), "std": round(float(np.std(scg_risk)), 4)},
        "lambda_2_vol_corr": round(float(np.corrcoef(lam2, vol)[0, 1]), 4),
        "scg_risk_vol_corr": round(float(np.corrcoef(scg_risk, vol)[0, 1]), 4),
        "edges_vol_corr": round(float(np.corrcoef(n_edges, vol)[0, 1]), 4),
    }
    logger.info("  λ₂↔vol=%.3f, SCG↔vol=%.3f, edges↔vol=%.3f",
                result["lambda_2_vol_corr"], result["scg_risk_vol_corr"], result["edges_vol_corr"])
    return result


if __name__ == "__main__":
    t0 = time.time()
    results = {}
    results["attention"] = run_attention_analysis()
    results["per_target"] = run_per_target_analysis()
    results["threshold_sensitivity"] = run_threshold_sensitivity()
    results["risk_decomposition"] = run_risk_decomposition()

    with open(RESULTS_DIR / "additional.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("═══ Additional experiments complete in %.1fs ═══", time.time() - t0)
    logger.info("New figures: %s", list(FIGURES_DIR.glob("*.png")))
