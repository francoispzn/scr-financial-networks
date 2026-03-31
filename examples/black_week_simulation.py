"""
Black Week Simulation Example

This script demonstrates how to use the SCR-Financial-Networks framework to
simulate the "Black Week" of the 2008 financial crisis, focusing on interbank
contagion dynamics.
"""

import logging
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# Add parent directory to path to import scr_financial
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scr_financial as scrf
from scr_financial.data.preprocessor import DataPreprocessor
from scr_financial.network.builder import FinancialNetworkBuilder
from scr_financial.network.coarse_graining import SpectralCoarseGraining
from scr_financial.abm.simulation import BankingSystemSimulation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANK_LIST = [
    "DE_DBK",   # Deutsche Bank
    "FR_BNP",   # BNP Paribas
    "ES_SAN",   # Santander
    "IT_UCG",   # UniCredit
    "NL_ING",   # ING
    "SE_NDA",   # Nordea
    "CH_UBS",   # UBS
    "UK_BARC",  # Barclays
    "UK_HSBC",  # HSBC
    "FR_ACA",   # Credit Agricole
]

ANALYSIS_START_DATE = "2008-01-01"
ANALYSIS_END_DATE = "2008-12-31"
BLACK_WEEK_START = "2008-09-15"  # Lehman Brothers bankruptcy

NETWORK_FILTER_THRESHOLD = 0.05
SIMULATION_STEPS = 10
OUTPUT_DIR = "output"
FIGURE_DPI = 300

# Aggregate bank attributes to carry over to clusters
CLUSTER_ATTRS = [
    "CET1_ratio",
    "LCR",
    "total_assets",
    "risk_weighted_assets",
    "cash",
    "interbank_assets",
    "interbank_liabilities",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Black Week simulation example."""
    logger.info("Starting Black Week Simulation Example")

    preprocessor = DataPreprocessor(
        start_date=ANALYSIS_START_DATE,
        end_date=ANALYSIS_END_DATE,
        bank_list=BANK_LIST,
    )

    logger.info("Loading bank data...")
    preprocessor.load_bank_node_data(
        {
            "solvency": "EBA_transparency",
            "liquidity": "EBA_aggregated",
            "market_risk": "NYU_VLAB",
        }
    )

    logger.info("Loading network data...")
    preprocessor.load_interbank_exposures("ECB_TARGET2")

    logger.info("Loading system indicators...")
    preprocessor.load_system_indicators()

    logger.info("Preprocessing network data...")
    preprocessor.normalize_edge_weights(method="degree")
    preprocessor.filter_network(method="threshold", threshold=NETWORK_FILTER_THRESHOLD)
    preprocessor.align_timescales()

    logger.info("Initializing network builder...")
    network_builder = FinancialNetworkBuilder(preprocessor)

    logger.info("Constructing network for %s...", BLACK_WEEK_START)
    G = network_builder.construct_network(BLACK_WEEK_START)

    logger.info("Performing spectral analysis...")
    network_builder.compute_laplacian()
    network_builder.spectral_analysis()

    k, gap = network_builder.find_spectral_gap()
    logger.info("Found spectral gap at k=%d with gap size %.4f", k, gap)

    logger.info("Performing spectral coarse-graining...")
    scg = SpectralCoarseGraining(network_builder)
    scg.coarse_grain(k)
    scg.rescale()

    clusters = scg.identify_clusters()
    cluster_mapping = {node: cluster for node, cluster in zip(G.nodes(), clusters)}
    n_clusters = len(set(clusters))
    logger.info("Identified %d clusters", n_clusters)

    cg_graph = scg.create_coarse_grained_graph()

    # ------------------------------------------------------------------
    # Extract bank / network data for the simulation
    # ------------------------------------------------------------------
    time_point_data = preprocessor.get_data_for_timepoint(BLACK_WEEK_START)

    bank_data = {}
    for bank_id in BANK_LIST:
        bank_data[bank_id] = {}
        for category, data in time_point_data["node_data"].items():
            if isinstance(data, pd.DataFrame):
                bank_row = data[data["bank_id"] == bank_id]
                if not bank_row.empty:
                    for col in bank_row.columns:
                        if col not in ("bank_id", "date"):
                            bank_data[bank_id][col] = bank_row[col].values[0]
            else:
                if bank_id in data:
                    bank_data[bank_id][category] = data[bank_id]

    network_data = {}
    for bank_id in BANK_LIST:
        network_data[bank_id] = {}
        for target_id in BANK_LIST:
            if bank_id != target_id:
                edge_data = time_point_data["edge_data"]["interbank_exposures"]
                edge = edge_data[
                    (edge_data["source"] == bank_id)
                    & (edge_data["target"] == target_id)
                ]
                if not edge.empty:
                    network_data[bank_id][target_id] = edge["weight"].values[0]

    system_indicators = time_point_data["system_data"]

    # ------------------------------------------------------------------
    # Shocks based on historical events
    # ------------------------------------------------------------------
    shocks = {
        1: {  # Day 1: Lehman Brothers bankruptcy
            "DE_DBK": {"CET1_ratio": -1.0, "LCR": -15},
            "UK_BARC": {"CET1_ratio": -0.8, "LCR": -10},
            "system": {"funding_stress": 0.1},
        },
        3: {  # Day 3: AIG bailout
            "FR_BNP": {"CET1_ratio": -0.7, "LCR": -12},
            "CH_UBS": {"CET1_ratio": -1.2, "LCR": -18},
            "system": {"funding_stress": 0.15},
        },
        5: {  # Day 5: Money market funds "breaking the buck"
            "system": {"funding_stress": 0.25, "CISS": 0.2},
        },
        7: {  # Day 7: Short selling ban
            "UK_HSBC": {"CET1_ratio": 0.5},
            "UK_BARC": {"CET1_ratio": 0.6},
            "system": {"funding_stress": 0.05},
        },
    }

    # ------------------------------------------------------------------
    # Full-network simulation
    # ------------------------------------------------------------------
    logger.info("Running simulation with full network...")
    simulation = BankingSystemSimulation(bank_data, network_data, system_indicators)
    full_results = simulation.run_simulation(SIMULATION_STEPS, shocks)

    # ------------------------------------------------------------------
    # Coarse-grained simulation
    # ------------------------------------------------------------------
    logger.info("Initializing coarse-grained simulation...")

    cg_bank_data = {}
    for cluster_id in range(n_clusters):
        cluster_name = f"cluster_{cluster_id}"
        cg_bank_data[cluster_name] = {}
        banks_in_cluster = [
            b for b, c in cluster_mapping.items() if c == cluster_id
        ]
        for attr in CLUSTER_ATTRS:
            values = [
                bank_data[b].get(attr, 0)
                for b in banks_in_cluster
                if b in bank_data
            ]
            cg_bank_data[cluster_name][attr] = (
                sum(values) / len(values) if values else 0
            )

    cg_network_data = {}
    for i in range(n_clusters):
        source_cluster = f"cluster_{i}"
        cg_network_data[source_cluster] = {}
        banks_in_source = [b for b, c in cluster_mapping.items() if c == i]

        for j in range(n_clusters):
            if i == j:
                continue
            target_cluster = f"cluster_{j}"
            banks_in_target = [b for b, c in cluster_mapping.items() if c == j]

            total_exposure = sum(
                network_data.get(src, {}).get(tgt, 0)
                for src in banks_in_source
                for tgt in banks_in_target
            )
            if total_exposure > 0:
                cg_network_data[source_cluster][target_cluster] = total_exposure

    cg_shocks = {}
    for day, shock in shocks.items():
        cg_shocks[day] = {}
        for target, params in shock.items():
            if target == "system":
                cg_shocks[day]["system"] = params
            else:
                cluster_id = cluster_mapping.get(target)
                if cluster_id is not None:
                    cluster_name = f"cluster_{cluster_id}"
                    if cluster_name not in cg_shocks[day]:
                        cg_shocks[day][cluster_name] = {}
                    for param, value in params.items():
                        if param in cg_shocks[day][cluster_name]:
                            cg_shocks[day][cluster_name][param] += value
                        else:
                            cg_shocks[day][cluster_name][param] = value

    cg_simulation = BankingSystemSimulation(
        cg_bank_data, cg_network_data, system_indicators
    )
    logger.info("Running coarse-grained simulation...")
    cg_results = cg_simulation.run_simulation(SIMULATION_STEPS, cg_shocks)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    logger.info("Analyzing and visualizing results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CET1 ratios
    plt.figure(figsize=(12, 6))
    for bank_id in BANK_LIST:
        cet1_values = [
            state["bank_states"][bank_id]["CET1_ratio"] for state in full_results
        ]
        plt.plot(range(len(cet1_values)), cet1_values, "--", alpha=0.5, label=f"{bank_id} (Full)")
    for cluster_id in range(n_clusters):
        cluster_name = f"cluster_{cluster_id}"
        cet1_values = [
            state["bank_states"][cluster_name]["CET1_ratio"] for state in cg_results
        ]
        plt.plot(
            range(len(cet1_values)),
            cet1_values,
            "-",
            linewidth=2,
            label=f"{cluster_name} (CG)",
        )
    plt.title("CET1 Ratios During Black Week")
    plt.xlabel("Days")
    plt.ylabel("CET1 Ratio (%)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.savefig(
        os.path.join(OUTPUT_DIR, "black_week_cet1_ratios.png"),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    # LCR values
    plt.figure(figsize=(12, 6))
    for bank_id in BANK_LIST:
        lcr_values = [
            state["bank_states"][bank_id]["LCR"] for state in full_results
        ]
        plt.plot(range(len(lcr_values)), lcr_values, "--", alpha=0.5, label=f"{bank_id} (Full)")
    for cluster_id in range(n_clusters):
        cluster_name = f"cluster_{cluster_id}"
        lcr_values = [
            state["bank_states"][cluster_name]["LCR"] for state in cg_results
        ]
        plt.plot(
            range(len(lcr_values)),
            lcr_values,
            "-",
            linewidth=2,
            label=f"{cluster_name} (CG)",
        )
    plt.title("Liquidity Coverage Ratios During Black Week")
    plt.xlabel("Days")
    plt.ylabel("LCR (%)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.savefig(
        os.path.join(OUTPUT_DIR, "black_week_lcr_values.png"),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    # System indicators
    plt.figure(figsize=(12, 6))
    ciss_values = [state["system_indicators"]["CISS"] for state in full_results]
    funding_stress_values = [
        state["system_indicators"]["funding_stress"] for state in full_results
    ]
    plt.plot(range(len(ciss_values)), ciss_values, "-", linewidth=2, label="CISS")
    plt.plot(
        range(len(funding_stress_values)),
        funding_stress_values,
        "-",
        linewidth=2,
        label="Funding Stress",
    )
    plt.title("System Stress Indicators During Black Week")
    plt.xlabel("Days")
    plt.ylabel("Stress Level")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.savefig(
        os.path.join(OUTPUT_DIR, "black_week_system_indicators.png"),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    # Network density
    plt.figure(figsize=(12, 6))

    full_density = []
    for state in full_results:
        adj_matrix = np.zeros((len(BANK_LIST), len(BANK_LIST)))
        for i, source in enumerate(BANK_LIST):
            for j, target in enumerate(BANK_LIST):
                if source in state["network"] and target in state["network"][source]:
                    adj_matrix[i, j] = state["network"][source][target]
        n = adj_matrix.shape[0]
        full_density.append(np.count_nonzero(adj_matrix) / (n * (n - 1)))

    cg_density = []
    for state in cg_results:
        adj_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            source = f"cluster_{i}"
            for j in range(n_clusters):
                target = f"cluster_{j}"
                if source in state["network"] and target in state["network"][source]:
                    adj_matrix[i, j] = state["network"][source][target]
        n = adj_matrix.shape[0]
        cg_density.append(
            np.count_nonzero(adj_matrix) / (n * (n - 1)) if n > 1 else 0
        )

    plt.plot(range(len(full_density)), full_density, "-", linewidth=2, label="Full Network")
    plt.plot(range(len(cg_density)), cg_density, "-", linewidth=2, label="Coarse-Grained Network")
    plt.title("Network Density During Black Week")
    plt.xlabel("Days")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.savefig(
        os.path.join(OUTPUT_DIR, "black_week_network_density.png"),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    logger.info("Performance Comparison:")
    logger.info("  Number of banks in full network: %d", len(BANK_LIST))
    logger.info("  Number of clusters in coarse-grained network: %d", n_clusters)
    logger.info("  Reduction factor: %.2fx", len(BANK_LIST) / n_clusters)

    logger.info("Spectral Properties:")
    logger.info("  Spectral gap index: %d", k)
    logger.info("  Spectral gap size: %.4f", gap)

    error = scg.compute_coarse_graining_error()
    logger.info("  Coarse-graining error: %.4f", error)

    errors = scg.compare_diffusion_dynamics(time_steps=5)
    logger.info("  Diffusion dynamics errors: %s", [f"{e:.4f}" for e in errors])

    logger.info("Simulation complete. Results saved to '%s'.", OUTPUT_DIR)


if __name__ == "__main__":
    main()
