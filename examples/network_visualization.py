"""
Network Visualization Example

This script demonstrates how to visualize financial networks and their
coarse-grained representations using the SCR-Financial-Networks framework.
"""

import logging
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Add parent directory to path to import scr_financial
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scr_financial as scrf
from scr_financial.data.preprocessor import DataPreprocessor
from scr_financial.network.builder import FinancialNetworkBuilder
from scr_financial.network.coarse_graining import SpectralCoarseGraining
from scr_financial.utils.vizualisation import (
    create_interactive_network,
    plot_coarse_grained_network,
    plot_heatmap,
    plot_network,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANK_LIST = [
    "DE_DBK",  # Deutsche Bank
    "FR_BNP",  # BNP Paribas
    "ES_SAN",  # Santander
    "IT_UCG",  # UniCredit
    "NL_ING",  # ING
    "SE_NDA",  # Nordea
    "CH_UBS",  # UBS
    "UK_BARC",  # Barclays
    "UK_HSBC",  # HSBC
    "FR_ACA",  # Credit Agricole
]

ANALYSIS_START_DATE = "2020-01-01"
ANALYSIS_END_DATE = "2020-12-31"
TIME_POINT = "2020-06-30"

NETWORK_FILTER_THRESHOLD = 0.05
OUTPUT_DIR = "output"
FIGURE_DPI = 300

# Normalisation divisor used when colouring by CET1 ratio
CET1_NORM_SCALE = 20.0

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
    """Run the network visualization example."""
    logger.info("Starting Network Visualization Example")

    # Initialize data preprocessor
    logger.info("Initializing data preprocessor...")
    preprocessor = DataPreprocessor(
        start_date=ANALYSIS_START_DATE,
        end_date=ANALYSIS_END_DATE,
        bank_list=BANK_LIST,
    )

    # Load bank data
    logger.info("Loading bank data...")
    preprocessor.load_bank_node_data(
        {
            "solvency": "EBA_transparency",
            "liquidity": "EBA_aggregated",
            "market_risk": "NYU_VLAB",
        }
    )

    # Load network data
    logger.info("Loading network data...")
    preprocessor.load_interbank_exposures("ECB_TARGET2")

    # Normalize and filter network
    logger.info("Preprocessing network data...")
    preprocessor.normalize_edge_weights(method="degree")
    preprocessor.filter_network(
        method="threshold", threshold=NETWORK_FILTER_THRESHOLD
    )

    # Initialize network builder
    logger.info("Initializing network builder...")
    network_builder = FinancialNetworkBuilder(preprocessor)

    # Construct network for a specific time point
    logger.info("Constructing network for %s...", TIME_POINT)
    G = network_builder.construct_network(TIME_POINT)

    # Compute Laplacian and perform spectral analysis
    logger.info("Performing spectral analysis...")
    network_builder.compute_laplacian()
    network_builder.spectral_analysis()

    # Find spectral gap
    k, gap = network_builder.find_spectral_gap()
    logger.info("Found spectral gap at k=%d with gap size %.4f", k, gap)

    # Initialize spectral coarse-graining
    logger.info("Performing spectral coarse-graining...")
    scg = SpectralCoarseGraining(network_builder)
    scg.coarse_grain(k)

    # Identify clusters
    clusters = scg.identify_clusters()
    cluster_mapping = {
        node: cluster for node, cluster in zip(G.nodes(), clusters)
    }
    logger.info("Identified %d clusters", len(set(clusters)))

    # Create coarse-grained graph
    cg_graph = scg.create_coarse_grained_graph()

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Basic Network Visualization
    logger.info("Creating basic network visualization...")
    plot_network(
        G,
        node_color="CET1_ratio",
        node_size="total_assets",
        edge_width="weight",
        layout="spring",
        figsize=(12, 10),
        title=f"Banking Network on {TIME_POINT}",
        cmap="RdYlGn",
        save_path=os.path.join(OUTPUT_DIR, "banking_network.png"),
    )

    # 2. Visualize Spectral Embedding
    logger.info("Creating spectral embedding visualization...")
    plt.figure(figsize=(10, 8))

    # Use first two non-trivial eigenvectors for 2D embedding
    x = network_builder.eigenvectors[:, 1]
    y = network_builder.eigenvectors[:, 2]

    # Color nodes by CET1 ratio
    cet1_values = [G.nodes[node].get("CET1_ratio", 0) for node in G.nodes()]
    node_colors = matplotlib.colormaps["RdYlGn"](
        np.array(cet1_values) / CET1_NORM_SCALE
    )

    # Size nodes by total assets
    node_sizes_raw = [
        G.nodes[node].get("total_assets", 1e9) / 1e10 for node in G.nodes()
    ]
    node_sizes = [100 + 1000 * s for s in node_sizes_raw]

    plt.scatter(x, y, c=node_colors, s=node_sizes, alpha=0.8)

    for i, node in enumerate(G.nodes()):
        plt.annotate(node, (x[i], y[i]), fontsize=8)

    plt.title("Spectral Embedding of Banking Network")
    plt.xlabel("Eigenvector 1")
    plt.ylabel("Eigenvector 2")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(OUTPUT_DIR, "spectral_embedding.png"),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    # 3. Visualize Clusters
    logger.info("Creating cluster visualization...")
    plt.figure(figsize=(12, 10))

    pos = nx.spring_layout(G, seed=42)

    cluster_colors = matplotlib.colormaps["tab10"](
        np.array(clusters, dtype=float) % 10 / 10.0
    )

    node_sizes_raw = [
        G.nodes[node].get("total_assets", 1e9) / 1e10 for node in G.nodes()
    ]
    node_sizes = [100 + 1000 * s for s in node_sizes_raw]

    nx.draw_networkx(
        G,
        pos=pos,
        node_color=cluster_colors,
        node_size=node_sizes,
        with_labels=True,
        font_size=8,
        font_weight="bold",
        alpha=0.8,
        edge_color="gray",
    )

    plt.title(f"Banking Network Clusters (k={k})")
    plt.axis("off")
    plt.savefig(
        os.path.join(OUTPUT_DIR, "banking_clusters.png"),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    # 4. Visualize Original and Coarse-Grained Networks Side by Side
    logger.info("Creating side-by-side visualization...")
    plot_coarse_grained_network(
        G,
        cg_graph,
        cluster_mapping,
        figsize=(18, 8),
        save_path=os.path.join(OUTPUT_DIR, "original_vs_coarse_grained.png"),
    )

    # 5. Visualize Adjacency Matrix as Heatmap
    logger.info("Creating adjacency matrix heatmap...")
    adj_matrix = nx.to_numpy_array(G)
    plot_heatmap(
        adj_matrix,
        row_labels=list(G.nodes()),
        col_labels=list(G.nodes()),
        figsize=(12, 10),
        title="Interbank Exposure Matrix",
        cmap="YlOrRd",
        save_path=os.path.join(OUTPUT_DIR, "adjacency_heatmap.png"),
    )

    # 6. Visualize Coarse-Grained Adjacency Matrix
    logger.info("Creating coarse-grained adjacency matrix heatmap...")
    cg_adj_matrix = nx.to_numpy_array(cg_graph)
    plot_heatmap(
        cg_adj_matrix,
        row_labels=list(cg_graph.nodes()),
        col_labels=list(cg_graph.nodes()),
        figsize=(10, 8),
        title="Coarse-Grained Interbank Exposure Matrix",
        cmap="YlOrRd",
        save_path=os.path.join(OUTPUT_DIR, "cg_adjacency_heatmap.png"),
    )

    # 7. Visualize Eigenvalue Spectrum
    logger.info("Creating eigenvalue spectrum visualization...")
    plt.figure(figsize=(10, 6))

    eigenvalues = network_builder.eigenvalues
    plt.plot(range(len(eigenvalues)), eigenvalues, "o-", markersize=6)

    plt.axvline(
        x=k, color="r", linestyle="--", label=f"Spectral Gap (k={k})"
    )

    plt.title("Eigenvalue Spectrum of Laplacian Matrix")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "eigenvalue_spectrum.png"),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    # 8. Visualize Eigenvectors
    logger.info("Creating eigenvector visualization...")
    plt.figure(figsize=(12, 8))

    num_eigenvectors = min(5, len(eigenvalues))
    for i in range(1, num_eigenvectors):
        plt.plot(
            range(len(G.nodes())),
            network_builder.eigenvectors[:, i],
            "o-",
            label=f"Eigenvector {i}",
        )

    plt.title("First Few Non-Trivial Eigenvectors")
    plt.xlabel("Node Index")
    plt.ylabel("Eigenvector Component")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "eigenvectors.png"),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    # 9. Create Interactive Network Visualization
    logger.info("Creating interactive network visualization...")
    create_interactive_network(
        G,
        node_color="CET1_ratio",
        node_size="total_assets",
        edge_width="weight",
        save_path=os.path.join(OUTPUT_DIR, "interactive_banking_network.html"),
    )

    # 10. Create Interactive Coarse-Grained Network Visualization
    logger.info("Creating interactive coarse-grained network visualization...")
    create_interactive_network(
        cg_graph,
        node_color="CET1_ratio",
        node_size="size",
        edge_width="weight",
        save_path=os.path.join(OUTPUT_DIR, "interactive_cg_network.html"),
    )

    # 11. Visualize Diffusion Dynamics Comparison
    logger.info("Creating diffusion dynamics comparison...")
    times, errors = scg.compare_diffusion_dynamics(time_steps=20)

    plt.figure(figsize=(10, 6))
    plt.plot(times, errors, "o-", markersize=6)
    plt.title(
        "Diffusion Dynamics Error Between Original and Coarse-Grained Networks"
    )
    plt.xlabel("Diffusion Time")
    plt.ylabel("Relative Error")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(OUTPUT_DIR, "diffusion_dynamics_error.png"),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    # 12. Visualize Bank Metrics by Cluster
    logger.info("Creating bank metrics by cluster visualization...")
    n_clusters = len(set(clusters))

    cluster_metrics: dict = {
        cluster_id: {"CET1_ratio": [], "LCR": [], "total_assets": []}
        for cluster_id in range(n_clusters)
    }

    for node in G.nodes():
        cluster_id = cluster_mapping[node]
        for metric in cluster_metrics[cluster_id]:
            if metric in G.nodes[node]:
                cluster_metrics[cluster_id][metric].append(G.nodes[node][metric])

    cluster_avgs = {
        cluster_id: {
            metric: np.mean(values) if values else 0.0
            for metric, values in m.items()
        }
        for cluster_id, m in cluster_metrics.items()
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metric_names = ["CET1_ratio", "LCR", "total_assets"]
    titles = [
        "Average CET1 Ratio by Cluster",
        "Average LCR by Cluster",
        "Average Total Assets by Cluster",
    ]

    tab10 = matplotlib.colormaps["tab10"]

    for i, (metric, title) in enumerate(zip(metric_names, titles)):
        values = [
            cluster_avgs[cid][metric] for cid in range(n_clusters)
        ]

        if metric == "total_assets":
            values = [v / 1e9 for v in values]
            ylabel = "Billions EUR"
        else:
            ylabel = metric

        bar_colors = tab10(
            np.arange(n_clusters, dtype=float) / max(n_clusters, 1)
        )
        axes[i].bar(range(n_clusters), values, color=bar_colors)
        axes[i].set_title(title)
        axes[i].set_xlabel("Cluster ID")
        axes[i].set_ylabel(ylabel)
        axes[i].set_xticks(range(n_clusters))
        axes[i].set_xticklabels(
            [f"Cluster {cid}" for cid in range(n_clusters)]
        )
        axes[i].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "cluster_metrics.png"),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    logger.info(
        "Visualization complete. Results saved to '%s'.", OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
