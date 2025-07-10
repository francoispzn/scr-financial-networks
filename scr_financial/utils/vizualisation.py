"""
Visualization utilities for financial network analysis.

This module provides functions for visualizing financial networks,
time series data, and simulation results.
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from pyvis.network import Network

logger = logging.getLogger(__name__)


def plot_network(
    G: nx.Graph,
    node_color: str = "CET1_ratio",
    node_size: str = "total_assets",
    edge_width: str = "weight",
    layout: str = "spring",
    figsize: Tuple[int, int] = (12, 10),
    title: str = "Financial Network",
    cmap: str = "RdYlGn",
    save_path: Optional[str] = None,
) -> None:
    """Plot a financial network.

    Args:
        G: Graph to plot.
        node_color: Node attribute to use for coloring.
        node_size: Node attribute to use for sizing.
        edge_width: Edge attribute to use for width.
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai',
            'spectral').
        figsize: Figure size.
        title: Plot title.
        cmap: Colormap for node colors.
        save_path: Path to save the figure (optional).
    """
    plt.figure(figsize=figsize)

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Get node colors
    norm = None
    if bool(nx.get_node_attributes(G, node_color)):
        node_colors_raw = [G.nodes[n].get(node_color, 0) for n in G.nodes()]
        vmin = min(node_colors_raw)
        vmax = max(node_colors_raw)
        norm = plt.Normalize(vmin, vmax)
        node_colors = matplotlib.colormaps[cmap](norm(node_colors_raw))
    else:
        node_colors = "skyblue"

    # Get node sizes
    if bool(nx.get_node_attributes(G, node_size)):
        node_sizes_raw = [
            G.nodes[n].get(node_size, 1000) / 1e10 for n in G.nodes()
        ]
        node_sizes = [100 + 1000 * s for s in node_sizes_raw]
    else:
        node_sizes = 300

    # Get edge widths
    if bool(nx.get_edge_attributes(G, edge_width)):
        edge_widths_raw = [G[u][v].get(edge_width, 1) for u, v in G.edges()]
        max_width = max(edge_widths_raw)
        edge_widths = [1 + 5 * w / max_width for w in edge_widths_raw]
    else:
        edge_widths = 1

    nx.draw_networkx(
        G,
        pos=pos,
        node_color=node_colors,
        node_size=node_sizes,
        width=edge_widths,
        with_labels=True,
        font_size=10,
        font_weight="bold",
        alpha=0.8,
        edge_color="gray",
    )

    # Add colorbar if node_color attribute exists
    if bool(nx.get_node_attributes(G, node_color)) and norm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label(node_color)

    plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_coarse_grained_network(
    original_G: nx.Graph,
    cg_G: nx.Graph,
    clusters: Dict[str, int],
    figsize: Tuple[int, int] = (18, 8),
    save_path: Optional[str] = None,
) -> None:
    """Plot original and coarse-grained networks side by side.

    Args:
        original_G: Original network.
        cg_G: Coarse-grained network.
        clusters: Mapping from original nodes to cluster IDs.
        figsize: Figure size.
        save_path: Path to save the figure (optional).
    """
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    pos_orig = nx.spring_layout(original_G, seed=42)
    cluster_ids = [clusters[n] for n in original_G.nodes()]

    nx.draw_networkx(
        original_G,
        pos=pos_orig,
        node_color=cluster_ids,
        cmap="tab10",
        node_size=300,
        with_labels=True,
        font_size=8,
        font_weight="bold",
        alpha=0.8,
        edge_color="gray",
    )
    plt.title("Original Network")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    pos_cg = nx.spring_layout(cg_G, seed=42)
    node_sizes = [cg_G.nodes[n].get("size", 1) * 100 for n in cg_G.nodes()]

    nx.draw_networkx(
        cg_G,
        pos=pos_cg,
        node_color=range(len(cg_G.nodes())),
        cmap="tab10",
        node_size=node_sizes,
        with_labels=True,
        font_size=10,
        font_weight="bold",
        alpha=0.8,
        edge_color="gray",
    )
    plt.title("Coarse-Grained Network")
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_time_series(
    data: pd.DataFrame,
    columns: List[str],
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Time Series",
    save_path: Optional[str] = None,
) -> None:
    """Plot time series data.

    Args:
        data: DataFrame containing time series data.
        columns: List of columns to plot.
        figsize: Figure size.
        title: Plot title.
        save_path: Path to save the figure (optional).
    """
    plt.figure(figsize=figsize)

    for col in columns:
        if col in data.columns:
            plt.plot(data.index, data[col], label=col)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Heatmap",
    cmap: str = "coolwarm",
    save_path: Optional[str] = None,
) -> None:
    """Plot a heatmap.

    Args:
        matrix: Matrix to plot.
        row_labels: Labels for rows (optional).
        col_labels: Labels for columns (optional).
        figsize: Figure size.
        title: Plot title.
        cmap: Colormap.
        save_path: Path to save the figure (optional).
    """
    plt.figure(figsize=figsize)

    sns.heatmap(
        matrix,
        annot=True,
        cmap=cmap,
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=0.5,
        fmt=".2f",
    )

    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def create_interactive_network(
    G: nx.Graph,
    node_color: str = "CET1_ratio",
    node_size: str = "total_assets",
    edge_width: str = "weight",
    height: str = "800px",
    width: str = "100%",
    save_path: str = "interactive_network.html",
) -> None:
    """Create an interactive network visualization using pyvis.

    Args:
        G: Graph to visualize.
        node_color: Node attribute to use for coloring.
        node_size: Node attribute to use for sizing.
        edge_width: Edge attribute to use for width.
        height: Height of the visualization.
        width: Width of the visualization.
        save_path: Path to save the HTML file.
    """
    net = Network(height=height, width=width, notebook=False)

    cmap = matplotlib.colormaps["RdYlGn"]

    # Initialise defaults so per-node loop never hits a NameError
    norm = None
    color_values: Dict = {}

    if bool(nx.get_node_attributes(G, node_color)):
        color_values_list = [
            G.nodes[n].get(node_color, 0) for n in G.nodes()
        ]
        vmin = min(color_values_list)
        vmax = max(color_values_list)
        norm = plt.Normalize(vmin, vmax)
        color_values = dict(zip(G.nodes(), color_values_list))

    for node in G.nodes():
        title = f"Node: {node}<br>"
        for attr, value in G.nodes[node].items():
            if isinstance(value, (int, float)):
                title += f"{attr}: {value:.2f}<br>"
            else:
                title += f"{attr}: {value}<br>"

        if norm is not None and node in color_values:
            rgba = cmap(norm(color_values[node]))
            color = (
                f"rgb({int(rgba[0] * 255)}, "
                f"{int(rgba[1] * 255)}, "
                f"{int(rgba[2] * 255)})"
            )
        else:
            color = "#97C2FC"

        if bool(nx.get_node_attributes(G, node_size)):
            size_val = G.nodes[node].get(node_size, 1000) / 1e10
            size = max(10, min(100, 10 + 50 * size_val))
        else:
            size = 25

        net.add_node(node, title=title, color=color, size=size)

    for source, target, data in G.edges(data=True):
        if edge_width in data:
            width = data[edge_width]
            width = max(1, min(10, width / 1e9))
        else:
            width = 1

        edge_title = f"Edge: {source} → {target}<br>"
        for attr, value in data.items():
            if isinstance(value, (int, float)):
                edge_title += f"{attr}: {value:.2f}<br>"
            else:
                edge_title += f"{attr}: {value}<br>"

        net.add_edge(source, target, title=edge_title, width=width)

    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250)
    net.save_graph(save_path)
