"""
Visualization utilities for financial network analysis.

This module provides functions for visualizing financial networks,
time series data, and simulation results.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from pyvis.network import Network


def plot_network(G: nx.Graph, node_color: str = 'CET1_ratio', node_size: str = 'total_assets',
                edge_width: str = 'weight', layout: str = 'spring', figsize: Tuple[int, int] = (12, 10),
                title: str = 'Financial Network', cmap: str = 'RdYlGn', save_path: Optional[str] = None) -> None:
    """
    Plot a financial network.
    
    Parameters
    ----------
    G : networkx.Graph
        Graph to plot
    node_color : str
        Node attribute to use for coloring
    node_size : str
        Node attribute to use for sizing
    edge_width : str
        Edge attribute to use for width
    layout : str
        Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
    figsize : tuple
        Figure size
    title : str
        Plot title
    cmap : str
        Colormap for node colors
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Get node colors
    if node_color in nx.get_node_attributes(G, node_color):
        node_colors = [G.nodes[n].get(node_color, 0) for n in G.nodes()]
        vmin = min(node_colors)
        vmax = max(node_colors)
        # Normalize for colormap
        norm = plt.Normalize(vmin, vmax)
        node_colors = plt.cm.get_cmap(cmap)(norm(node_colors))
    else:
        node_colors = 'skyblue'
    
    # Get node sizes
    if node_size in nx.get_node_attributes(G, node_size):
        node_sizes = [G.nodes[n].get(node_size, 1000) / 1e10 for n in G.nodes()]
        # Scale sizes
        node_sizes = [100 + 1000 * s for s in node_sizes]
    else:
        node_sizes = 300
    
    # Get edge widths
    if edge_width in nx.get_edge_attributes(G, edge_width):
        edge_widths = [G[u][v].get(edge_width, 1) for u, v in G.edges()]
        # Scale widths
        max_width = max(edge_widths)
        edge_widths = [1 + 5 * w / max_width for w in edge_widths]
    else:
        edge_widths = 1
    
    # Draw the network
    nx.draw_networkx(
        G,
        pos=pos,
        node_color=node_colors,
        node_size=node_sizes,
        width=edge_widths,
        with_labels=True,
        font_size=10,
        font_weight='bold',
        alpha=0.8,
        edge_color='gray'
    )
    
    # Add colorbar if node_color is a node attribute
    if node_color in nx.get_node_attributes(G, node_color):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label(node_color)
    
    plt.title(title)
    plt.axis('off')
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_coarse_grained_network(original_G: nx.Graph, cg_G: nx.Graph, clusters: Dict[str, int],
                              figsize: Tuple[int, int] = (18, 8), save_path: Optional[str] = None) -> None:
    """
    Plot original and coarse-grained networks side by side.
    
    Parameters
    ----------
    original_G : networkx.Graph
        Original network
    cg_G : networkx.Graph
        Coarse-grained network
    clusters : dict
        Mapping from original nodes to cluster IDs
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    # Plot original network
    plt.subplot(1, 2, 1)
    
    # Choose layout
    pos_orig = nx.spring_layout(original_G, seed=42)
    
    # Color nodes by cluster
    cluster_ids = [clusters[n] for n in original_G.nodes()]
    
    # Draw the network
    nx.draw_networkx(
        original_G,
        pos=pos_orig,
        node_color=cluster_ids,
        cmap='tab10',
        node_size=300,
        with_labels=True,
        font_size=8,
        font_weight='bold',
        alpha=0.8,
        edge_color='gray'
    )
    
    plt.title("Original Network")
    plt.axis('off')
    
    # Plot coarse-grained network
    plt.subplot(1, 2, 2)
    
    # Choose layout
    pos_cg = nx.spring_layout(cg_G, seed=42)
    
    # Get node sizes based on number of original nodes
    node_sizes = [cg_G.nodes[n].get('size', 1) * 100 for n in cg_G.nodes()]
    
    # Draw the network
    nx.draw_networkx(
        cg_G,
        pos=pos_cg,
        node_color=range(len(cg_G.nodes())),
        cmap='tab10',
        node_size=node_sizes,
        with_labels=True,
        font_size=10,
        font_weight='bold',
        alpha=0.8,
        edge_color='gray'
    )
    
    plt.title("Coarse-Grained Network")
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_time_series(data: pd.DataFrame, columns: List[str], figsize: Tuple[int, int] = (12, 6),
                   title: str = 'Time Series', save_path: Optional[str] = None) -> None:
    """
    Plot time series data.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing time series data
    columns : list
        List of columns to plot
    figsize : tuple
        Figure size
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    for col in columns:
        if col in data.columns:
            plt.plot(data.index, data[col], label=col)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_heatmap(matrix: np.ndarray, row_labels: Optional[List[str]] = None,
               col_labels: Optional[List[str]] = None, figsize: Tuple[int, int] = (10, 8),
               title: str = 'Heatmap', cmap: str = 'coolwarm',
               save_path: Optional[str] = None) -> None:
    """
    Plot a heatmap.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix to plot
    row_labels : list, optional
        Labels for rows
    col_labels : list, optional
        Labels for columns
    figsize : tuple
        Figure size
    title : str
        Plot title
    cmap : str
        Colormap
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        matrix,
        annot=True,
        cmap=cmap,
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=0.5,
        fmt='.2f'
    )
    
    plt.title(title)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_interactive_network(G: nx.Graph, node_color: str = 'CET1_ratio',
                             node_size: str = 'total_assets', edge_width: str = 'weight',
                             height: str = '800px', width: str = '100%',
                             save_path: str = 'interactive_network.html') -> None:
    """
    Create an interactive network visualization using pyvis.
    
    Parameters
    ----------
    G : networkx.Graph
        Graph to visualize
    node_color : str
        Node attribute to use for coloring
    node_size : str
        Node attribute to use for sizing
    edge_width : str
        Edge attribute to use for width
    height : str
        Height of the visualization
    width : str
        Width of the visualization
    save_path : str
        Path to save the HTML file
    """
    # Create pyvis network
    net = Network(height=height, width=width, notebook=False)
    
    # Get color map
    cmap = plt.cm.RdYlGn
    
    # Get node attribute values for coloring
    if node_color in nx.get_node_attributes(G, node_color):
        color_values = [G.nodes[n].get(node_color, 0) for n in G.nodes()]
        vmin = min(color_values)
        vmax = max(color_values)
        norm = plt.Normalize(vmin, vmax)
    
    # Add nodes
    for node in G.nodes():
        # Create node title (tooltip)
        title = f"Node: {node}<br>"
        for attr, value in G.nodes[node].items():
            if isinstance(value, (int, float)):
                title += f"{attr}: {value:.2f}<br>"
            else:
                title += f"{attr}: {value}<br>"
        
        # Get node color
        if node_color in nx.get_node_attributes(G, node_color):
            color_value = G.nodes[node].get(node_color, 0)
            rgba = cmap(norm(color_value))
            color = f"rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})"
        else:
            color = "#97C2FC"  # Default pyvis blue
        
        # Get node size
        if node_size in nx.get_node_attributes(G, node_size):
            size = G.nodes[node].get(node_size, 1000) / 1e10
            size = max(10, min(100, 10 + 50 * size))  # Scale between 10 and 100
        else:
            size = 25
        
        # Add node to network
        net.add_node(node, title=title, color=color, size=size)
    
    # Add edges
    for source, target, data in G.edges(data=True):
        # Get edge width
        if edge_width in data:
            width = data[edge_width]
            # Scale width
            width = max(1, min(10, width / 1e9))  # Scale between 1 and 10
        else:
            width = 1
        
        # Create edge title (tooltip)
        title = f"Edge: {source} → {target}<br>"
        for attr, value in data.items():
            if isinstance(value, (int, float)):
                title += f"{attr}: {value:.2f}<br>"
            else:
                title += f"{attr}: {value}<br>"
        
        # Add edge to network
        net.add_edge(source, target, title=title, width=width)
    
    # Set physics options for better visualization
    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250)
    
    # Save to HTML file
    net.save_graph(save_path)
