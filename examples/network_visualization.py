"""
Network Visualization Example

This script demonstrates how to visualize financial networks and their coarse-grained
representations using the SCR-Financial-Networks framework.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from pyvis.network import Network

# Add parent directory to path to import scr_financial
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scr_financial as scrf
from scr_financial.data.preprocessor import DataPreprocessor
from scr_financial.network.builder import FinancialNetworkBuilder
from scr_financial.network.coarse_graining import SpectralCoarseGraining
from scr_financial.utils.visualization import (
    plot_network,
    plot_coarse_grained_network,
    plot_heatmap,
    create_interactive_network
)


def main():
    """Run the network visualization example."""
    print("Starting Network Visualization Example")
    
    # Define European banks to include in the analysis
    bank_list = [
        "DE_DBK",  # Deutsche Bank
        "FR_BNP",  # BNP Paribas
        "ES_SAN",  # Santander
        "IT_UCG",  # UniCredit
        "NL_ING",  # ING
        "SE_NDA",  # Nordea
        "CH_UBS",  # UBS
        "UK_BARC", # Barclays
        "UK_HSBC", # HSBC
        "FR_ACA"   # Credit Agricole
    ]
    
    # Initialize data preprocessor
    print("Initializing data preprocessor...")
    preprocessor = DataPreprocessor(
        start_date='2020-01-01',
        end_date='2020-12-31',
        bank_list=bank_list
    )
    
    # Load bank data
    print("Loading bank data...")
    preprocessor.load_bank_node_data({
        'solvency': 'EBA_transparency',
        'liquidity': 'EBA_aggregated',
        'market_risk': 'NYU_VLAB'
    })
    
    # Load network data
    print("Loading network data...")
    preprocessor.load_interbank_exposures('ECB_TARGET2')
    
    # Normalize and filter network
    print("Preprocessing network data...")
    preprocessor.normalize_edge_weights(method='degree')
    preprocessor.filter_network(method='threshold', threshold=0.05)
    
    # Initialize network builder
    print("Initializing network builder...")
    network_builder = FinancialNetworkBuilder(preprocessor)
    
    # Construct network for a specific time point
    time_point = '2020-06-30'
    print(f"Constructing network for {time_point}...")
    G = network_builder.construct_network(time_point)
    
    # Compute Laplacian and perform spectral analysis
    print("Performing spectral analysis...")
    network_builder.compute_laplacian()
    network_builder.spectral_analysis()
    
    # Find spectral gap
    k, gap = network_builder.find_spectral_gap()
    print(f"Found spectral gap at k={k} with gap size {gap:.4f}")
    
    # Initialize spectral coarse-graining
    print("Performing spectral coarse-graining...")
    scg = SpectralCoarseGraining(network_builder)
    scg.coarse_grain(k)
    
    # Identify clusters
    clusters = scg.identify_clusters()
    cluster_mapping = {node: cluster for node, cluster in zip(G.nodes(), clusters)}
    print(f"Identified {len(set(clusters))} clusters")
    
    # Create coarse-grained graph
    cg_graph = scg.create_coarse_grained_graph()
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Basic Network Visualization
    print("Creating basic network visualization...")
    plot_network(
        G,
        node_color='CET1_ratio',
        node_size='total_assets',
        edge_width='weight',
        layout='spring',
        figsize=(12, 10),
        title=f'Banking Network on {time_point}',
        cmap='RdYlGn',
        save_path=os.path.join(output_dir, "banking_network.png")
    )
    
    # 2. Visualize Spectral Embedding
    print("Creating spectral embedding visualization...")
    plt.figure(figsize=(10, 8))
    
    # Use first two non-trivial eigenvectors for 2D embedding
    x = network_builder.eigenvectors[:, 1]
    y = network_builder.eigenvectors[:, 2]
    
    # Color nodes by CET1 ratio
    cet1_values = [G.nodes[node].get('CET1_ratio', 0) for node in G.nodes()]
    node_colors = plt.cm.RdYlGn(np.array(cet1_values) / 20)  # Normalize to 0-1 range
    
    # Size nodes by total assets
    node_sizes = [G.nodes[node].get('total_assets', 1e9) / 1e10 for node in G.nodes()]
    node_sizes = [100 + 1000 * s for s in node_sizes]  # Scale for visualization
    
    plt.scatter(x, y, c=node_colors, s=node_sizes, alpha=0.8)
    
    for i, node in enumerate(G.nodes()):
        plt.annotate(node, (x[i], y[i]), fontsize=8)
    
    plt.title("Spectral Embedding of Banking Network")
    plt.xlabel("Eigenvector 1")
    plt.ylabel("Eigenvector 2")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "spectral_embedding.png"), dpi=300, bbox_inches='tight')
    
    # 3. Visualize Clusters
    print("Creating cluster visualization...")
    plt.figure(figsize=(12, 10))
    
    # Choose layout
    pos = nx.spring_layout(G, seed=42)
    
    # Color nodes by cluster
    cluster_colors = plt.cm.tab10(np.array(clusters) % 10)
    
    # Size nodes by total assets
    node_sizes = [G.nodes[node].get('total_assets', 1e9) / 1e10 for node in G.nodes()]
    node_sizes = [100 + 1000 * s for s in node_sizes]  # Scale for visualization
    
    # Draw the network
    nx.draw_networkx(
        G, 
        pos=pos,
        node_color=cluster_colors,
        node_size=node_sizes,
        with_labels=True,
        font_size=8,
        font_weight='bold',
        alpha=0.8,
        edge_color='gray'
    )
    
    plt.title(f"Banking Network Clusters (k={k})")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "banking_clusters.png"), dpi=300, bbox_inches='tight')
    
    # 4. Visualize Original and Coarse-Grained Networks Side by Side
    print("Creating side-by-side visualization...")
    plot_coarse_grained_network(
        G,
        cg_graph,
        cluster_mapping,
        figsize=(18, 8),
        save_path=os.path.join(output_dir, "original_vs_coarse_grained.png")
    )
    
    # 5. Visualize Adjacency Matrix as Heatmap
    print("Creating adjacency matrix heatmap...")
    # Get adjacency matrix
    adj_matrix = nx.to_numpy_array(G)
    
    # Plot heatmap
    plot_heatmap(
        adj_matrix,
        row_labels=list(G.nodes()),
        col_labels=list(G.nodes()),
        figsize=(12, 10),
        title="Interbank Exposure Matrix",
        cmap="YlOrRd",
        save_path=os.path.join(output_dir, "adjacency_heatmap.png")
    )
    
    # 6. Visualize Coarse-Grained Adjacency Matrix
    print("Creating coarse-grained adjacency matrix heatmap...")
    # Get coarse-grained adjacency matrix
    cg_adj_matrix = nx.to_numpy_array(cg_graph)
    
    # Plot heatmap
    plot_heatmap(
        cg_adj_matrix,
        row_labels=list(cg_graph.nodes()),
        col_labels=list(cg_graph.nodes()),
        figsize=(10, 8),
        title="Coarse-Grained Interbank Exposure Matrix",
        cmap="YlOrRd",
        save_path=os.path.join(output_dir, "cg_adjacency_heatmap.png")
    )
    
    # 7. Visualize Eigenvalue Spectrum
    print("Creating eigenvalue spectrum visualization...")
    plt.figure(figsize=(10, 6))
    
    eigenvalues = network_builder.eigenvalues
    plt.plot(range(len(eigenvalues)), eigenvalues, 'o-', markersize=6)
    
    # Highlight spectral gap
    plt.axvline(x=k, color='r', linestyle='--', label=f'Spectral Gap (k={k})')
    
    plt.title("Eigenvalue Spectrum of Laplacian Matrix")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "eigenvalue_spectrum.png"), dpi=300, bbox_inches='tight')
    
    # 8. Visualize Eigenvectors
    print("Creating eigenvector visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot first few non-trivial eigenvectors
    num_eigenvectors = min(5, len(eigenvalues))
    for i in range(1, num_eigenvectors):  # Skip the first eigenvector (constant)
        plt.plot(range(len(G.nodes())), network_builder.eigenvectors[:, i], 'o-', label=f'Eigenvector {i}')
    
    plt.title("First Few Non-Trivial Eigenvectors")
    plt.xlabel("Node Index")
    plt.ylabel("Eigenvector Component")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "eigenvectors.png"), dpi=300, bbox_inches='tight')
    
    # 9. Create Interactive Network Visualization
    print("Creating interactive network visualization...")
    create_interactive_network(
        G,
        node_color='CET1_ratio',
        node_size='total_assets',
        edge_width='weight',
        save_path=os.path.join(output_dir, "interactive_banking_network.html")
    )
    
    # 10. Create Interactive Coarse-Grained Network Visualization
    print("Creating interactive coarse-grained network visualization...")
    create_interactive_network(
        cg_graph,
        node_color='CET1_ratio',
        node_size='size',
        edge_width='weight',
        save_path=os.path.join(output_dir, "interactive_cg_network.html")
    )
    
    # 11. Visualize Diffusion Dynamics Comparison
    print("Creating diffusion dynamics comparison...")
    times, errors = scg.compare_diffusion_dynamics(time_steps=20)
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, errors, 'o-', markersize=6)
    plt.title("Diffusion Dynamics Error Between Original and Coarse-Grained Networks")
    plt.xlabel("Diffusion Time")
    plt.ylabel("Relative Error")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "diffusion_dynamics_error.png"), dpi=300, bbox_inches='tight')
    
    # 12. Visualize Bank Metrics by Cluster
    print("Creating bank metrics by cluster visualization...")
    # Collect metrics by cluster
    cluster_metrics = {}
    for cluster_id in range(len(set(clusters))):
        cluster_metrics[cluster_id] = {
            'CET1_ratio': [],
            'LCR': [],
            'total_assets': []
        }
    
    for node in G.nodes():
        cluster_id = cluster_mapping[node]
        for metric in cluster_metrics[cluster_id]:
            if metric in G.nodes[node]:
                cluster_metrics[cluster_id][metric].append(G.nodes[node][metric])
    
    # Compute averages
    cluster_avgs = {}
    for cluster_id, metrics in cluster_metrics.items():
        cluster_avgs[cluster_id] = {
            metric: np.mean(values) if values else 0
            for metric, values in metrics.items()
        }
    
    # Plot metrics by cluster
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['CET1_ratio', 'LCR', 'total_assets']
    titles = ['Average CET1 Ratio by Cluster', 'Average LCR by Cluster', 'Average Total Assets by Cluster']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        values = [cluster_avgs[cluster_id][metric] for cluster_id in range(len(set(clusters)))]
        
        if metric == 'total_assets':
            # Convert to billions for readability
            values = [v / 1e9 for v in values]
            ylabel = 'Billions EUR'
        else:
            ylabel = metric
        
        axes[i].bar(range(len(set(clusters))), values, color=plt.cm.tab10(range(len(set(clusters)))))
        axes[i].set_title(title)
        axes[i].set_xlabel('Cluster ID')
        axes[i].set_ylabel(ylabel)
        axes[i].set_xticks(range(len(set(clusters))))
        axes[i].set_xticklabels([f'Cluster {i}' for i in range(len(set(clusters)))])
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_metrics.png"), dpi=300, bbox_inches='tight')
    
    print("\nVisualization complete. Results saved to the 'output' directory.")


if __name__ == "__main__":
    main()