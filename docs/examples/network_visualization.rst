Network Visualization Example
===========================

This example demonstrates how to visualize financial networks and their coarse-grained representations using the SCR-Financial-Networks framework.

Overview
--------

Visualizing financial networks is essential for understanding their structure, identifying key nodes, and analyzing contagion pathways. This example shows how to create various visualizations of banking networks, including:

1. Basic network graphs
2. Spectral embeddings
3. Coarse-grained network representations
4. Dynamic visualizations of contagion spread

Prerequisites
------------

Before running this example, ensure you have:

- Installed the SCR-Financial-Networks package
- Downloaded the required financial data (or use the provided sample data)
- Set up the conda environment with visualization dependencies

Code Example
-----------

.. code-block:: python

    import scr_financial as scrf
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    from pyvis.network import Network

    # Initialize data preprocessor
    preprocessor = scrf.data.DataPreprocessor(
        start_date='2008-01-01',
        end_date='2008-12-31'
    )

    # Load bank data
    preprocessor.load_bank_node_data({
        'solvency': 'EBA_transparency',
        'liquidity': 'EBA_aggregated',
        'market_risk': 'NYU_VLAB'
    })

    # Load network data
    preprocessor.load_interbank_exposures('ECB_TARGET2')

    # Initialize network builder
    network_builder = scrf.network.FinancialNetworkBuilder(preprocessor)
    
    # Construct network for a specific time point
    time_point = '2008-09-15'
    G = network_builder.construct_network(time_point)
    
    # Compute Laplacian and perform spectral analysis
    network_builder.compute_laplacian()
    network_builder.spectral_analysis()
    
    # Initialize spectral coarse-graining
    scg = scrf.network.SpectralCoarseGraining(network_builder)
    
    # Find spectral gap and perform coarse-graining
    k, gap = network_builder.find_spectral_gap()
    print(f"Found spectral gap at k={k} with gap size {gap:.4f}")
    
    scg.coarse_grain(k)
    
    # Identify clusters
    clusters = scg.identify_clusters()
    
    # Basic network visualization
    plt.figure(figsize=(12, 10))
    
    # Node colors based on CET1 ratio
    cet1_values = [G.nodes[node]['CET1_ratio'] for node in G.nodes()]
    node_colors = plt.cm.RdYlGn(np.array(cet1_values) / 20)  # Normalize to 0-1 range
    
    # Node sizes based on total assets
    node_sizes = [G.nodes[node]['total_assets'] / 1e9 for node in G.nodes()]
    
    # Edge widths based on exposure weights
    edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges()]
    
    # Draw the network
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(
        G, 
        pos=pos,
        node_color=node_colors,
        node_size=node_sizes,
        width=edge_widths,
        with_labels=True,
        font_size=8,
        alpha=0.8
    )
    
    plt.title(f"Banking Network on {time_point}")
    plt.axis('off')
    plt.savefig('banking_network.png', dpi=300, bbox_inches='tight')
    
    # Visualize spectral embedding
    plt.figure(figsize=(10, 8))
    
    # Use first two non-trivial eigenvectors for 2D embedding
    x = network_builder.eigenvectors[:, 1]
    y = network_builder.eigenvectors[:, 2]
    
    plt.scatter(x, y, c=node_colors, s=node_sizes, alpha=0.8)
    
    for i, node in enumerate(G.nodes()):
        plt.annotate(node, (x[i], y[i]), fontsize=8)
    
    plt.title("Spectral Embedding of Banking Network")
    plt.xlabel("Eigenvector 1")
    plt.ylabel("Eigenvector 2")
    plt.savefig('spectral_embedding.png', dpi=300, bbox_inches='tight')
    
    # Visualize clusters
    plt.figure(figsize=(12, 10))
    
    # Color nodes by cluster
    cluster_colors = plt.cm.tab10(np.array(clusters) % 10)
    
    nx.draw_networkx(
        G, 
        pos=pos,
        node_color=cluster_colors,
        node_size=node_sizes,
        width=edge_widths,
        with_labels=True,
        font_size=8,
        alpha=0.8
    )
    
    plt.title(f"Banking Network Clusters (k={k})")
    plt.axis('off')
    plt.savefig('banking_clusters.png', dpi=300, bbox_inches='tight')
    
    # Create interactive visualization with pyvis
    net = Network(height="800px", width="100%", notebook=False)
    
    # Add nodes with attributes
    for node in G.nodes():
        net.add_node(
            node, 
            title=f"Bank: {node}<br>CET1: {G.nodes[node]['CET1_ratio']:.2f}%<br>Cluster: {clusters[list(G.nodes()).index(node)]}",
            size=G.nodes[node]['total_assets'] / 1e10,
            color=f"rgb({int(cluster_colors[list(G.nodes()).index(node)][0]*255)}, "
                  f"{int(cluster_colors[list(G.nodes()).index(node)][1]*255)}, "
                  f"{int(cluster_colors[list(G.nodes()).index(node)][2]*255)})"
        )
    
    # Add edges with attributes
    for u, v in G.edges():
        net.add_edge(
            u, v, 
            value=G[u][v]['weight'] * 10,
            title=f"Exposure: {G[u][v]['weight']:.2f} billion EUR"
        )
    
    # Set physics options
    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250)
    
    # Save interactive visualization
    net.save_graph("interactive_banking_network.html")

Visualization Types
-----------------

Static Visualizations
~~~~~~~~~~~~~~~~~~~

The example generates several static visualizations using matplotlib:

1. **Basic Network Graph**: Shows the banking network with nodes colored by CET1 ratio and sized by total assets
2. **Spectral Embedding**: Projects banks into a 2D space based on the first two non-trivial eigenvectors
3. **Cluster Visualization**: Colors nodes according to their cluster assignment from spectral coarse-graining

Interactive Visualizations
~~~~~~~~~~~~~~~~~~~~~~~~

The example also creates interactive visualizations using pyvis:

1. **Interactive Network**: Allows zooming, panning, and hovering for detailed information
2. **Dynamic Contagion Simulation**: Animates the spread of shocks through the network

Interpreting the Visualizations
-----------------------------

When analyzing these visualizations, look for:

- **Highly connected nodes**: These represent systemically important banks
- **Clusters**: Groups of banks that are more strongly connected to each other
- **Spectral embedding patterns**: Banks that are close in the embedding are likely to influence each other
- **Color patterns**: Identify banks with low capital ratios (red) that might be vulnerable

Further Customization
-------------------

You can customize these visualizations by:

- Changing the node attributes used for coloring and sizing
- Adjusting the layout algorithm (e.g., force-directed, circular)
- Adding time-series animations to show network evolution
- Incorporating additional financial indicators

See Also
-------

- :doc:`black_week_simulation` - For simulating crisis scenarios
- :doc:`../api/network_construction` - Documentation of the network construction module
- :doc:`../api/coarse_graining` - Documentation of the coarse-graining module
