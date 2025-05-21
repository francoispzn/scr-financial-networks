"""
Black Week Simulation Example

This script demonstrates how to use the SCR-Financial-Networks framework to simulate
the "Black Week" of the 2008 financial crisis, focusing on interbank contagion dynamics.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime, timedelta

# Add parent directory to path to import scr_financial
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import scr_financial as scrf
from scr_financial.data.preprocessor import DataPreprocessor
from scr_financial.network.builder import FinancialNetworkBuilder
from scr_financial.network.coarse_graining import SpectralCoarseGraining
from scr_financial.abm.simulation import BankingSystemSimulation


def main():
    """Run the Black Week simulation example."""
    print("Starting Black Week Simulation Example")
    
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
        start_date='2008-01-01',
        end_date='2008-12-31',
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
    
    # Load system indicators
    print("Loading system indicators...")
    preprocessor.load_system_indicators()
    
    # Normalize and filter network
    print("Preprocessing network data...")
    preprocessor.normalize_edge_weights(method='degree')
    preprocessor.filter_network(method='threshold', threshold=0.05)
    preprocessor.align_timescales()
    
    # Initialize framework
    print("Initializing network builder...")
    network_builder = FinancialNetworkBuilder(preprocessor)
    
    # Set up for Black Week 2008 analysis
    black_week_start = '2008-09-15'  # Lehman Brothers bankruptcy
    print(f"Constructing network for {black_week_start}...")
    G = network_builder.construct_network(black_week_start)
    
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
    scg.rescale()
    
    # Identify clusters
    clusters = scg.identify_clusters()
    cluster_mapping = {node: cluster for node, cluster in zip(G.nodes(), clusters)}
    print(f"Identified {len(set(clusters))} clusters")
    
    # Create coarse-grained graph
    cg_graph = scg.create_coarse_grained_graph()
    
    # Get data for simulation
    time_point_data = preprocessor.get_data_for_timepoint(black_week_start)
    
    # Extract bank data
    bank_data = {}
    for bank_id in bank_list:
        bank_data[bank_id] = {}
        for category, data in time_point_data['node_data'].items():
            if isinstance(data, pd.DataFrame):
                bank_row = data[data['bank_id'] == bank_id]
                if not bank_row.empty:
                    for col in bank_row.columns:
                        if col not in ['bank_id', 'date']:
                            bank_data[bank_id][col] = bank_row[col].values[0]
            else:
                # If data is a Series or other structure
                if bank_id in data:
                    bank_data[bank_id][category] = data[bank_id]
    
    # Extract network data
    network_data = {}
    for bank_id in bank_list:
        network_data[bank_id] = {}
        for target_id in bank_list:
            if bank_id != target_id:
                edge_data = time_point_data['edge_data']['interbank_exposures']
                edge = edge_data[(edge_data['source'] == bank_id) & (edge_data['target'] == target_id)]
                if not edge.empty:
                    network_data[bank_id][target_id] = edge['weight'].values[0]
    
    # Extract system indicators
    system_indicators = time_point_data['system_data']
    
    # Initialize simulation
    print("Initializing banking system simulation...")
    simulation = BankingSystemSimulation(bank_data, network_data, system_indicators)
    
    # Define shocks based on historical events
    shocks = {
        1: {  # Day 1: Lehman Brothers bankruptcy
            "DE_DBK": {'CET1_ratio': -1.0, 'LCR': -15},  # Deutsche Bank had exposure to Lehman
            "UK_BARC": {'CET1_ratio': -0.8, 'LCR': -10},  # Barclays had exposure to Lehman
            "system": {'funding_stress': 0.1}
        },
        3: {  # Day 3: AIG bailout
            "FR_BNP": {'CET1_ratio': -0.7, 'LCR': -12},  # BNP Paribas had exposure to AIG
            "CH_UBS": {'CET1_ratio': -1.2, 'LCR': -18},  # UBS had significant exposure to AIG
            "system": {'funding_stress': 0.15}
        },
        5: {  # Day 5: Money market funds "breaking the buck"
            "system": {'funding_stress': 0.25, 'CISS': 0.2}
        },
        7: {  # Day 7: Short selling ban
            "UK_HSBC": {'CET1_ratio': 0.5},  # HSBC benefited from short selling ban
            "UK_BARC": {'CET1_ratio': 0.6},  # Barclays benefited from short selling ban
            "system": {'funding_stress': 0.05}  # Slight improvement in funding conditions
        }
    }
    
    # Run simulation with full network
    print("Running simulation with full network...")
    full_results = simulation.run_simulation(10, shocks)
    
    # Initialize coarse-grained simulation
    print("Initializing coarse-grained simulation...")
    
    # Map bank data to clusters
    cg_bank_data = {}
    for cluster_id in range(len(set(clusters))):
        cluster_name = f"cluster_{cluster_id}"
        cg_bank_data[cluster_name] = {}
        
        # Find banks in this cluster
        banks_in_cluster = [bank_id for bank_id, c in cluster_mapping.items() if c == cluster_id]
        
        # Aggregate data for banks in cluster
        for attr in ['CET1_ratio', 'LCR', 'total_assets', 'risk_weighted_assets', 'cash',
                     'interbank_assets', 'interbank_liabilities']:
            values = [bank_data[bank_id].get(attr, 0) for bank_id in banks_in_cluster if bank_id in bank_data]
            if values:
                cg_bank_data[cluster_name][attr] = sum(values) / len(values)
            else:
                cg_bank_data[cluster_name][attr] = 0
    
    # Map network data to clusters
    cg_network_data = {}
    for i in range(len(set(clusters))):
        source_cluster = f"cluster_{i}"
        cg_network_data[source_cluster] = {}
        
        for j in range(len(set(clusters))):
            if i != j:
                target_cluster = f"cluster_{j}"
                
                # Find banks in source and target clusters
                banks_in_source = [bank_id for bank_id, c in cluster_mapping.items() if c == i]
                banks_in_target = [bank_id for bank_id, c in cluster_mapping.items() if c == j]
                
                # Aggregate exposures between clusters
                total_exposure = 0
                for source_bank in banks_in_source:
                    for target_bank in banks_in_target:
                        if source_bank in network_data and target_bank in network_data[source_bank]:
                            total_exposure += network_data[source_bank][target_bank]
                
                if total_exposure > 0:
                    cg_network_data[source_cluster][target_cluster] = total_exposure
    
    # Map shocks to clusters
    cg_shocks = {}
    for day, shock in shocks.items():
        cg_shocks[day] = {}
        
        for target, params in shock.items():
            if target == "system":
                cg_shocks[day]["system"] = params
            else:
                # Find which cluster this bank belongs to
                cluster_id = cluster_mapping.get(target)
                if cluster_id is not None:
                    cluster_name = f"cluster_{cluster_id}"
                    if cluster_name not in cg_shocks[day]:
                        cg_shocks[day][cluster_name] = {}
                    
                    # Add shock parameters
                    for param, value in params.items():
                        if param in cg_shocks[day][cluster_name]:
                            cg_shocks[day][cluster_name][param] += value
                        else:
                            cg_shocks[day][cluster_name][param] = value
    
    # Initialize coarse-grained simulation
    cg_simulation = BankingSystemSimulation(cg_bank_data, cg_network_data, system_indicators)
    
    # Run coarse-grained simulation
    print("Running coarse-grained simulation...")
    cg_results = cg_simulation.run_simulation(10, cg_shocks)
    
    # Analyze and visualize results
    print("Analyzing and visualizing results...")
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot CET1 ratios over time
    plt.figure(figsize=(12, 6))
    
    # Full network results
    for bank_id in bank_list:
        cet1_values = [state['bank_states'][bank_id]['CET1_ratio'] for state in full_results]
        plt.plot(range(len(cet1_values)), cet1_values, '--', alpha=0.5, label=f"{bank_id} (Full)")
    
    # Coarse-grained network results
    for cluster_id in range(len(set(clusters))):
        cluster_name = f"cluster_{cluster_id}"
        cet1_values = [state['bank_states'][cluster_name]['CET1_ratio'] for state in cg_results]
        plt.plot(range(len(cet1_values)), cet1_values, '-', linewidth=2, label=f"{cluster_name} (CG)")
    
    plt.title("CET1 Ratios During Black Week")
    plt.xlabel("Days")
    plt.ylabel("CET1 Ratio (%)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, "black_week_cet1_ratios.png"), dpi=300, bbox_inches='tight')
    
    # Plot LCR values over time
    plt.figure(figsize=(12, 6))
    
    # Full network results
    for bank_id in bank_list:
        lcr_values = [state['bank_states'][bank_id]['LCR'] for state in full_results]
        plt.plot(range(len(lcr_values)), lcr_values, '--', alpha=0.5, label=f"{bank_id} (Full)")
    
    # Coarse-grained network results
    for cluster_id in range(len(set(clusters))):
        cluster_name = f"cluster_{cluster_id}"
        lcr_values = [state['bank_states'][cluster_name]['LCR'] for state in cg_results]
        plt.plot(range(len(lcr_values)), lcr_values, '-', linewidth=2, label=f"{cluster_name} (CG)")
    
    plt.title("Liquidity Coverage Ratios During Black Week")
    plt.xlabel("Days")
    plt.ylabel("LCR (%)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, "black_week_lcr_values.png"), dpi=300, bbox_inches='tight')
    
    # Plot system indicators
    plt.figure(figsize=(12, 6))
    
    # Full network results
    ciss_values = [state['system_indicators']['CISS'] for state in full_results]
    funding_stress_values = [state['system_indicators']['funding_stress'] for state in full_results]
    
    plt.plot(range(len(ciss_values)), ciss_values, '-', linewidth=2, label="CISS")
    plt.plot(range(len(funding_stress_values)), funding_stress_values, '-', linewidth=2, label="Funding Stress")
    
    plt.title("System Stress Indicators During Black Week")
    plt.xlabel("Days")
    plt.ylabel("Stress Level")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, "black_week_system_indicators.png"), dpi=300, bbox_inches='tight')
    
    # Plot network density over time
    plt.figure(figsize=(12, 6))
    
    # Calculate network density for full network
    full_density = []
    for state in full_results:
        adj_matrix = np.zeros((len(bank_list), len(bank_list)))
        for i, source in enumerate(bank_list):
            for j, target in enumerate(bank_list):
                if source in state['network'] and target in state['network'][source]:
                    adj_matrix[i, j] = state['network'][source][target]
        
        # Compute density
        n = adj_matrix.shape[0]
        density = np.count_nonzero(adj_matrix) / (n * (n - 1))
        full_density.append(density)
    
    # Calculate network density for coarse-grained network
    cg_density = []
    for state in cg_results:
        adj_matrix = np.zeros((len(set(clusters)), len(set(clusters))))
        for i in range(len(set(clusters))):
            source = f"cluster_{i}"
            for j in range(len(set(clusters))):
                target = f"cluster_{j}"
                if source in state['network'] and target in state['network'][source]:
                    adj_matrix[i, j] = state['network'][source][target]
        
        # Compute density
        n = adj_matrix.shape[0]
        if n > 1:
            density = np.count_nonzero(adj_matrix) / (n * (n - 1))
        else:
            density = 0
        cg_density.append(density)
    
    plt.plot(range(len(full_density)), full_density, '-', linewidth=2, label="Full Network")
    plt.plot(range(len(cg_density)), cg_density, '-', linewidth=2, label="Coarse-Grained Network")
    
    plt.title("Network Density During Black Week")
    plt.xlabel("Days")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, "black_week_network_density.png"), dpi=300, bbox_inches='tight')
    
    # Compare computational performance
    print("\nPerformance Comparison:")
    print(f"Number of banks in full network: {len(bank_list)}")
    print(f"Number of clusters in coarse-grained network: {len(set(clusters))}")
    print(f"Reduction factor: {len(bank_list) / len(set(clusters)):.2f}x")
    
    # Compare spectral properties
    print("\nSpectral Properties:")
    print(f"Spectral gap index: {k}")
    print(f"Spectral gap size: {gap:.4f}")
    
    # Compute coarse-graining error
    error = scg.compute_coarse_graining_error()
    print(f"Coarse-graining error: {error:.4f}")
    
    # Validate diffusion dynamics
    errors = scg.compare_diffusion_dynamics(time_steps=5)
    print(f"Diffusion dynamics errors: {[f'{e:.4f}' for e in errors]}")
    
    print("\nSimulation complete. Results saved to the 'output' directory.")


if __name__ == "__main__":
    main()
