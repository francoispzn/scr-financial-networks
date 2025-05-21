"""
Tests for the coarse-graining module.

This module contains tests for the SpectralCoarseGraining class and related
coarse-graining functionality.
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime

from scr_financial.data.preprocessor import DataPreprocessor
from scr_financial.network.builder import FinancialNetworkBuilder
from scr_financial.network.coarse_graining import SpectralCoarseGraining


class TestSpectralCoarseGraining(unittest.TestCase):
    """Test cases for the SpectralCoarseGraining class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize data preprocessor
        self.start_date = '2020-01-01'
        self.end_date = '2020-12-31'
        self.bank_list = ["DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
                          "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA"]
        self.preprocessor = DataPreprocessor(self.start_date, self.end_date, self.bank_list)
        
        # Load sample data
        self.preprocessor.load_bank_node_data({
            'solvency': 'EBA_transparency',
            'liquidity': 'EBA_aggregated'
        })
        self.preprocessor.load_interbank_exposures('ECB_TARGET2')
        
        # Initialize network builder and construct network
        self.network_builder = FinancialNetworkBuilder(self.preprocessor)
        time_point = '2020-06-30'
        self.network_builder.construct_network(time_point)
        self.network_builder.compute_laplacian()
        self.network_builder.spectral_analysis()
        
        # Initialize coarse-graining
        self.scg = SpectralCoarseGraining(self.network_builder)
    
    def test_initialization(self):
        """Test initialization of SpectralCoarseGraining."""
        self.assertEqual(self.scg.network_builder, self.network_builder)
        self.assertIsNone(self.scg.coarse_grained_laplacian)
        self.assertIsNone(self.scg.coarse_grained_adjacency)
        self.assertIsNone(self.scg.rescaled_adjacency)
        self.assertIsNone(self.scg.clusters)
        self.assertIsNone(self.scg.cluster_mapping)
        self.assertIsNone(self.scg.coarse_grained_graph)
    
    def test_coarse_grain(self):
        """Test coarse-graining."""
        # Perform coarse-graining
        cg_laplacian = self.scg.coarse_grain()
        
        self.assertIsNotNone(cg_laplacian)
        self.assertEqual(self.scg.coarse_grained_laplacian, cg_laplacian)
        self.assertIsNotNone(self.scg.coarse_grained_adjacency)
        
        # Check that matrices have the right shape
        n = len(self.network_builder.G.nodes())
        self.assertEqual(cg_laplacian.shape, (n, n))
        self.assertEqual(self.scg.coarse_grained_adjacency.shape, (n, n))
    
    def test_rescale(self):
        """Test rescaling."""
        # First perform coarse-graining
        self.scg.coarse_grain()
        
        # Rescale
        rescaled_adjacency = self.scg.rescale()
        
        self.assertIsNotNone(rescaled_adjacency)
        self.assertEqual(self.scg.rescaled_adjacency, rescaled_adjacency)
        
        # Check that matrix has the right shape
        n = len(self.network_builder.G.nodes())
        self.assertEqual(rescaled_adjacency.shape, (n, n))
    
    def test_identify_clusters(self):
        """Test identifying clusters."""
        # First perform coarse-graining
        self.scg.coarse_grain()
        
        # Identify clusters
        clusters = self.scg.identify_clusters()
        
        self.assertIsNotNone(clusters)
        self.assertEqual(self.scg.clusters, clusters)
        self.assertIsNotNone(self.scg.cluster_mapping)
        
        # Check that all nodes are assigned to a cluster
        self.assertEqual(len(clusters), len(self.network_builder.G.nodes()))
        
        # Check that cluster IDs are valid
        n_clusters = len(set(clusters))
        self.assertTrue(all(0 <= c < n_clusters for c in clusters))
    
    def test_create_coarse_grained_graph(self):
        """Test creating coarse-grained graph."""
        # First perform coarse-graining and identify clusters
        self.scg.coarse_grain()
        self.scg.identify_clusters()
        
        # Create coarse-grained graph
        cg_graph = self.scg.create_coarse_grained_graph()
        
        self.assertIsNotNone(cg_graph)
        self.assertEqual(self.scg.coarse_grained_graph, cg_graph)
        self.assertIsInstance(cg_graph, nx.DiGraph)
        
        # Check that graph has the right number of nodes
        n_clusters = len(set(self.scg.clusters))
        self.assertEqual(len(cg_graph.nodes()), n_clusters)
        
        # Check that node names follow the expected format
        for node in cg_graph.nodes():
            self.assertTrue(node.startswith('cluster_'))
    
    def test_map_node_to_cluster(self):
        """Test mapping node to cluster."""
        # First perform coarse-graining, identify clusters, and create coarse-grained graph
        self.scg.coarse_grain()
        self.scg.identify_clusters()
        self.scg.create_coarse_grained_graph()
        
        # Map a node to its cluster
        node = self.bank_list[0]
        cluster = self.scg.map_node_to_cluster(node)
        
        self.assertIsNotNone(cluster)
        self.assertTrue(cluster.startswith('cluster_'))
    
    def test_compute_coarse_graining_error(self):
        """Test computing coarse-graining error."""
        # First perform coarse-graining
        self.scg.coarse_grain()
        
        # Compute error
        error = self.scg.compute_coarse_graining_error()
        
        self.assertIsInstance(error, float)
        self.assertGreaterEqual(error, 0.0)
    
    def test_compare_diffusion_dynamics(self):
        """Test comparing diffusion dynamics."""
        # First perform coarse-graining
        self.scg.coarse_grain()
        
        # Compare diffusion dynamics
        errors = self.scg.compare_diffusion_dynamics(time_steps=5)
        
        self.assertIsInstance(errors, list)
        self.assertEqual(len(errors), 5)
        self.assertTrue(all(isinstance(e, float) for e in errors))
        self.assertTrue(all(e >= 0.0 for e in errors))


if __name__ == '__main__':
    unittest.main()