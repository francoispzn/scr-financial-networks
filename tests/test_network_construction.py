"""
Tests for the network construction module.

This module contains tests for the FinancialNetworkBuilder class and related
network construction functionality.
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime

from scr_financial.data.preprocessor import DataPreprocessor
from scr_financial.network.builder import FinancialNetworkBuilder


class TestFinancialNetworkBuilder(unittest.TestCase):
    """Test cases for the FinancialNetworkBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize data preprocessor
        self.start_date = '2020-01-01'
        self.end_date = '2020-12-31'
        self.bank_list = ["DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING"]
        self.preprocessor = DataPreprocessor(self.start_date, self.end_date, self.bank_list)
        
        # Load sample data
        self.preprocessor.load_bank_node_data({
            'solvency': 'EBA_transparency',
            'liquidity': 'EBA_aggregated'
        })
        self.preprocessor.load_interbank_exposures('ECB_TARGET2')
        
        # Initialize network builder
        self.network_builder = FinancialNetworkBuilder(self.preprocessor)
    
    def test_initialization(self):
        """Test initialization of FinancialNetworkBuilder."""
        self.assertEqual(self.network_builder.preprocessor, self.preprocessor)
        self.assertIsNone(self.network_builder.G)
        self.assertIsNone(self.network_builder.adjacency_matrix)
        self.assertIsNone(self.network_builder.laplacian)
        self.assertIsNone(self.network_builder.eigenvalues)
        self.assertIsNone(self.network_builder.eigenvectors)
    
    def test_construct_network(self):
        """Test network construction."""
        time_point = '2020-06-30'
        G = self.network_builder.construct_network(time_point)
        
        self.assertIsInstance(G, nx.DiGraph)
        self.assertEqual(self.network_builder.G, G)
        
        # Check that graph contains expected banks
        for bank_id in self.bank_list:
            self.assertIn(bank_id, G.nodes())
        
        # Check that graph has edges
        self.assertGreater(len(G.edges()), 0)
        
        # Check that adjacency matrix was created
        self.assertIsNotNone(self.network_builder.adjacency_matrix)
    
    def test_compute_laplacian(self):
        """Test computation of Laplacian matrix."""
        # First construct network
        time_point = '2020-06-30'
        self.network_builder.construct_network(time_point)
        
        # Compute Laplacian
        laplacian = self.network_builder.compute_laplacian(normalized=True)
        
        self.assertIsNotNone(laplacian)
        self.assertEqual(self.network_builder.laplacian, laplacian)
        
        # Check that Laplacian has the right shape
        n = len(self.network_builder.G.nodes())
        self.assertEqual(laplacian.shape, (n, n))
    
    def test_spectral_analysis(self):
        """Test spectral analysis."""
        # First construct network and compute Laplacian
        time_point = '2020-06-30'
        self.network_builder.construct_network(time_point)
        self.network_builder.compute_laplacian()
        
        # Perform spectral analysis
        eigenvalues, eigenvectors = self.network_builder.spectral_analysis()
        
        self.assertIsNotNone(eigenvalues)
        self.assertIsNotNone(eigenvectors)
        self.assertEqual(self.network_builder.eigenvalues, eigenvalues)
        self.assertEqual(self.network_builder.eigenvectors, eigenvectors)
        
        # Check that eigenvalues and eigenvectors have the right shape
        n = len(self.network_builder.G.nodes())
        self.assertEqual(len(eigenvalues), n)
        self.assertEqual(eigenvectors.shape, (n, n))
    
    def test_find_spectral_gap(self):
        """Test finding spectral gap."""
        # First construct network, compute Laplacian, and perform spectral analysis
        time_point = '2020-06-30'
        self.network_builder.construct_network(time_point)
        self.network_builder.compute_laplacian()
        self.network_builder.spectral_analysis()
        
        # Find spectral gap
        k, gap = self.network_builder.find_spectral_gap()
        
        self.assertIsInstance(k, int)
        self.assertIsInstance(gap, float)
        self.assertGreater(k, 0)
        self.assertGreater(gap, 0)
    
    def test_get_node_attribute_matrix(self):
        """Test getting node attribute matrix."""
        # First construct network
        time_point = '2020-06-30'
        self.network_builder.construct_network(time_point)
        
        # Get node attribute matrix
        attr_matrix = self.network_builder.get_node_attribute_matrix('CET1_ratio')
        
        self.assertIsInstance(attr_matrix, np.ndarray)
        self.assertEqual(len(attr_matrix), len(self.network_builder.G.nodes()))
    
    def test_compute_centrality_measures(self):
        """Test computing centrality measures."""
        # First construct network
        time_point = '2020-06-30'
        self.network_builder.construct_network(time_point)
        
        # Compute centrality measures
        centrality_measures = self.network_builder.compute_centrality_measures()
        
        self.assertIsInstance(centrality_measures, dict)
        self.assertIn('degree', centrality_measures)
        self.assertIn('eigenvector', centrality_measures)
        self.assertIn('betweenness', centrality_measures)
        self.assertIn('closeness', centrality_measures)
        self.assertIn('pagerank', centrality_measures)
        
        # Check that each measure includes all nodes
        for measure, values in centrality_measures.items():
            self.assertEqual(set(values.keys()), set(self.network_builder.G.nodes()))


if __name__ == '__main__':
    unittest.main()
