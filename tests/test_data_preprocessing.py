"""
Tests for the data preprocessing module.

This module contains tests for the DataPreprocessor class and related
data collection and preprocessing functionality.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime

from scr_financial.data.preprocessor import DataPreprocessor
from scr_financial.data.collectors.eba_collector import EBACollector
from scr_financial.data.collectors.ecb_collector import ECBCollector
from scr_financial.data.collectors.market_collector import MarketDataCollector


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for the DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.start_date = '2020-01-01'
        self.end_date = '2020-12-31'
        self.bank_list = ["DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING"]
        self.preprocessor = DataPreprocessor(self.start_date, self.end_date, self.bank_list)
    
    def test_initialization(self):
        """Test initialization of DataPreprocessor."""
        self.assertEqual(self.preprocessor.start_date, self.start_date)
        self.assertEqual(self.preprocessor.end_date, self.end_date)
        self.assertEqual(self.preprocessor.bank_list, self.bank_list)
        self.assertIsInstance(self.preprocessor.eba_collector, EBACollector)
        self.assertIsInstance(self.preprocessor.ecb_collector, ECBCollector)
        self.assertIsInstance(self.preprocessor.market_collector, MarketDataCollector)
    
    def test_load_bank_node_data(self):
        """Test loading bank node data."""
        data_sources = {
            'solvency': 'EBA_transparency',
            'liquidity': 'EBA_aggregated'
        }
        
        node_data = self.preprocessor.load_bank_node_data(data_sources)
        
        self.assertIn('solvency', node_data)
        self.assertIn('liquidity', node_data)
        
        # Check that data contains expected banks
        for bank_id in self.bank_list:
            self.assertIn(bank_id, node_data['solvency']['bank_id'].values)
            self.assertIn(bank_id, node_data['liquidity']['bank_id'].values)
    
    def test_load_interbank_exposures(self):
        """Test loading interbank exposures."""
        edge_data = self.preprocessor.load_interbank_exposures('ECB_TARGET2')
        
        self.assertIn('interbank_exposures', edge_data)
        
        # Check that data contains expected columns
        self.assertIn('source', edge_data['interbank_exposures'].columns)
        self.assertIn('target', edge_data['interbank_exposures'].columns)
        self.assertIn('weight', edge_data['interbank_exposures'].columns)
        
        # Check that data contains connections between banks in bank_list
        for bank_id in self.bank_list:
            self.assertIn(bank_id, edge_data['interbank_exposures']['source'].values)
    
    def test_normalize_edge_weights(self):
        """Test normalization of edge weights."""
        # First load some data
        self.preprocessor.load_interbank_exposures('ECB_TARGET2')
        
        # Test degree normalization
        normalized_data = self.preprocessor.normalize_edge_weights(method='degree')
        
        self.assertIn('interbank_exposures', normalized_data)
        
        # Check that weights are normalized (sum should be different)
        original_sum = self.preprocessor.edge_data['interbank_exposures']['weight'].sum()
        normalized_sum = normalized_data['interbank_exposures']['weight'].sum()
        
        self.assertNotEqual(original_sum, normalized_sum)
    
    def test_filter_network(self):
        """Test filtering of network."""
        # First load and normalize data
        self.preprocessor.load_interbank_exposures('ECB_TARGET2')
        self.preprocessor.normalize_edge_weights()
        
        # Count edges before filtering
        edge_count_before = len(self.preprocessor.edge_data['interbank_exposures'])
        
        # Apply filtering
        filtered_data = self.preprocessor.filter_network(method='threshold', threshold=0.1)
        
        # Count edges after filtering
        edge_count_after = len(filtered_data['interbank_exposures'])
        
        # There should be fewer edges after filtering
        self.assertLess(edge_count_after, edge_count_before)
    
    def test_get_data_for_timepoint(self):
        """Test getting data for a specific time point."""
        # First load some data
        self.preprocessor.load_bank_node_data({
            'solvency': 'EBA_transparency',
            'liquidity': 'EBA_aggregated'
        })
        self.preprocessor.load_interbank_exposures('ECB_TARGET2')
        
        # Get data for a specific time point
        time_point = '2020-06-30'
        data = self.preprocessor.get_data_for_timepoint(time_point)
        
        self.assertIn('node_data', data)
        self.assertIn('edge_data', data)
        self.assertIn('system_data', data)
        
        # Check that node data contains expected categories
        self.assertIn('solvency', data['node_data'])
        self.assertIn('liquidity', data['node_data'])
        
        # Check that edge data contains expected categories
        self.assertIn('interbank_exposures', data['edge_data'])


class TestEBACollector(unittest.TestCase):
    """Test cases for the EBACollector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = EBACollector()
        self.start_date = '2020-01-01'
        self.end_date = '2020-12-31'
        self.bank_list = ["DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING"]
    
    def test_collect_transparency_data(self):
        """Test collecting transparency data."""
        data = self.collector.collect_transparency_data(
            self.start_date, 
            self.end_date, 
            self.bank_list
        )
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('bank_id', data.columns)
        self.assertIn('CET1_ratio', data.columns)
        self.assertIn
