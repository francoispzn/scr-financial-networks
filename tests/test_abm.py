"""
Tests for the Agent-Based Model module.

This module contains tests for the BankAgent, BankingSystemSimulation,
and decision model classes.
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime

from scr_financial.abm.bank_agent import BankAgent
from scr_financial.abm.simulation import BankingSystemSimulation
from scr_financial.abm.decision_models import (
    DefaultDecisionModel,
    StressDecisionModel,
    LearningDecisionModel
)


class TestBankAgent(unittest.TestCase):
    """Test cases for the BankAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bank_id = "DE_DBK"
        self.initial_state = {
            'CET1_ratio': 13.2,
            'Tier1_leverage_ratio': 4.5,
            'LCR': 145,
            'NSFR': 115,
            'total_assets': 1.2e12,
            'risk_weighted_assets': 3.5e11,
            'cash': 1.5e11,
            'interbank_assets': 2.0e11,
            'interbank_liabilities': 1.8e11
        }
        self.bank = BankAgent(self.bank_id, self.initial_state)
    
    def test_initialization(self):
        """Test initialization of BankAgent."""
        self.assertEqual(self.bank.id, self.bank_id)
        self.assertEqual(self.bank.state, self.initial_state)
        self.assertEqual(self.bank.connections, {})
        self.assertEqual(self.bank.memory, [])
        self.assertIsInstance(self.bank.decision_model, DefaultDecisionModel)
    
    def test_update_state(self):
        """Test updating bank state."""
        new_state = {
            'CET1_ratio': 12.8,
            'LCR': 140
        }
        
        self.bank.update_state(new_state)
        
        # Check that state was updated
        self.assertEqual(self.bank.state['CET1_ratio'], 12.8)
        self.assertEqual(self.bank.state['LCR'], 140)
        
        # Check that memory was updated
        self.assertEqual(len(self.bank.memory), 1)
        self.assertEqual(self.bank.memory[0], self.initial_state)
    
    def test_assess_solvency(self):
        """Test assessing solvency."""
        # Bank should be solvent with initial state
        self.assertTrue(self.bank.assess_solvency())
        
        # Update state to make bank insolvent
        self.bank.update_state({'CET1_ratio': 4.0})
        
        # Bank should now be insolvent
        self.assertFalse(self.bank.assess_solvency())
    
    def test_assess_liquidity(self):
        """Test assessing liquidity."""
        # Bank should be liquid with initial state
        self.assertTrue(self.bank.assess_liquidity())
        
        # Update state to make bank illiquid
        self.bank.update_state({'LCR': 90})
        
        # Bank should now be illiquid
        self.assertFalse(self.bank.assess_liquidity())
    
    def test_calculate_lending_capacity(self):
        """Test calculating lending capacity."""
        capacity = self.bank.calculate_lending_capacity()
        
        self.assertIsInstance(capacity, float)
        self.assertGreater(capacity, 0.0)
        
        # Lending capacity should depend on LCR and CET1 ratio
        original_capacity = capacity
        
        # Increase LCR and check that capacity increases
        self.bank.update_state({'LCR': 160})
        new_capacity = self.bank.calculate_lending_capacity()
        self.assertGreater(new_capacity, original_capacity)
    
    def test_decide_lending_action(self):
        """Test deciding lending action."""
        # Create potential borrowers
        potential_borrowers = {
            "FR_BNP": {
                'CET1_ratio': 12.8,
                'LCR': 152
            },
            "ES_SAN": {
                'CET1_ratio': 11.9,
                'LCR': 158
            }
        }
        
        # Set market sentiment
        market_sentiment = 0.8
        
        # Make lending decision
        decision = self.bank.decide_lending_action(potential_borrowers, market_sentiment)
        
        self.assertIsInstance(decision, dict)
        self.assertIn('action', decision)
        self.assertEqual(decision['action'], 'lend')
        self.assertIn('allocations', decision)
        self.assertIsInstance(decision['allocations'], dict)
        
        # Check that allocations are made to both borrowers
        self.assertIn("FR_BNP", decision['allocations'])
        self.assertIn("ES_SAN", decision['allocations'])
        
        # Update state to make bank illiquid
        self.bank.update_state({'LCR': 90})
        
        # Make lending decision again
        decision = self.bank.decide_lending_action(potential_borrowers, market_sentiment)
        
        # Bank should now reduce lending
        self.assertEqual(decision['action'], 'reduce_lending')
    
    def test_update_connections(self):
        """Test updating connections."""
        bank_id = "FR_BNP"
        weight = 1.5e10
        
        self.bank.update_connections(bank_id, weight)
        
        self.assertIn(bank_id, self.bank.connections)
        self.assertEqual(self.bank.connections[bank_id], weight)
    
    def test_get_connection_strength(self):
        """Test getting connection strength."""
        bank_id = "FR_BNP"
        weight = 1.5e10
        
        # Add connection
        self.bank.update_connections(bank_id, weight)
        
        # Get connection strength
        strength = self.bank.get_connection_strength(bank_id)
        
        self.assertEqual(strength, weight)
        
        # Get connection strength for non-existent connection
        strength = self.bank.get_connection_strength("NONEXISTENT")
        
        self.assertEqual(strength, 0.0)


class TestBankingSystemSimulation(unittest.TestCase):
    """Test cases for the BankingSystemSimulation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create bank data
        self.bank_data = {
            "DE_DBK": {
                'CET1_ratio': 13.2,
                'Tier1_leverage_ratio': 4.5,
                'LCR': 145,
                'NSFR': 115,
                'total_assets': 1.2e12,
                'risk_weighted_assets': 3.5e11,
                'cash': 1.5e11,
                'interbank_assets': 2.0e11,
                'interbank_liabilities': 1.8e11
            },
            "FR_BNP": {
                'CET1_ratio': 12.8,
                'Tier1_leverage_ratio': 4.7,
                'LCR': 152,
                'NSFR': 118,
                'total_assets': 1.5e12,
                'risk_weighted_assets': 4.0e11,
                'cash': 1.8e11,
                'interbank_assets': 2.5e11,
                'interbank_liabilities': 2.2e11
            },
            "ES_SAN": {
                'CET1_ratio': 11.9,
                'Tier1_leverage_ratio': 5.1,
                'LCR': 158,
                'NSFR': 112,
                'total_assets': 1.0e12,
                'risk_weighted_assets': 3.0e11,
                'cash': 1.2e11,
                'interbank_assets': 1.8e11,
                'interbank_liabilities': 1.5e11
            }
        }
        
        # Create network data
        self.network_data = {
            "DE_DBK": {
                "FR_BNP": 1.5e10,
                "ES_SAN": 8.0e9
            },
            "FR_BNP": {
                "DE_DBK": 1.5e10,
                "ES_SAN": 1.4e10
            },
            "ES_SAN": {
                "DE_DBK": 8.0e9,
                "FR_BNP": 1.4e10
            }
        }
        
        # Create system indicators
        self.system_indicators = {
            'CISS': 0.15,
            'funding_stress': 0.1,
            'credit_to_GDP_gap': 2.5
        }
        
        # Initialize simulation
        self.simulation = BankingSystemSimulation(
            self.bank_data,
            self.network_data,
            self.system_indicators
        )
    
    def test_initialization(self):
        """Test initialization of BankingSystemSimulation."""
        self.assertEqual(len(self.simulation.banks), 3)
        self.assertEqual(self.simulation.network, self.network_data)
        self.assertEqual(self.simulation.system_indicators, self.system_indicators)
        self.assertEqual(self.simulation.time, 0)
        self.assertEqual(self.simulation.history, [])
        
        # Check that banks were initialized correctly
        for bank_id, bank in self.simulation.banks.items():
            self.assertIsInstance(bank, BankAgent)
            self.assertEqual(bank.id, bank_id)
            self.assertEqual(bank.state, self.bank_data[bank_id])
        
        # Check that connections were initialized correctly
        for bank_id, bank in self.simulation.banks.items():
            for target_id, weight in self.network_data.get(bank_id, {}).items():
                self.assertEqual(bank.connections.get(target_id), weight)
    
    def test_calculate_market_sentiment(self):
        """Test calculating market sentiment."""
        sentiment = self.simulation.calculate_market_sentiment()
        
        self.assertIsInstance(sentiment, float)
        self.assertGreaterEqual(sentiment, 0.0)
        self.assertLessEqual(sentiment, 1.0)
        
        # Sentiment should depend on CISS and funding stress
        expected_sentiment = max(0, min(1, 1 - self.system_indicators['CISS'])) * max(0, min(1, 1 - self.system_indicators['funding_stress']))
        self.assertAlmostEqual(sentiment, expected_sentiment)
    
    def test_apply_external_shock(self):
        """Test applying external shock."""
        # Define shock
        shock_params = {
            "DE_DBK": {
                'CET1_ratio': -1.0,
                'LCR': -20
            },
            "system": {
                'funding_stress': 0.2
            }
        }
        
        # Apply shock
        self.simulation.apply_external_shock(shock_params)
        
        # Check that bank state was updated
        self.assertEqual(self.simulation.banks["DE_DBK"].state['CET1_ratio'], 12.2)
        self.assertEqual(self.simulation.banks["DE_DBK"].state['LCR'], 125)
        
        # Check that system indicators were updated
        self.assertEqual(self.simulation.system_indicators['funding_stress'], 0.3)
    
    def test_simulate_interbank_lending(self):
        """Test simulating interbank lending."""
        # Run one round of interbank lending
        self.simulation.simulate_interbank_lending()
        
        # Check that network was updated
        self.assertEqual(self.simulation.network, {
            bank_id: bank.connections for bank_id, bank in self.simulation.banks.items()
        })
    
    def test_run_simulation(self):
        """Test running simulation."""
        # Define shocks
        shocks = {
            1: {
                "DE_DBK": {
                    'CET1_ratio': -1.0,
                    'LCR': -20
                }
            },
            3: {
                "system": {
                    'funding_stress': 0.2
                }
            }
        }
        
        # Run simulation
        steps = 5
        history = self.simulation.run_simulation(steps, shocks)
        
        self.assertEqual(len(history), steps)
        self.assertEqual(self.simulation.time, steps)
        
        # Check that history contains expected keys
        for state in history:
            self.assertIn('time', state)
            self.assertIn('bank_states', state)
            self.assertIn('network', state)
            self.assertIn('system_indicators', state)
    
    def test_get_adjacency_matrix(self):
        """Test getting adjacency matrix."""
        adj_matrix = self.simulation.get_adjacency_matrix()
        
        self.assertIsInstance(adj_matrix, np.ndarray)
        self.assertEqual(adj_matrix.shape, (3, 3))  # 3 banks
    
    def test_get_bank_metrics(self):
        """Test getting bank metrics."""
        metrics = self.simulation.get_bank_metrics()
        
        self.assertIsInstance(metrics, pd.DataFrame)
        self.assertEqual(len(metrics), 3)  # 3 banks
        self.assertIn('bank_id', metrics.columns)
        self.assertIn('time', metrics.columns)
        self.assertIn('CET1_ratio', metrics.columns)
        self.assertIn('LCR', metrics.columns)
    
    def test_get_system_metrics(self):
        """Test getting system metrics."""
        metrics = self.simulation.get_system_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('CISS', metrics)
        self.assertIn('funding_stress', metrics)
        self.assertIn('credit_to_GDP_gap', metrics)
        self.assertIn('avg_CET1_ratio', metrics)
        self.assertIn('avg_LCR', metrics)
        self.assertIn('network_density', metrics)


class TestDecisionModels(unittest.TestCase):
    """Test cases for decision models."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a bank agent
        self.bank_id = "DE_DBK"
        self.initial_state = {
            'CET1_ratio': 13.2,
            'Tier1_leverage_ratio': 4.5,
            'LCR': 145,
            'NSFR': 115,
            'total_assets': 1.2e12,
            'risk_weighted_assets': 3.5e11,
            'cash': 1.5e11,
            'interbank_assets': 2.0e11,
            'interbank_liabilities': 1.8e11
        }
        
        # Create potential borrowers/lenders
        self.potential_counterparties = {
            "FR_BNP": {
                'CET1_ratio': 12.8,
                'LCR': 152
            },
            "ES_SAN": {
                'CET1_ratio': 11.9,
                'LCR': 158
            }
        }
        
        # Create market sentiment
        self.market_sentiment = 0.8
        
        # Create decision models
        self.default_model = DefaultDecisionModel()
        self.stress_model = StressDecisionModel()
        self.learning_model = LearningDecisionModel()
        
        # Create bank agents with different models
        self.default_bank = BankAgent(self.bank_id, self.initial_state, self.default_model)
        self.stress_bank = BankAgent(self.bank_id, self.initial_state, self.stress_model)
        self.learning_bank = BankAgent(self.bank_id, self.initial_state, self.learning_model)
    
    def test_default_decision_model(self):
        """Test DefaultDecisionModel."""
        # Test lending decision
        lending_decision = self.default_model.decide_lending_action(
            self.default_bank,
            self.potential_counterparties,
            self.market_sentiment
        )
        
        self.assertIsInstance(lending_decision, dict)
        self.assertIn('action', lending_decision)
        self.assertEqual(lending_decision['action'], 'lend')
        self.assertIn('allocations', lending_decision)
        
        # Test response to shock
        shock_params = {
            'CET1_ratio': -1.0,
            'LCR': -20
        }
        
        response = self.default_model.respond_to_shock(self.default_bank, shock_params)
        
        self.assertIsInstance(response, dict)
        self.assertIn('actions', response)
        self.assertIsInstance(response['actions'], list)
    
    def test_stress_decision_model(self):
        """Test StressDecisionModel."""
        # Test lending decision with low market sentiment
        low_sentiment = 0.2
        
        lending_decision = self.stress_model.decide_lending_action(
            self.stress_bank,
            self.potential_counterparties,
            low_sentiment
        )
        
        self.assertIsInstance(lending_decision, dict)
        self.assertIn('action', lending_decision)
        self.assertEqual(lending_decision['action'], 'reduce_lending')
        
        # Test lending decision with normal market sentiment
        lending_decision = self.stress_model.decide_lending_action(
            self.stress_bank,
            self.potential_counterparties,
            self.market_sentiment
        )
        
        self.assertIsInstance(lending_decision, dict)
        self.assertIn('action', lending_decision)
        self.assertEqual(lending_decision['action'], 'lend')
    
    def test_learning_decision_model(self):
        """Test LearningDecisionModel."""
        # Test lending decision
        lending_decision = self.learning_model.decide_lending_action(
            self.learning_bank,
            self.potential_counterparties,
            self.market_sentiment
        )
        
        self.assertIsInstance(lending_decision, dict)
        self.assertIn('action', lending_decision)
        self.assertEqual(lending_decision['action'], 'lend')
        
        # Test updating interaction score
        lender_id = "DE_DBK"
        borrower_id = "FR_BNP"
        outcome = 0.8  # Positive outcome
        
        self.learning_model.update_interaction_score(lender_id, borrower_id, outcome)
        
        # Get interaction score
        score = self.learning_model.get_interaction_score(lender_id, borrower_id)
        
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)  # Should be positive


if __name__ == '__main__':
    unittest.main()
