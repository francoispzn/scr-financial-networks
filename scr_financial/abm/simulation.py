"""
Banking system simulation for financial network analysis.

This module provides the BankingSystemSimulation class for simulating
interbank lending and contagion dynamics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import networkx as nx

from .bank_agent import BankAgent


class BankingSystemSimulation:
    """
    Implements the agent-based simulation engine.
    
    Parameters
    ----------
    bank_data : dict
        Dictionary mapping bank IDs to their initial states
    network_data : dict
        Dictionary representing the network structure
    system_indicators : dict
        Dictionary of system-wide indicators
    
    Attributes
    ----------
    banks : dict
        Dictionary of bank agents
    network : dict
        Current network structure
    time : int
        Current simulation time
    history : list
        History of system states
    """
    
    def __init__(
        self, 
        bank_data: Dict[str, Dict[str, Any]], 
        network_data: Dict[str, Dict[str, float]], 
        system_indicators: Dict[str, Any]
    ):
        """Initialize the simulation with bank data, network data, and system indicators."""
        self.banks = {}
        self.network = network_data.copy()
        self.system_indicators = system_indicators.copy()
        self.time = 0
        self.history = []
        
        # Initialize bank agents
        for bank_id, data in bank_data.items():
            self.banks[bank_id] = BankAgent(bank_id, data)
            
        # Initialize connections
        for bank_id, connections in network_data.items():
            if bank_id in self.banks:
                for target_id, strength in connections.items():
                    if target_id in self.banks:
                        self.banks[bank_id].connections[target_id] = strength
    
    def calculate_market_sentiment(self) -> float:
        """
        Calculate overall market sentiment based on system indicators.
        
        Returns
        -------
        float
            Market sentiment (0-1 scale, 1 being positive)
        """
        # Simple model: inverse of CISS (higher CISS = lower sentiment)
        ciss = self.system_indicators.get('CISS', 0.5)
        
        # Normalize to 0-1 range, where 1 is positive sentiment
        sentiment = max(0, min(1, 1 - ciss))
        
        # Adjust based on funding stress if available
        if 'funding_stress' in self.system_indicators:
            funding_stress = self.system_indicators['funding_stress']
            sentiment *= max(0, min(1, 1 - funding_stress))
        
        return sentiment
    
    def apply_external_shock(self, shock_params: Dict[str, Any]) -> None:
        """
        Apply an external shock to the system.
        
        Parameters
        ----------
        shock_params : dict
            Dictionary containing shock parameters
        """
        # Apply bank-specific shocks
        for bank_id, shock in shock_params.items():
            if bank_id == 'system':
                # System-wide shock
                for indicator, value in shock.items():
                    if indicator in self.system_indicators:
                        self.system_indicators[indicator] += value
            elif bank_id in self.banks:
                # Bank-specific shock
                new_state = {}
                for key, value in shock.items():
                    if key in self.banks[bank_id].state:
                        new_state[key] = self.banks[bank_id].state[key] + value
                
                self.banks[bank_id].update_state(new_state)
                
                # Let the bank respond to the shock
                response = self.banks[bank_id].respond_to_shock(shock)
                
                # Process response actions
                if response and 'actions' in response:
                    for action in response['actions']:
                        if action['type'] == 'reduce_lending':
                            # Reduce connections to other banks
                            for target_id, reduction in action.get('targets', {}).items():
                                if target_id in self.banks[bank_id].connections:
                                    self.banks[bank_id].connections[target_id] *= (1 - reduction)
    
    def simulate_interbank_lending(self) -> None:
        """
        Simulate one round of interbank lending decisions.
        """
        # Get current bank states for decision-making
        current_states = {bank_id: bank.state for bank_id, bank in self.banks.items()}
        
        # Calculate market sentiment
        sentiment = self.calculate_market_sentiment()
        
        # Each bank makes lending decisions
        lending_decisions = {}
        for bank_id, bank in self.banks.items():
            decision = bank.decide_lending_action(current_states, sentiment)
            lending_decisions[bank_id] = decision
        
        # Each bank makes borrowing decisions
        borrowing_decisions = {}
        for bank_id, bank in self.banks.items():
            decision = bank.decide_borrowing_action(current_states, sentiment)
            borrowing_decisions[bank_id] = decision
        
        # Match lending and borrowing decisions
        self._match_lending_borrowing(lending_decisions, borrowing_decisions)
        
        # Update network data based on new connections
        self.update_network()
    
    def _match_lending_borrowing(
        self, 
        lending_decisions: Dict[str, Dict[str, Any]], 
        borrowing_decisions: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Match lending and borrowing decisions between banks.
        
        Parameters
        ----------
        lending_decisions : dict
            Dictionary of lending decisions by bank
        borrowing_decisions : dict
            Dictionary of borrowing decisions by bank
        """
        # Process lending decisions
        for lender_id, decision in lending_decisions.items():
            if decision["action"] == "lend":
                for borrower_id, amount in decision.get("allocations", {}).items():
                    # Check if borrower wants to borrow
                    borrower_decision = borrowing_decisions.get(borrower_id, {})
                    if borrower_decision.get("action") == "borrow":
                        # Get maximum amount borrower wants from this lender
                        max_borrow = borrower_decision.get("sources", {}).get(lender_id, 0)
                        
                        # Determine actual amount (minimum of offer and request)
                        actual_amount = min(amount, max_borrow)
                        
                        if actual_amount > 0:
                            # Update lender's state
                            self.banks[lender_id].state["interbank_assets"] = self.banks[lender_id].state.get("interbank_assets", 0) + actual_amount
                            self.banks[lender_id].state["cash"] = self.banks[lender_id].state.get("cash", 0) - actual_amount
                            
                            # Update borrower's state
                            self.banks[borrower_id].state["interbank_liabilities"] = self.banks[borrower_id].state.get("interbank_liabilities", 0) + actual_amount
                            self.banks[borrower_id].state["cash"] = self.banks[borrower_id].state.get("cash", 0) + actual_amount
                            
                            # Update connection strength
                            self.banks[lender_id].update_connections(borrower_id, actual_amount)
    
    def update_network(self) -> None:
        """
        Update network data based on current bank connections.
        """
        for bank_id, bank in self.banks.items():
            self.network[bank_id] = bank.connections
    
    def run_simulation(self, steps: int, shocks: Optional[Dict[int, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Run the simulation for a specified number of steps.
        
        Parameters
        ----------
        steps : int
            Number of simulation steps
        shocks : dict, optional
            Dictionary mapping time steps to shock parameters
            
        Returns
        -------
        list
            History of system states
        """
        for step in range(steps):
            self.time += 1
            
            # Apply any scheduled shocks
            if shocks and self.time in shocks:
                self.apply_external_shock(shocks[self.time])
            
            # Run interbank lending simulation
            self.simulate_interbank_lending()
            
            # Record system state
            self.record_state()
            
        return self.history
    
    def record_state(self) -> None:
        """
        Record the current state of the system.
        """
        state = {
            "time": self.time,
            "bank_states": {bank_id: bank.state.copy() for bank_id, bank in self.banks.items()},
            "network": self.network.copy(),
            "system_indicators": self.system_indicators.copy()
        }
        
        self.history.append(state)
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the adjacency matrix of the current network.
        
        Returns
        -------
        numpy.ndarray
            Adjacency matrix
        """
        # Get list of bank IDs
        bank_ids = list(self.banks.keys())
        n = len(bank_ids)
        
        # Create empty adjacency matrix
        adj_matrix = np.zeros((n, n))
        
        # Fill with connection weights
        for i, source in enumerate(bank_ids):
            for j, target in enumerate(bank_ids):
                if target in self.banks[source].connections:
                    adj_matrix[i, j] = self.banks[source].connections[target]
        
        return adj_matrix
    
    def get_bank_metrics(self) -> pd.DataFrame:
        """
        Get current metrics for all banks.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing bank metrics
        """
        data = []
        
        for bank_id, bank in self.banks.items():
            row = {"bank_id": bank_id, "time": self.time}
            row.update(bank.state)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system-wide metrics.
        
        Returns
        -------
        dict
            Dictionary of system metrics
        """
        metrics = self.system_indicators.copy()
        
        # Add derived metrics
        
        # Average CET1 ratio
        cet1_values = [bank.state.get('CET1_ratio', 0) for bank in self.banks.values()]
        metrics['avg_CET1_ratio'] = sum(cet1_values) / len(cet1_values) if cet1_values else 0
        
        # Average LCR
        lcr_values = [bank.state.get('LCR', 0) for bank in self.banks.values()]
        metrics['avg_LCR'] = sum(lcr_values) / len(lcr_values) if lcr_values else 0
        
        # Network density
        adj_matrix = self.get_adjacency_matrix()
        n = adj_matrix.shape[0]
        if n > 1:
            metrics['network_density'] = np.count_nonzero(adj_matrix) / (n * (n - 1))
        else:
            metrics['network_density'] = 0
        
        return metrics
