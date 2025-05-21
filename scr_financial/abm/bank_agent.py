"""
Bank agent implementation for financial network simulations.

This module provides the BankAgent class which represents a bank in the
agent-based model and implements its decision-making behaviors.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable

from .decision_models import DefaultDecisionModel


class BankAgent:
    """
    Represents a bank in the agent-based model.
    
    Parameters
    ----------
    bank_id : str
        Unique identifier for the bank
    initial_state : dict
        Dictionary containing initial bank attributes
    decision_model : object, optional
        Decision model to use for agent behavior, by default DefaultDecisionModel()
    
    Attributes
    ----------
    id : str
        Bank identifier
    state : dict
        Current state of the bank
    connections : dict
        Dictionary of connections to other banks
    memory : list
        List of past states
    memory_length : int
        Maximum number of past states to remember
    """
    
    def __init__(
        self, 
        bank_id: str, 
        initial_state: Dict[str, Any],
        decision_model: Optional[object] = None
    ):
        """Initialize a bank agent with an ID and initial state."""
        self.id = bank_id
        self.state = initial_state.copy()
        self.connections = {}
        self.memory = []
        self.memory_length = 10
        
        # Set default decision model if none provided
        self.decision_model = decision_model if decision_model else DefaultDecisionModel()
    
    def update_state(self, new_state: Dict[str, Any]) -> None:
        """
        Update the bank's state with new information.
        
        Parameters
        ----------
        new_state : dict
            Dictionary containing updated attributes
        """
        # Store current state in memory
        self.memory.append(self.state.copy())
        
        # Limit memory length
        if len(self.memory) > self.memory_length:
            self.memory = self.memory[-self.memory_length:]
        
        # Update state
        for key, value in new_state.items():
            if key in self.state:
                self.state[key] = value
            else:
                self.state[key] = value
    
    def assess_solvency(self) -> bool:
        """
        Determine if the bank is solvent based on capital ratios.
        
        Returns
        -------
        bool
            True if the bank is solvent, False otherwise
        """
        if 'CET1_ratio' in self.state:
            return self.state['CET1_ratio'] >= 4.5  # Minimum CET1 requirement
        return True  # Default to solvent if no data
    
    def assess_liquidity(self) -> bool:
        """
        Determine if the bank has adequate liquidity.
        
        Returns
        -------
        bool
            True if the bank has adequate liquidity, False otherwise
        """
        if 'LCR' in self.state:
            return self.state['LCR'] >= 100  # LCR requirement
        return True  # Default to liquid if no data
    
    def calculate_lending_capacity(self) -> float:
        """
        Calculate how much the bank can lend to other banks.
        
        Returns
        -------
        float
            Lending capacity in currency units
        """
        # Simple model: lending capacity is a function of excess liquidity and capital
        
        # Base capacity from total assets
        if 'total_assets' in self.state:
            base_capacity = self.state['total_assets'] * 0.1  # 10% of assets
        else:
            base_capacity = 1e9  # Default value
        
        # Adjust for liquidity
        liquidity_factor = 1.0
        if 'LCR' in self.state:
            excess_liquidity = max(0, self.state['LCR'] - 100) / 100
            liquidity_factor = 1.0 + excess_liquidity
        
        # Adjust for capital
        capital_factor = 1.0
        if 'CET1_ratio' in self.state:
            excess_capital = max(0, self.state['CET1_ratio'] - 8.0) / 4.0
            capital_factor = 1.0 + excess_capital
        
        return base_capacity * liquidity_factor * capital_factor
    
    def decide_lending_action(
        self, 
        potential_borrowers: Dict[str, Dict[str, Any]], 
        market_sentiment: float
    ) -> Dict[str, Any]:
        """
        Decide lending actions based on current state and market conditions.
        
        Parameters
        ----------
        potential_borrowers : dict
            Dictionary mapping bank IDs to their states
        market_sentiment : float
            Market sentiment indicator (0-1 scale, 1 being positive)
            
        Returns
        -------
        dict
            Dictionary containing lending decisions
        """
        # Use decision model to make lending decisions
        return self.decision_model.decide_lending_action(
            self, 
            potential_borrowers, 
            market_sentiment
        )
    
    def decide_borrowing_action(
        self, 
        potential_lenders: Dict[str, Dict[str, Any]], 
        market_sentiment: float
    ) -> Dict[str, Any]:
        """
        Decide borrowing actions based on current state and market conditions.
        
        Parameters
        ----------
        potential_lenders : dict
            Dictionary mapping bank IDs to their states
        market_sentiment : float
            Market sentiment indicator (0-1 scale, 1 being positive)
            
        Returns
        -------
        dict
            Dictionary containing borrowing decisions
        """
        # Use decision model to make borrowing decisions
        return self.decision_model.decide_borrowing_action(
            self, 
            potential_lenders, 
            market_sentiment
        )
    
    def respond_to_shock(self, shock_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Respond to an external shock.
        
        Parameters
        ----------
        shock_params : dict
            Dictionary containing shock parameters
            
        Returns
        -------
        dict
            Dictionary containing response actions
        """
        # Use decision model to determine response
        return self.decision_model.respond_to_shock(self, shock_params)
    
    def update_connections(self, bank_id: str, weight: float) -> None:
        """
        Update connection strength with another bank.
        
        Parameters
        ----------
        bank_id : str
            Identifier of the connected bank
        weight : float
            New connection weight
        """
        self.connections[bank_id] = weight
    
    def get_connection_strength(self, bank_id: str) -> float:
        """
        Get connection strength with another bank.
        
        Parameters
        ----------
        bank_id : str
            Identifier of the connected bank
            
        Returns
        -------
        float
            Connection strength
        """
        return self.connections.get(bank_id, 0.0)
