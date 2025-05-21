"""
Decision models for bank agents in financial network simulations.

This module provides various decision-making models that determine how
bank agents behave in different scenarios.
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np


class DefaultDecisionModel:
    """
    Default decision model for bank agents.
    
    This model implements simple rule-based decision making for lending,
    borrowing, and responding to shocks.
    """
    
    def decide_lending_action(
        self, 
        bank, 
        potential_borrowers: Dict[str, Dict[str, Any]], 
        market_sentiment: float
    ) -> Dict[str, Any]:
        """
        Decide lending actions based on current state and market conditions.
        
        Parameters
        ----------
        bank : BankAgent
            The bank agent making the decision
        potential_borrowers : dict
            Dictionary mapping bank IDs to their states
        market_sentiment : float
            Market sentiment indicator (0-1 scale, 1 being positive)
            
        Returns
        -------
        dict
            Dictionary containing lending decisions
        """
        if not bank.assess_liquidity():
            # Bank has liquidity issues, reduce lending
            return {"action": "reduce_lending", "amount": 0}
            
        lending_capacity = bank.calculate_lending_capacity()
        
        # Adjust for market sentiment (0-1 scale, 1 being positive)
        adjusted_capacity = lending_capacity * market_sentiment
        
        # Allocate lending to potential borrowers based on their creditworthiness
        allocations = {}
        for borrower_id, borrower_data in potential_borrowers.items():
            # Skip self
            if borrower_id == bank.id:
                continue
                
            # Skip if borrower has solvency issues
            if borrower_data.get('CET1_ratio', 0) < 6.0:  # Higher threshold for lending
                continue
                
            # Allocate based on existing relationship and borrower health
            relationship_strength = bank.get_connection_strength(borrower_id)
            borrower_health = min(1, borrower_data.get('CET1_ratio', 0) / 15)  # Normalize to 0-1
            
            # Allocation formula
            base_allocation = adjusted_capacity * 0.1  # Base allocation is 10% of capacity
            relationship_factor = 1 + 2 * relationship_strength  # Existing relationships get up to 3x
            health_factor = borrower_health
            
            allocation = base_allocation * relationship_factor * health_factor
            
            if allocation > 0:
                allocations[borrower_id] = allocation
                
        return {"action": "lend", "allocations": allocations}
    
    def decide_borrowing_action(
        self, 
        bank, 
        potential_lenders: Dict[str, Dict[str, Any]], 
        market_sentiment: float
    ) -> Dict[str, Any]:
        """
        Decide borrowing actions based on current state and market conditions.
        
        Parameters
        ----------
        bank : BankAgent
            The bank agent making the decision
        potential_lenders : dict
            Dictionary mapping bank IDs to their states
        market_sentiment : float
            Market sentiment indicator (0-1 scale, 1 being positive)
            
        Returns
        -------
        dict
            Dictionary containing borrowing decisions
        """
        # Determine borrowing need based on liquidity
        if bank.state.get('LCR', 120) > 120:
            # No need to borrow if LCR is high
            return {"action": "no_borrowing"}
        
        # Calculate borrowing need
        if 'LCR' in bank.state:
            # Target LCR of 120%
            lcr_gap = max(0, 120 - bank.state['LCR'])
            borrowing_need = lcr_gap * bank.state.get('net_cash_outflows', 1e9) / 100
        else:
            # Default borrowing need
            borrowing_need = 0.05 * bank.state.get('total_assets', 1e9)
        
        # Adjust for market sentiment
        adjusted_need = borrowing_need * (2 - market_sentiment)  # Borrow more in bad times
        
        # Identify potential sources
        sources = {}
        for lender_id, lender_data in potential_lenders.items():
            # Skip self
            if lender_id == bank.id:
                continue
                
            # Skip if lender has liquidity issues
            if lender_data.get('LCR', 0) < 110:  # Only borrow from liquid banks
                continue
                
            # Allocate based on existing relationship and lender health
            relationship_strength = bank.get_connection_strength(lender_id)
            lender_health = min(1, lender_data.get('LCR', 0) / 150)  # Normalize to 0-1
            
            # Allocation formula
            base_request = adjusted_need * 0.2  # Request 20% of need from each lender
            relationship_factor = 1 + relationship_strength  # Prefer existing relationships
            health_factor = lender_health
            
            request = base_request * relationship_factor * health_factor
            
            if request > 0:
                sources[lender_id] = request
        
        return {"action": "borrow", "sources": sources, "total_need": borrowing_need}
    
    def respond_to_shock(self, bank, shock_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Respond to an external shock.
        
        Parameters
        ----------
        bank : BankAgent
            The bank agent responding to the shock
        shock_params : dict
            Dictionary containing shock parameters
            
        Returns
        -------
        dict
            Dictionary containing response actions
        """
        response = {"actions": []}
        
        # Check if solvency is affected
        if 'CET1_ratio' in shock_params:
            if shock_params['CET1_ratio'] < 0:
                # Negative shock to capital ratio
                # Reduce lending to preserve capital
                reduction_targets = {}
                
                # Reduce connections proportionally to shock magnitude
                reduction_factor = min(0.8, abs(shock_params['CET1_ratio']) / 10)
                
                for target_id, strength in bank.connections.items():
                    reduction_targets[target_id] = reduction_factor
                
                response["actions"].append({
                    "type": "reduce_lending",
                    "targets": reduction_targets,
                    "reason": "capital_preservation"
                })
        
        # Check if liquidity is affected
        if 'LCR' in shock_params:
            if shock_params['LCR'] < 0:
                # Negative shock to liquidity
                # Reduce lending and seek additional funding
                reduction_targets = {}
                
                # Reduce connections proportionally to shock magnitude
                reduction_factor = min(0.9, abs(shock_params['LCR']) / 50)
                
                for target_id, strength in bank.connections.items():
                    reduction_targets[target_id] = reduction_factor
                
                response["actions"].append({
                    "type": "reduce_lending",
                    "targets": reduction_targets,
                    "reason": "liquidity_preservation"
                })
                
                response["actions"].append({
                    "type": "seek_funding",
                    "amount": bank.state.get('total_assets', 1e9) * 0.05,
                    "reason": "liquidity_restoration"
                })
        
        return response


class StressDecisionModel(DefaultDecisionModel):
    """
    Decision model for bank agents during stress scenarios.
    
    This model extends the default model with more conservative
    behavior during market stress.
    """
    
    def decide_lending_action(
        self, 
        bank, 
        potential_borrowers: Dict[str, Dict[str, Any]], 
        market_sentiment: float
    ) -> Dict[str, Any]:
        """
        More conservative lending during stress.
        """
        # If market sentiment is very low, reduce lending drastically
        if market_sentiment < 0.3:
            return {"action": "reduce_lending", "amount": 0}
        
        # Otherwise use default model but with reduced capacity
        decision = super().decide_lending_action(bank, potential_borrowers, market_sentiment)
        
        if decision["action"] == "lend" and "allocations" in decision:
            # Reduce all allocations by a stress factor
            stress_factor = market_sentiment  # Linear reduction based on sentiment
            for borrower_id in decision["allocations"]:
                decision["allocations"][borrower_id] *= stress_factor
        
        return decision
    
    def respond_to_shock(self, bank, shock_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        More aggressive response to shocks.
        """
        # Get base response
        response = super().respond_to_shock(bank, shock_params)
        
        # Amplify reduction factors
        for action in response["actions"]:
            if action["type"] == "reduce_lending" and "targets" in action:
                for target_id in action["targets"]:
                    # Increase reduction by 50%
                    action["targets"][target_id] = min(1.0, action["targets"][target_id] * 1.5)
        
        return response


class LearningDecisionModel(DefaultDecisionModel):
    """
    Decision model that learns from past interactions.
    
    This model extends the default model with learning capabilities
    based on past interactions with other banks.
    """
    
    def __init__(self):
        """Initialize the learning decision model."""
        super().__init__()
        self.interaction_history = {}
        self.learning_rate = 0.1
    
    def decide_lending_action(
        self, 
        bank, 
        potential_borrowers: Dict[str, Dict[str, Any]], 
        market_sentiment: float
    ) -> Dict[str, Any]:
        """
        Lending decisions influenced by past interactions.
        """
        decision = super().decide_lending_action(bank, potential_borrowers, market_sentiment)
        
        if decision["action"] == "lend" and "allocations" in decision:
            # Adjust allocations based on past interactions
            for borrower_id in list(decision["allocations"].keys()):
                # Get interaction score (-1 to 1, where 1 is positive)
                score = self.get_interaction_score(bank.id, borrower_id)
                
                # Adjust allocation based on score
                adjustment_factor = 1 + score  # Range: 0 to 2
                decision["allocations"][borrower_id] *= adjustment_factor
                
                # Remove allocation if factor is too low
                if adjustment_factor < 0.2:
                    del decision["allocations"][borrower_id]
        
        return decision
    
    def update_interaction_score(self, lender_id: str, borrower_id: str, outcome: float) -> None:
        """
        Update interaction score based on outcome.
        
        Parameters
        ----------
        lender_id : str
            ID of the lending bank
        borrower_id : str
            ID of the borrowing bank
        outcome : float
            Outcome score (-1 to 1, where 1 is positive)
        """
        key = (lender_id, borrower_id)
        
        if key not in self.interaction_history:
            self.interaction_history[key] = 0
        
        # Update score with learning rate
        self.interaction_history[key] = (1 - self.learning_rate) * self.interaction_history[key] + self.learning_rate * outcome
    
    def get_interaction_score(self, lender_id: str, borrower_id: str) -> float:
        """
        Get interaction score between two banks.
        
        Parameters
        ----------
        lender_id : str
            ID of the lending bank
        borrower_id : str
            ID of the borrowing bank
            
        Returns
        -------
        float
            Interaction score (-1 to 1, where 1 is positive)
        """
        key = (lender_id, borrower_id)
        return self.interaction_history.get(key, 0)
