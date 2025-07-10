"""
Bank agent implementation for financial network simulations.

This module provides the BankAgent class which represents a bank in the
agent-based model and implements its decision-making behaviors.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .decision_models import DefaultDecisionModel

logger = logging.getLogger(__name__)

# Regulatory thresholds (Basel III / CRD IV)
_MIN_CET1_SOLVENCY = 4.5   # Minimum CET1 ratio (%) required for solvency
_MIN_LCR_LIQUIDITY = 100.0  # Minimum LCR (%) required for adequate liquidity
_LENDING_CAPACITY_FRACTION = 0.10   # Fraction of total assets usable as base lending capacity
_EXCESS_CAPITAL_NORMALISER = 4.0    # Divisor to scale excess CET1 into a capacity factor
_CET1_LENDING_THRESHOLD = 8.0       # CET1 % above which capital is considered "excess"


class BankAgent:
    """
    Represents a bank in the agent-based model.

    Parameters
    ----------
    bank_id : str
        Unique identifier for the bank.
    initial_state : dict
        Dictionary containing initial bank attributes.
    decision_model : object, optional
        Decision model to use for agent behavior, by default DefaultDecisionModel().
    stochastic : bool
        Enable Ornstein-Uhlenbeck noise on regulatory ratios each step.

    Attributes
    ----------
    id : str
        Bank identifier.
    state : dict
        Current state of the bank.
    connections : dict
        Dictionary of connections to other banks.
    memory : list
        List of past states (capped at ``memory_length`` entries).
    memory_length : int
        Maximum number of past states to remember (default 10).
    """

    # Ornstein-Uhlenbeck parameters per ratio:  (theta, sigma, clamp_lo, clamp_hi)
    _OU_PARAMS = {
        "CET1_ratio": (0.05, 0.25, 0.0, 30.0),
        "LCR":        (0.04, 1.5,  20.0, 300.0),
        "NSFR":       (0.04, 0.8,  30.0, 250.0),
    }
    # Loss-given-default rate applied to counterparty interbank assets
    _LGD = 0.40

    def __init__(
        self,
        bank_id: str,
        initial_state: Dict[str, Any],
        decision_model: Optional[object] = None,
        stochastic: bool = True,
    ):
        """Initialize a bank agent with an ID and initial state."""
        self.id = bank_id
        self.state = initial_state.copy()
        self.connections: Dict[str, float] = {}
        self.memory: List[Dict[str, Any]] = []
        self.memory_length = 10
        self.stochastic = stochastic
        self._defaulted = False

        # Store long-run means for OU reversion (snapshot at init)
        self._ou_mu: Dict[str, float] = {}
        for key in self._OU_PARAMS:
            if key in self.state:
                self._ou_mu[key] = float(self.state[key])

        # Set default decision model if none provided
        self.decision_model = decision_model if decision_model else DefaultDecisionModel()
    
    # ── Stochastic dynamics ────────────────────────────────────────────────

    def evolve_ratios(self, rng: np.random.Generator, dt: float = 1.0) -> None:
        """Apply one discrete Ornstein-Uhlenbeck step to CET1, LCR, NSFR.

        dx = theta*(mu - x)*dt + sigma*sqrt(dt)*N(0,1)
        """
        if not self.stochastic or self._defaulted:
            return
        for key, (theta, sigma, lo, hi) in self._OU_PARAMS.items():
            x = self.state.get(key)
            mu = self._ou_mu.get(key)
            if x is None or mu is None:
                continue
            dx = theta * (mu - x) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
            self.state[key] = float(np.clip(x + dx, lo, hi))

    def apply_counterparty_loss(self, loss_amount: float) -> None:
        """Apply a credit loss from a defaulted counterparty.

        Reduces interbank_assets, CET1 proportionally, and cash.
        """
        if loss_amount <= 0 or self._defaulted:
            return
        # Reduce interbank assets
        ib = self.state.get("interbank_assets", 0.0)
        actual_loss = min(loss_amount, ib)
        self.state["interbank_assets"] = ib - actual_loss

        # CET1 hit: loss / estimated RWA (assume RWA ≈ 35% of total_assets)
        total_assets = self.state.get("total_assets", 1e9)
        rwa_est = total_assets * 0.35
        if rwa_est > 0:
            self.state["CET1_ratio"] = self.state.get("CET1_ratio", 0.0) - (actual_loss / rwa_est) * 100

        # Cash fire-sale cost (10% of loss)
        cash = self.state.get("cash", 0.0)
        self.state["cash"] = cash - actual_loss * 0.10

    def is_defaulted(self) -> bool:
        """Bank is in default if CET1 < 0 or cash < 0."""
        if self._defaulted:
            return True
        return self.state.get("CET1_ratio", 10.0) < 0 or self.state.get("cash", 1.0) < 0

    def freeze(self) -> None:
        """Mark bank as defaulted — no further lending/evolution."""
        self._defaulted = True

    # ── State management ──────────────────────────────────────────────────

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
        
        # Update state (merge new values; add new keys if they don't exist yet)
        self.state.update(new_state)
    
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
            base_capacity = 0.0  # No total_assets known — cannot fabricate a capacity
        
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
        action = self.decision_model.decide_lending_action(
            self,
            potential_borrowers,
            market_sentiment,
        )
        logger.debug("BankAgent %s lending action: %s", self.id, action.get("action"))
        return action
    
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
        action = self.decision_model.decide_borrowing_action(
            self,
            potential_lenders,
            market_sentiment,
        )
        logger.debug("BankAgent %s borrowing action: %s", self.id, action.get("action"))
        return action
    
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

    def __repr__(self) -> str:
        return (
            f"BankAgent(id={self.id!r},"
            f" solvent={self.assess_solvency()},"
            f" liquid={self.assess_liquidity()})"
        )
