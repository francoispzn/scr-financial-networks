"""
Banking system simulation for financial network analysis.

This module provides the BankingSystemSimulation class for simulating
interbank lending and contagion dynamics.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .bank_agent import BankAgent

logger = logging.getLogger(__name__)


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
        system_indicators: Dict[str, Any],
        stochastic: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize the simulation with bank data, network data, and system indicators."""
        if not bank_data:
            raise ValueError("bank_data must not be empty.")
        self._initial_bank_data: Dict[str, Dict[str, Any]] = {
            bank_id: data.copy() for bank_id, data in bank_data.items()
        }
        self._initial_network_data: Dict[str, Dict[str, float]] = {
            bank_id: connections.copy()
            for bank_id, connections in network_data.items()
        }
        self.stochastic = stochastic
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.banks: Dict[str, BankAgent] = {}
        self.network = network_data.copy()
        self.system_indicators = system_indicators.copy()
        self.time = 0
        self.history: List[Dict[str, Any]] = []

        # Initialize bank agents
        for bank_id, data in bank_data.items():
            self.banks[bank_id] = BankAgent(bank_id, data, stochastic=stochastic)

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

                        # Cap to available cash so lender cash never goes below zero
                        available_cash = self.banks[lender_id].state.get("cash", float("inf"))
                        actual_amount = min(actual_amount, available_cash)

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
    
    # ── Stochastic dynamics ────────────────────────────────────────────────

    def _evolve_bank_ratios(self) -> None:
        """Apply one OU step to every bank's ratios + small CISS noise."""
        for bank in self.banks.values():
            bank.evolve_ratios(self.rng)
        # Small mean-reverting noise on CISS (σ=0.01)
        ciss = self.system_indicators.get("CISS", 0.5)
        ciss_mu = self.system_indicators.get("_ciss_mu", ciss)
        if "_ciss_mu" not in self.system_indicators:
            self.system_indicators["_ciss_mu"] = ciss
        dciss = 0.03 * (ciss_mu - ciss) + 0.01 * self.rng.standard_normal()
        self.system_indicators["CISS"] = float(np.clip(ciss + dciss, 0.0, 1.0))

    # ── Channel 1: Credit contagion ──────────────────────────────────────

    def _propagate_defaults(self) -> None:
        """Iterative contagion cascade: defaulted banks impose losses on counterparties."""
        newly_defaulted = {
            bid for bid, bank in self.banks.items()
            if bank.is_defaulted() and not bank._defaulted
        }
        if not newly_defaulted:
            return

        while newly_defaulted:
            # Freeze all newly-defaulted banks
            for bid in newly_defaulted:
                self.banks[bid].freeze()
                logger.warning("Bank %s defaulted at t=%d (credit channel)", bid, self.time)

            next_round: set = set()
            for bid in newly_defaulted:
                # Find counterparties that have interbank exposure to this bank
                for other_id, other_bank in self.banks.items():
                    if other_bank._defaulted or other_id == bid:
                        continue
                    exposure = other_bank.connections.get(bid, 0.0)
                    if exposure > 0:
                        loss = exposure * BankAgent._LGD
                        other_bank.apply_counterparty_loss(loss)
                        if other_bank.is_defaulted():
                            next_round.add(other_id)
            newly_defaulted = next_round

    # ── Channel 2: Funding liquidity stress ───────────────────────────────

    _FUNDING_LCR_THRESHOLD = 100.0   # LCR below which funding stress triggers
    _FUNDING_SENSITIVITY = 0.05      # Sensitivity parameter for withdrawal probability
    _FUNDING_WITHDRAWAL_RATE = 0.03  # Max fraction of cash drained per stressed counterparty

    def _propagate_funding_stress(self) -> None:
        """Funding liquidity channel: banks with low LCR face counterparty withdrawals.

        When a bank's LCR drops below the regulatory minimum (100%), its
        counterparties withdraw short-term funding proportionally to the gap.
        This drains cash and can trigger a liquidity spiral.
        """
        for bid, bank in self.banks.items():
            if bank._defaulted:
                continue
            lcr = bank.state.get("LCR", 130.0)
            if lcr >= self._FUNDING_LCR_THRESHOLD:
                continue

            # Withdrawal probability increases with LCR gap (sigmoid)
            lcr_gap = self._FUNDING_LCR_THRESHOLD - lcr
            withdrawal_prob = 1.0 / (1.0 + np.exp(-self._FUNDING_SENSITIVITY * lcr_gap))

            for other_id, other_bank in self.banks.items():
                if other_bank._defaulted or other_id == bid:
                    continue
                exposure = other_bank.connections.get(bid, 0.0)
                if exposure <= 0:
                    continue

                # Stochastic withdrawal decision
                if self.rng.random() < withdrawal_prob:
                    withdrawal = exposure * self._FUNDING_WITHDRAWAL_RATE
                    cash = bank.state.get("cash", 0.0)
                    bank.state["cash"] = cash - withdrawal

                    # LCR deteriorates further as liquid assets drain
                    total_assets = bank.state.get("total_assets", 1e9)
                    if total_assets > 0:
                        lcr_hit = (withdrawal / total_assets) * 100.0
                        bank.state["LCR"] = max(0.0, bank.state.get("LCR", 130.0) - lcr_hit)

                    if bank.is_defaulted() and not bank._defaulted:
                        bank.freeze()
                        logger.warning("Bank %s defaulted at t=%d (funding channel)", bid, self.time)
                        break  # This bank is done

    # ── Channel 3: Fire-sale / asset correlation ──────────────────────────

    _FIRE_SALE_HAIRCUT = 0.02   # Base haircut per correlated default
    _FIRE_SALE_AMPLIFIER = 1.5  # Amplification when multiple banks default simultaneously

    def _propagate_fire_sales(self) -> None:
        """Asset correlation channel: defaults cause mark-to-market losses on correlated assets.

        When banks default, surviving banks holding correlated assets suffer
        mark-to-market losses proportional to correlation strength and the
        number of simultaneous defaults. This captures the fire-sale externality.
        """
        defaulted_ids = [
            bid for bid, bank in self.banks.items() if bank._defaulted
        ]
        if not defaulted_ids:
            return

        # Amplification factor increases with simultaneous defaults
        n_defaults = len(defaulted_ids)
        amplifier = 1.0 + (n_defaults - 1) * (self._FIRE_SALE_AMPLIFIER - 1.0) / max(len(self.banks) - 1, 1)

        adj = self.get_adjacency_matrix()
        bank_ids = list(self.banks.keys())
        id_to_idx = {bid: i for i, bid in enumerate(bank_ids)}

        for bid, bank in self.banks.items():
            if bank._defaulted:
                continue
            idx = id_to_idx[bid]
            total_haircut = 0.0
            for def_id in defaulted_ids:
                def_idx = id_to_idx.get(def_id)
                if def_idx is None:
                    continue
                correlation = adj[idx, def_idx]
                if correlation > 0:
                    total_haircut += correlation * self._FIRE_SALE_HAIRCUT * amplifier

            if total_haircut > 0:
                # Cap haircut to prevent unrealistic total wipeouts
                total_haircut = min(total_haircut, 0.15)

                # Mark-to-market loss on total assets
                ta = bank.state.get("total_assets", 1e9)
                asset_loss = ta * total_haircut
                bank.state["total_assets"] = ta - asset_loss

                # CET1 hit proportional to loss
                rwa_est = ta * 0.35
                if rwa_est > 0:
                    bank.state["CET1_ratio"] = bank.state.get("CET1_ratio", 0.0) - (asset_loss / rwa_est) * 100

                if bank.is_defaulted() and not bank._defaulted:
                    bank.freeze()
                    logger.warning("Bank %s defaulted at t=%d (fire-sale channel)", bid, self.time)

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
        if steps <= 0:
            raise ValueError(f"steps must be a positive integer, got {steps}.")
        logger.info(
            "run_simulation starting: %d steps, %d banks.",
            steps,
            len(self.banks),
        )
        for step in range(steps):
            self.time += 1

            # Apply any scheduled shocks
            if shocks and self.time in shocks:
                logger.debug(
                    "Applying shock at time step %d: %s", self.time, shocks[self.time]
                )
                self.apply_external_shock(shocks[self.time])

            # Evolve stochastic ratios
            if self.stochastic:
                self._evolve_bank_ratios()

            # Run interbank lending simulation
            self.simulate_interbank_lending()

            # 3-channel contagion cascade
            self._propagate_defaults()        # Channel 1: credit losses
            self._propagate_funding_stress()   # Channel 2: liquidity withdrawals
            self._propagate_fire_sales()       # Channel 3: asset correlation losses

            # Record system state
            self.record_state()

        logger.info("run_simulation finished at time %d.", self.time)
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
    
    def reset(self) -> None:
        """
        Reset the simulation to its initial state.

        Clears ``history``, resets ``time`` to 0, and re-initialises every
        bank's state from the ``bank_data`` passed at construction.
        Network connections are also restored from the original
        ``network_data``.
        """
        self.history = []
        self.time = 0
        self.rng = np.random.default_rng(self._seed)
        for bank_id, data in self._initial_bank_data.items():
            if bank_id in self.banks:
                self.banks[bank_id].state = data.copy()
                self.banks[bank_id].memory = []
                self.banks[bank_id].connections = {}
                self.banks[bank_id]._defaulted = False
                # Restore OU long-run means
                for key in BankAgent._OU_PARAMS:
                    if key in data:
                        self.banks[bank_id]._ou_mu[key] = float(data[key])
        # Restore network connections from the original network snapshot
        self.network = {
            bank_id: connections.copy()
            for bank_id, connections in self._initial_network_data.items()
        }
        for bank_id, connections in self._initial_network_data.items():
            if bank_id in self.banks:
                for target_id, strength in connections.items():
                    if target_id in self.banks:
                        self.banks[bank_id].connections[target_id] = strength
        logger.info("Simulation reset to initial state.")

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
        cet1_values = [bank.state.get("CET1_ratio", 0) for bank in self.banks.values()]
        metrics["avg_CET1_ratio"] = float(np.mean(cet1_values)) if cet1_values else 0.0

        # Average LCR
        lcr_values = [bank.state.get("LCR", 0) for bank in self.banks.values()]
        metrics["avg_LCR"] = float(np.mean(lcr_values)) if lcr_values else 0.0
        
        # Network density
        adj_matrix = self.get_adjacency_matrix()
        n = adj_matrix.shape[0]
        if n > 1:
            metrics['network_density'] = np.count_nonzero(adj_matrix) / (n * (n - 1))
        else:
            metrics['network_density'] = 0
        
        return metrics
