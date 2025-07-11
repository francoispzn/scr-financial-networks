"""
Simplified implementations of established multi-channel ABM frameworks
for benchmarking against our 3-channel SCR ABM.

References
----------
Poledna, S., Molina-Borboa, J. L., Martinez-Jaramillo, S., van der Leij, M.,
    & Thurner, S. (2015). The multi-layer network nature of systemic risk and
    its implications for the costs of financial crises. *Journal of Financial
    Stability*, 20, 70-81.

Montagna, M. & Kok, C. (2016). Multi-layered interbank model for assessing
    systemic risk. *Kiel Working Paper No. 1873 / ECB Working Paper*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OU helper (lightweight, avoids coupling to bank_agent internals)
# ---------------------------------------------------------------------------

_OU_DEFAULTS: Dict[str, Tuple[float, float, float, float]] = {
    # key: (theta, sigma, clamp_lo, clamp_hi)
    "CET1_ratio": (0.05, 0.25, 0.0, 30.0),
    "LCR":        (0.04, 1.5,  20.0, 300.0),
    "NSFR":       (0.04, 0.8,  30.0, 250.0),
}


def _ou_step(
    x: float,
    mu: float,
    rng: np.random.Generator,
    theta: float = 0.05,
    sigma: float = 0.25,
    lo: float = 0.0,
    hi: float = 30.0,
    dt: float = 1.0,
) -> float:
    """Single Ornstein-Uhlenbeck step: dx = theta*(mu-x)*dt + sigma*sqrt(dt)*N(0,1)."""
    dx = theta * (mu - x) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
    return float(np.clip(x + dx, lo, hi))


# ---------------------------------------------------------------------------
# Shared bank state container
# ---------------------------------------------------------------------------

@dataclass
class _BankState:
    """Lightweight mutable bank state used by both literature ABMs."""

    bank_id: str
    CET1_ratio: float
    LCR: float
    NSFR: float
    total_assets: float
    cash: float
    interbank_assets: float
    defaulted: bool = False

    # OU long-run means (set once at init)
    _mu: Dict[str, float] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, bank_id: str, d: Dict[str, Any]) -> "_BankState":
        mu = {k: float(d.get(k, 0.0)) for k in _OU_DEFAULTS}
        return cls(
            bank_id=bank_id,
            CET1_ratio=float(d.get("CET1_ratio", 14.0)),
            LCR=float(d.get("LCR", 130.0)),
            NSFR=float(d.get("NSFR", 110.0)),
            total_assets=float(d.get("total_assets", 1e9)),
            cash=float(d.get("cash", 1e8)),
            interbank_assets=float(d.get("interbank_assets", 5e7)),
            _mu=mu,
        )

    def is_defaulted(self) -> bool:
        if self.defaulted:
            return True
        return self.CET1_ratio < 0 or self.cash < 0

    def freeze(self) -> None:
        self.defaulted = True

    def evolve_ou(self, rng: np.random.Generator, dt: float = 1.0) -> None:
        if self.defaulted:
            return
        for key, (theta, sigma, lo, hi) in _OU_DEFAULTS.items():
            x = getattr(self, key)
            mu = self._mu.get(key, x)
            setattr(self, key, _ou_step(x, mu, rng, theta, sigma, lo, hi, dt))


# ============================================================================
# Poledna et al. (2015) — Multi-layer contagion
# ============================================================================

class PolednaABM:
    """Simplified Poledna et al. (2015) multi-layer ABM.

    The original model uses 4 distinct exposure layers.  We approximate this
    by splitting the single adjacency matrix into two synthetic layers based
    on edge weight (strong vs. weak) and applying layer-specific LGD rates.

    Parameters
    ----------
    bank_data : dict
        ``{bank_id: {CET1_ratio, LCR, NSFR, total_assets, cash, ...}}``.
    adjacency : numpy.ndarray
        Square adjacency / correlation matrix (N x N), same ordering as
        ``sorted(bank_data.keys())``.
    strong_threshold : float
        Quantile threshold (0-1) above which an edge is considered "strong".
    lgd_strong : float
        Loss-given-default for the strong-exposure layer.
    lgd_weak : float
        Loss-given-default for the weak-exposure layer.
    """

    def __init__(
        self,
        bank_data: Dict[str, Dict[str, Any]],
        adjacency: np.ndarray,
        strong_threshold: float = 0.5,
        lgd_strong: float = 0.45,
        lgd_weak: float = 0.25,
    ) -> None:
        self.bank_ids: List[str] = sorted(bank_data.keys())
        self._id2idx = {bid: i for i, bid in enumerate(self.bank_ids)}
        self.n = len(self.bank_ids)

        # Validate adjacency shape
        if adjacency.shape != (self.n, self.n):
            raise ValueError(
                f"adjacency shape {adjacency.shape} does not match "
                f"number of banks ({self.n})."
            )

        # Split into two layers
        nonzero_weights = adjacency[adjacency > 0]
        if len(nonzero_weights) == 0:
            cutoff = 0.0
        else:
            cutoff = float(np.quantile(nonzero_weights, strong_threshold))

        self.layer_strong = np.where(adjacency >= cutoff, adjacency, 0.0)
        self.layer_weak = np.where(
            (adjacency > 0) & (adjacency < cutoff), adjacency, 0.0
        )

        self.lgd_strong = lgd_strong
        self.lgd_weak = lgd_weak

        # Bank states
        self.banks: List[_BankState] = [
            _BankState.from_dict(bid, bank_data[bid]) for bid in self.bank_ids
        ]

        # Tracking
        self._results: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []

    # ---- internal helpers --------------------------------------------------

    def _apply_shock(self, shock: Optional[Dict[str, Any]]) -> None:
        if shock is None:
            return
        for bid, delta in shock.items():
            if bid not in self._id2idx:
                continue
            bank = self.banks[self._id2idx[bid]]
            if bank.defaulted:
                continue
            for key, val in delta.items():
                if hasattr(bank, key):
                    setattr(bank, key, getattr(bank, key) + val)

    def _cascade(self) -> int:
        """Run iterative multi-layer cascade.  Returns cascade depth."""
        depth = 0
        newly_defaulted = {
            i for i, b in enumerate(self.banks)
            if b.is_defaulted() and not b.defaulted
        }

        while newly_defaulted:
            depth += 1
            for i in newly_defaulted:
                self.banks[i].freeze()
                logger.debug(
                    "PolednaABM: bank %s defaulted (cascade depth %d)",
                    self.bank_ids[i], depth,
                )

            next_round: set = set()
            for i in newly_defaulted:
                for j in range(self.n):
                    if self.banks[j].defaulted or j == i:
                        continue
                    # Exposure of j to i (j lent to i)
                    exp_strong = self.layer_strong[j, i]
                    exp_weak = self.layer_weak[j, i]
                    loss = exp_strong * self.lgd_strong + exp_weak * self.lgd_weak
                    if loss <= 0:
                        continue
                    bj = self.banks[j]
                    # Credit loss on interbank assets
                    actual = min(loss, bj.interbank_assets)
                    bj.interbank_assets -= actual
                    rwa = bj.total_assets * 0.35
                    if rwa > 0:
                        bj.CET1_ratio -= (actual / rwa) * 100
                    bj.cash -= actual * 0.10
                    if bj.is_defaulted():
                        next_round.add(j)
            newly_defaulted = next_round

        return depth

    def _snapshot(self) -> Dict[str, Any]:
        cet1s = [b.CET1_ratio for b in self.banks if not b.defaulted]
        return {
            "n_defaults": sum(1 for b in self.banks if b.defaulted),
            "avg_cet1": float(np.mean(cet1s)) if cet1s else 0.0,
            "min_cet1": float(np.min(cet1s)) if cet1s else 0.0,
        }

    # ---- public API --------------------------------------------------------

    def run(
        self,
        n_steps: int,
        shock: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        """Run the multi-layer ABM simulation.

        Parameters
        ----------
        n_steps : int
            Number of simulation time steps.
        shock : dict, optional
            ``{bank_id: {attribute: delta_value}}`` applied at t=1.
        seed : int
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        max_depth = 0

        for t in range(1, n_steps + 1):
            # Apply shock at the first step
            if t == 1 and shock is not None:
                self._apply_shock(shock)

            # OU evolution
            for b in self.banks:
                b.evolve_ou(rng)

            # Multi-layer cascade
            depth = self._cascade()
            max_depth = max(max_depth, depth)

            snap = self._snapshot()
            snap["time"] = t
            snap["cascade_depth"] = depth
            self._history.append(snap)

        final = self._snapshot()
        final["cascade_depth"] = max_depth
        self._results = final

    def get_results(self) -> Dict[str, Any]:
        """Return summary dict: ``{n_defaults, avg_cet1, min_cet1, cascade_depth}``."""
        return dict(self._results)


# ============================================================================
# Montagna & Kok (2016) — Credit losses + fire-sale with endogenous price
# ============================================================================

class MontagnaKokABM:
    """Simplified Montagna & Kok (2016) ABM with endogenous price impact.

    Contagion occurs through two channels:
    1. **Credit losses** identical to our Channel 1 (counterparty default).
    2. **Fire-sale price impact**: when a bank defaults, it liquidates assets,
       depressing the market price by ``(defaulted_assets / total_market_assets)
       * price_impact_factor``.  All surviving banks suffer mark-to-market
       losses on their total-asset book.

    Parameters
    ----------
    bank_data : dict
        ``{bank_id: {CET1_ratio, LCR, NSFR, total_assets, cash, ...}}``.
    adjacency : numpy.ndarray
        Square adjacency / correlation matrix (N x N).
    price_impact_factor : float
        Elasticity of asset price to aggregate fire-sale volume.
    lgd : float
        Loss-given-default for the credit channel.
    """

    def __init__(
        self,
        bank_data: Dict[str, Dict[str, Any]],
        adjacency: np.ndarray,
        price_impact_factor: float = 0.1,
        lgd: float = 0.40,
    ) -> None:
        self.bank_ids: List[str] = sorted(bank_data.keys())
        self._id2idx = {bid: i for i, bid in enumerate(self.bank_ids)}
        self.n = len(self.bank_ids)
        self.adj = np.array(adjacency, dtype=float)
        self.price_impact_factor = price_impact_factor
        self.lgd = lgd

        self.banks: List[_BankState] = [
            _BankState.from_dict(bid, bank_data[bid]) for bid in self.bank_ids
        ]

        self._results: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []

    # ---- internal helpers --------------------------------------------------

    def _apply_shock(self, shock: Optional[Dict[str, Any]]) -> None:
        if shock is None:
            return
        for bid, delta in shock.items():
            if bid not in self._id2idx:
                continue
            bank = self.banks[self._id2idx[bid]]
            if bank.defaulted:
                continue
            for key, val in delta.items():
                if hasattr(bank, key):
                    setattr(bank, key, getattr(bank, key) + val)

    def _cascade(self) -> int:
        """Run credit-loss cascade + fire-sale price impact.  Returns cascade depth."""
        depth = 0
        newly_defaulted = {
            i for i, b in enumerate(self.banks)
            if b.is_defaulted() and not b.defaulted
        }

        while newly_defaulted:
            depth += 1

            # --- Phase A: freeze defaulted banks ---------------------------
            for i in newly_defaulted:
                self.banks[i].freeze()
                logger.debug(
                    "MontagnaKokABM: bank %s defaulted (cascade depth %d)",
                    self.bank_ids[i], depth,
                )

            # --- Phase B: credit losses (Channel 1 analogue) ---------------
            next_round: set = set()
            for i in newly_defaulted:
                for j in range(self.n):
                    if self.banks[j].defaulted or j == i:
                        continue
                    exposure = self.adj[j, i]
                    if exposure <= 0:
                        continue
                    loss = exposure * self.lgd
                    bj = self.banks[j]
                    actual = min(loss, bj.interbank_assets)
                    bj.interbank_assets -= actual
                    rwa = bj.total_assets * 0.35
                    if rwa > 0:
                        bj.CET1_ratio -= (actual / rwa) * 100
                    bj.cash -= actual * 0.10

            # --- Phase C: fire-sale price impact ----------------------------
            # Defaulted banks liquidate; price drops proportional to volume
            total_market_assets = sum(
                b.total_assets for b in self.banks if not b.defaulted
            )
            if total_market_assets > 0:
                defaulted_assets = sum(
                    self.banks[i].total_assets for i in newly_defaulted
                )
                price_drop = (
                    defaulted_assets / total_market_assets
                ) * self.price_impact_factor

                # Mark-to-market losses on all survivors
                for b in self.banks:
                    if b.defaulted:
                        continue
                    mtm_loss = b.total_assets * price_drop
                    b.total_assets -= mtm_loss
                    rwa = b.total_assets * 0.35
                    if rwa > 0:
                        b.CET1_ratio -= (mtm_loss / rwa) * 100

            # Check for new defaults after both channels
            for j in range(self.n):
                if not self.banks[j].defaulted and self.banks[j].is_defaulted():
                    next_round.add(j)

            newly_defaulted = next_round

        return depth

    def _snapshot(self) -> Dict[str, Any]:
        cet1s = [b.CET1_ratio for b in self.banks if not b.defaulted]
        return {
            "n_defaults": sum(1 for b in self.banks if b.defaulted),
            "avg_cet1": float(np.mean(cet1s)) if cet1s else 0.0,
            "min_cet1": float(np.min(cet1s)) if cet1s else 0.0,
        }

    # ---- public API --------------------------------------------------------

    def run(
        self,
        n_steps: int,
        shock: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        """Run the credit + fire-sale ABM simulation.

        Parameters
        ----------
        n_steps : int
            Number of simulation time steps.
        shock : dict, optional
            ``{bank_id: {attribute: delta_value}}`` applied at t=1.
        seed : int
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        max_depth = 0

        for t in range(1, n_steps + 1):
            if t == 1 and shock is not None:
                self._apply_shock(shock)

            # OU evolution
            for b in self.banks:
                b.evolve_ou(rng)

            # Credit + fire-sale cascade
            depth = self._cascade()
            max_depth = max(max_depth, depth)

            snap = self._snapshot()
            snap["time"] = t
            snap["cascade_depth"] = depth
            self._history.append(snap)

        final = self._snapshot()
        final["cascade_depth"] = max_depth
        self._results = final

    def get_results(self) -> Dict[str, Any]:
        """Return summary dict: ``{n_defaults, avg_cet1, min_cet1, cascade_depth}``."""
        return dict(self._results)


# ============================================================================
# Comparison runner
# ============================================================================

def compare_abm_frameworks(
    bank_data: Dict[str, Dict[str, Any]],
    adjacency: np.ndarray,
    shock_scenario: Dict[str, Any],
    n_steps: int = 20,
    n_runs: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all three ABM frameworks on identical inputs and compare outcomes.

    Frameworks compared:
    1. **SCR 3-channel** (our ``BankingSystemSimulation``)
    2. **Poledna et al. (2015)** multi-layer contagion
    3. **Montagna & Kok (2016)** credit + fire-sale with endogenous price

    Parameters
    ----------
    bank_data : dict
        ``{bank_id: {CET1_ratio, LCR, NSFR, total_assets, cash, ...}}``.
    adjacency : numpy.ndarray
        Square adjacency matrix (N x N).
    shock_scenario : dict
        ``{bank_id: {attribute: delta_value}}``.
    n_steps : int
        Simulation horizon per run.
    n_runs : int
        Number of Monte-Carlo repetitions (different RNG seeds).
    seed : int
        Base random seed (each run uses ``seed + run_index``).

    Returns
    -------
    pandas.DataFrame
        Columns: ``framework, mean_defaults, std_defaults, mean_cet1,
        std_cet1, mean_min_cet1, cascade_depth``.
    """
    bank_ids = sorted(bank_data.keys())
    n_banks = len(bank_ids)

    # Build network_data dict from adjacency for our ABM
    network_data: Dict[str, Dict[str, float]] = {}
    for i, src in enumerate(bank_ids):
        conns: Dict[str, float] = {}
        for j, tgt in enumerate(bank_ids):
            if adjacency[i, j] > 0 and i != j:
                conns[tgt] = float(adjacency[i, j])
        network_data[src] = conns

    # Accumulators  {framework_name: list of result dicts}
    accum: Dict[str, List[Dict[str, Any]]] = {
        "SCR 3-channel": [],
        "Poledna (2015)": [],
        "Montagna-Kok (2016)": [],
    }

    for run_idx in range(n_runs):
        run_seed = seed + run_idx

        # ---- SCR 3-channel (ours) -----------------------------------------
        from .simulation import BankingSystemSimulation  # local import to avoid circular

        sim = BankingSystemSimulation(
            bank_data=bank_data,
            network_data=network_data,
            system_indicators={"CISS": 0.5},
            stochastic=True,
            seed=run_seed,
        )
        sim.run_simulation(steps=n_steps, shocks={1: shock_scenario})
        n_def = sum(1 for b in sim.banks.values() if b._defaulted)
        cet1s = [
            b.state.get("CET1_ratio", 0.0)
            for b in sim.banks.values()
            if not b._defaulted
        ]
        accum["SCR 3-channel"].append({
            "n_defaults": n_def,
            "avg_cet1": float(np.mean(cet1s)) if cet1s else 0.0,
            "min_cet1": float(np.min(cet1s)) if cet1s else 0.0,
            "cascade_depth": n_steps,  # our ABM runs all channels each step
        })

        # ---- Poledna (2015) -----------------------------------------------
        pol = PolednaABM(bank_data, adjacency)
        pol.run(n_steps=n_steps, shock=shock_scenario, seed=run_seed)
        accum["Poledna (2015)"].append(pol.get_results())

        # ---- Montagna-Kok (2016) ------------------------------------------
        mk = MontagnaKokABM(bank_data, adjacency)
        mk.run(n_steps=n_steps, shock=shock_scenario, seed=run_seed)
        accum["Montagna-Kok (2016)"].append(mk.get_results())

    # Aggregate
    rows = []
    for fw_name, results_list in accum.items():
        defaults = [r["n_defaults"] for r in results_list]
        cet1_vals = [r["avg_cet1"] for r in results_list]
        min_cet1_vals = [r["min_cet1"] for r in results_list]
        depths = [r["cascade_depth"] for r in results_list]
        rows.append({
            "framework": fw_name,
            "mean_defaults": float(np.mean(defaults)),
            "std_defaults": float(np.std(defaults)),
            "mean_cet1": float(np.mean(cet1_vals)),
            "std_cet1": float(np.std(cet1_vals)),
            "mean_min_cet1": float(np.mean(min_cet1_vals)),
            "cascade_depth": float(np.mean(depths)),
        })

    return pd.DataFrame(rows)
