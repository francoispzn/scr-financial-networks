"""
Global simulation state for the dashboard.

Holds a single BankingSystemSimulation instance that is shared across
the Dash callbacks and the FastAPI routers. Thread safety is achieved via
a module-level lock; the simulation is not designed for concurrent mutations.
"""

import copy
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from scr_financial.abm.simulation import BankingSystemSimulation
from scr_financial.network.spectral import (
    compute_laplacian,
    eigendecomposition,
    find_spectral_gap,
    compute_diffusion_distance,
    analyze_spectral_properties,
)
from .demo_data import BANK_DATA, NETWORK_DATA, SYSTEM_INDICATORS

_lock = threading.Lock()


def _build_simulation() -> BankingSystemSimulation:
    """Instantiate a fresh simulation from demo data."""
    # Strip non-numeric 'name'/'country' fields that BankAgent doesn't use
    bank_data = {
        bank_id: {k: v for k, v in state.items() if k not in ("name", "country")}
        for bank_id, state in BANK_DATA.items()
    }
    return BankingSystemSimulation(
        bank_data=bank_data,
        network_data=copy.deepcopy(NETWORK_DATA),
        system_indicators=SYSTEM_INDICATORS.copy(),
    )


# ── Module-level singleton ───────────────────────────────────────────────────

_sim: BankingSystemSimulation = _build_simulation()


# ── Public helpers ───────────────────────────────────────────────────────────

def get_simulation() -> BankingSystemSimulation:
    """Return the current simulation instance (read-only)."""
    return _sim


def reset_simulation() -> None:
    """Reset the simulation to its initial state."""
    global _sim
    with _lock:
        _sim = _build_simulation()


def run_steps(steps: int, shocks: Optional[Dict[int, Any]] = None) -> List[Dict]:
    """Run *steps* steps; return the history list."""
    with _lock:
        return _sim.run_simulation(steps, shocks=shocks)


def apply_shock(shock_params: Dict[str, Any]) -> None:
    """Apply an external shock to the simulation."""
    with _lock:
        _sim.apply_external_shock(shock_params)


# ── Spectral helpers (always computed on current adjacency) ─────────────────

def get_spectral_data() -> Dict[str, Any]:
    """Return eigenvalues, eigenvectors, gap, and diffusion distance."""
    adj = _sim.get_adjacency_matrix()
    # Symmetrise for spectral analysis (undirected view)
    adj_sym = (adj + adj.T) / 2.0

    L = compute_laplacian(adj_sym, normalized=True)
    eigenvalues, eigenvectors = eigendecomposition(L)
    gap_idx, gap_size = find_spectral_gap(eigenvalues)
    props = analyze_spectral_properties(eigenvalues, eigenvectors)
    diff_dist = compute_diffusion_distance(L, t=1.0)

    return {
        "bank_ids": list(_sim.banks.keys()),
        "eigenvalues": eigenvalues.tolist(),
        "gap_index": int(gap_idx),
        "gap_size": float(gap_size),
        "algebraic_connectivity": float(props["algebraic_connectivity"]),
        "spectral_radius": float(props["spectral_radius"]),
        "participation_ratios": props["participation_ratios"].tolist(),
        "diffusion_distance": diff_dist.tolist(),
    }


def get_network_graph_data() -> Dict[str, Any]:
    """Return node and edge data suitable for Plotly network graph."""
    sim = _sim
    bank_ids = list(sim.banks.keys())

    nodes = []
    for bid in bank_ids:
        state = sim.banks[bid].state
        meta = BANK_DATA.get(bid, {})
        nodes.append({
            "id": bid,
            "label": meta.get("name", bid),
            "country": meta.get("country", ""),
            "CET1_ratio": state.get("CET1_ratio", 0.0),
            "LCR": state.get("LCR", 0.0),
            "total_assets": state.get("total_assets", 1e9),
            "solvent": sim.banks[bid].assess_solvency(),
            "liquid": sim.banks[bid].assess_liquidity(),
        })

    edges = []
    for source, targets in sim.network.items():
        for target, weight in targets.items():
            if weight > 0:
                edges.append({"source": source, "target": target, "weight": weight})

    return {"nodes": nodes, "edges": edges}
