"""
Global simulation state for the dashboard.

Builds the BankingSystemSimulation from the live data pipeline
(EBACollector → ECBCollector → DataPreprocessor) rather than any
hard-coded demo values.  Thread safety is achieved via a module-level lock.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import numpy as np

from scr_financial.abm.simulation import BankingSystemSimulation
from scr_financial.network.coarse_graining import SpectralCoarseGraining
from scr_financial.network.spectral import (
    compute_laplacian,
    eigendecomposition,
    find_spectral_gap,
    compute_diffusion_distance,
    analyze_spectral_properties,
)
from .data_loader import (
    load_simulation_inputs,
    BANK_LABELS,
    BANK_COUNTRIES,
    ALL_BANKS,
)
from .data_api import build_simulation_inputs_from_api

_lock = threading.Lock()

# Spectral cache — invalidated whenever the adjacency matrix changes
_spectral_cache: Dict[str, Any] = {}
_spectral_cache_key: str = ""

# Current configuration (mutable via reload_data())
_config: Dict[str, Any] = {
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    "bank_list": ALL_BANKS,
    "snapshot_date": None,
}

# Cached raw inputs so reset() can replay without hitting the pipeline again
_cached_bank_data: Dict[str, Any] = {}
_cached_network_data: Dict[str, Dict[str, float]] = {}
_cached_system_indicators: Dict[str, Any] = {}

_sim: BankingSystemSimulation


def _build_simulation(
    bank_data: Dict[str, Any],
    network_data: Dict[str, Dict[str, float]],
    system_indicators: Dict[str, Any],
) -> BankingSystemSimulation:
    return BankingSystemSimulation(
        bank_data=copy.deepcopy(bank_data),
        network_data=copy.deepcopy(network_data),
        system_indicators=system_indicators.copy(),
        stochastic=True,
    )


_data_source: str = "EBA"  # tracks which data source was used

def _initialise() -> None:
    """Load data: try live market APIs first, fall back to EBA pipeline."""
    global _sim, _cached_bank_data, _cached_network_data, _cached_system_indicators
    global _data_source

    # Try real market data first
    try:
        bd, nd, si = build_simulation_inputs_from_api(
            bank_ids=_config["bank_list"],
        )
        _data_source = "API"
        logger.info("Initialised from live market APIs (yfinance + ECB).")
    except Exception as exc:
        logger.warning("Live API fetch failed (%s); falling back to EBA pipeline.", exc)
        bd, nd, si = load_simulation_inputs(
            start_date=_config["start_date"],
            end_date=_config["end_date"],
            bank_list=_config["bank_list"],
            snapshot_date=_config.get("snapshot_date"),
        )
        _data_source = "EBA"

    _cached_bank_data = bd
    _cached_network_data = nd
    _cached_system_indicators = si
    _sim = _build_simulation(bd, nd, si)


def get_data_source() -> str:
    """Return 'API' or 'EBA' depending on how data was loaded."""
    return _data_source


_initialise()


# ── Public API ───────────────────────────────────────────────────────────────

def get_config() -> Dict[str, Any]:
    return _config.copy()


def reload_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    bank_list: Optional[List[str]] = None,
    snapshot_date: Optional[str] = None,
) -> None:
    """Re-fetch data from the pipeline with updated parameters."""
    global _sim, _cached_bank_data, _cached_network_data, _cached_system_indicators
    with _lock:
        if start_date:
            _config["start_date"] = start_date
        if end_date:
            _config["end_date"] = end_date
        if bank_list:
            _config["bank_list"] = bank_list
        if snapshot_date is not None:
            _config["snapshot_date"] = snapshot_date

        bd, nd, si = load_simulation_inputs(
            start_date=_config["start_date"],
            end_date=_config["end_date"],
            bank_list=_config["bank_list"],
            snapshot_date=_config.get("snapshot_date"),
        )
        _cached_bank_data = bd
        _cached_network_data = nd
        _cached_system_indicators = si
        _sim = _build_simulation(bd, nd, si)
        _spectral_cache.clear()
        _spectral_cache_key = ""


def load_from_data(
    bank_data: Dict[str, Any],
    network_data: Dict[str, Dict[str, float]],
    system_indicators: Dict[str, Any],
) -> None:
    """Replace simulation state with externally-fetched data (e.g. from data_api)."""
    global _sim, _cached_bank_data, _cached_network_data, _cached_system_indicators
    global _spectral_cache, _spectral_cache_key
    with _lock:
        _cached_bank_data = bank_data
        _cached_network_data = network_data
        _cached_system_indicators = system_indicators
        _sim = _build_simulation(bank_data, network_data, system_indicators)
        _spectral_cache = {}
        _spectral_cache_key = ""


def reset_simulation() -> None:
    """Reset the ABM to the loaded data snapshot without re-fetching."""
    global _sim, _spectral_cache, _spectral_cache_key, _scg_cache, _scg_cache_key
    with _lock:
        _sim = _build_simulation(
            _cached_bank_data, _cached_network_data, _cached_system_indicators
        )
        _spectral_cache = {}
        _spectral_cache_key = ""
        _scg_cache = {}
        _scg_cache_key = ""


def get_simulation() -> BankingSystemSimulation:
    return _sim


def run_steps(steps: int, shocks: Optional[Dict[int, Any]] = None) -> List[Dict]:
    with _lock:
        return _sim.run_simulation(steps, shocks=shocks)


def apply_shock(shock_params: Dict[str, Any]) -> None:
    with _lock:
        _sim.apply_external_shock(shock_params)


def apply_shock_and_record(shock_params: Dict[str, Any]) -> List[Dict]:
    """Apply shock, record state immediately, return full history."""
    with _lock:
        _sim.apply_external_shock(shock_params)
        _sim.record_state()
        return _sim.history


def apply_llm_bank_data(bank_data: Dict[str, Any]) -> None:
    """Overwrite bank states with data fetched by the LLM."""
    with _lock:
        for bank_id, fields in bank_data.items():
            if bank_id in _sim.banks:
                _sim.banks[bank_id].state.update(fields)


# ── Spectral helpers ─────────────────────────────────────────────────────────

def get_spectral_data() -> Dict[str, Any]:
    global _spectral_cache, _spectral_cache_key
    adj = _sim.get_adjacency_matrix()
    cache_key = hashlib.md5(adj.tobytes()).hexdigest()
    if cache_key == _spectral_cache_key and _spectral_cache:
        return _spectral_cache

    adj_sym = (adj + adj.T) / 2.0
    L = compute_laplacian(adj_sym, normalized=True)
    eigenvalues, eigenvectors = eigendecomposition(L)
    gap_idx, gap_size = find_spectral_gap(eigenvalues)
    props = analyze_spectral_properties(eigenvalues, eigenvectors)
    diff_dist = compute_diffusion_distance(L, t=1.0)

    result = {
        "bank_ids": list(_sim.banks.keys()),
        "eigenvalues": eigenvalues.tolist(),
        "gap_index": int(gap_idx),
        "gap_size": float(gap_size),
        "algebraic_connectivity": float(props["algebraic_connectivity"]),
        "spectral_radius": float(props["spectral_radius"]),
        "participation_ratios": props["participation_ratios"].tolist(),
        "diffusion_distance": diff_dist.tolist(),
    }
    _spectral_cache_key = cache_key
    _spectral_cache = result
    return result


def get_network_graph_data() -> Dict[str, Any]:
    bank_ids = list(_sim.banks.keys())

    nodes = []
    for bid in bank_ids:
        state = _sim.banks[bid].state
        nodes.append({
            "id": bid,
            "label": BANK_LABELS.get(bid, bid),
            "country": BANK_COUNTRIES.get(bid, ""),
            "CET1_ratio": state.get("CET1_ratio", 0.0),
            "LCR": state.get("LCR", 0.0),
            "NSFR": state.get("NSFR", 0.0),
            "total_assets": state.get("total_assets", 1e9),
            "solvent": _sim.banks[bid].assess_solvency(),
            "liquid": _sim.banks[bid].assess_liquidity(),
        })

    edges = []
    for source, targets in _sim.network.items():
        for target, weight in targets.items():
            if weight > 0:
                edges.append({"source": source, "target": target, "weight": weight})

    return {"nodes": nodes, "edges": edges}


# ── SCG helpers ─────────────────────────────────────────────────────────────

_scg_cache: Dict[str, Any] = {}
_scg_cache_key: str = ""


def get_coarse_grained_data() -> Dict[str, Any]:
    """Run spectral coarse-graining on the current adjacency and return results."""
    global _scg_cache, _scg_cache_key
    adj = _sim.get_adjacency_matrix()
    cache_key = hashlib.md5(adj.tobytes()).hexdigest()
    if cache_key == _scg_cache_key and _scg_cache:
        return _scg_cache

    bank_ids = list(_sim.banks.keys())
    scg = SpectralCoarseGraining.from_adjacency(adj, bank_ids)

    # Run pipeline
    scg.coarse_grain()
    scg.rescale()
    clusters = scg.identify_clusters()
    diffusion_errors = scg.compare_diffusion_dynamics(time_steps=15)

    # Original eigenvalues for comparison
    orig_evals = scg.network_builder.eigenvalues.tolist()

    # CG eigenvalues
    import scipy.linalg as la
    cg_evals, _ = la.eigh(scg.coarse_grained_laplacian)

    # CG adjacency as dense
    cg_adj = scg.coarse_grained_adjacency

    # Cluster assignments
    cluster_map = {bank_ids[i]: int(clusters[i]) for i in range(len(bank_ids))}

    result = {
        "bank_ids": bank_ids,
        "original_eigenvalues": orig_evals,
        "cg_eigenvalues": cg_evals.tolist(),
        "clusters": cluster_map,
        "n_clusters": int(clusters.max() + 1),
        "diffusion_errors": diffusion_errors,
        "cg_adjacency": cg_adj.tolist(),
        "cg_error": float(scg.compute_coarse_graining_error()),
    }
    _scg_cache_key = cache_key
    _scg_cache = result
    return result
