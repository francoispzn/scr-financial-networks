"""
Prediction helpers for the Evolution page.

Generates GNN training data from real daily market snapshots (yfinance),
trains the GNNPredictor (temporal GCN+LSTM), and builds SCG-vs-Basel comparison data.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from scr_financial.ml.gnn_predictor import GNNPredictor, TARGET_NAMES
from scr_financial.risk.metrics import compute_delta_covar, compute_mes
from . import simulation_state as sim_state

logger = logging.getLogger(__name__)


def generate_evolution_data(
    n_steps: int = 500,
    source: str = "market",
    corr_window: int = 60,
    stride: int = 1,
    progress_callback: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Generate graph snapshots for GNN training.

    Parameters
    ----------
    n_steps : int
        For 'abm' source: number of ABM steps.
        For 'market' source: ignored (uses all available daily data).
    source : str
        'market' — real daily data from yfinance (default, recommended)
        'abm' — stochastic ABM simulation
    corr_window : int
        Rolling correlation window for market data (trading days).
    stride : int
        Day stride for market snapshots (1=daily, 5=weekly).
    progress_callback : callable(current, total)
        For UI progress updates.
    """
    if source == "market":
        return _generate_from_market(corr_window, stride, progress_callback)
    return _generate_from_abm(n_steps, progress_callback)


def _generate_from_market(
    corr_window: int = 60,
    stride: int = 1,
    progress_callback: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Build daily graph snapshots from real market data."""
    from dashboard.data_api import build_daily_graph_snapshots

    snapshots = build_daily_graph_snapshots(
        lookback_years=3,
        corr_window=corr_window,
        min_corr=0.3,
        stride=stride,
        progress_callback=progress_callback,
    )
    if not snapshots:
        logger.warning("No market snapshots — falling back to ABM")
        return _generate_from_abm(500, progress_callback)

    # Add scalar features for dashboard chart compatibility
    for snap in snapshots:
        snap.setdefault("avg_cet1", 0.0)
        snap.setdefault("min_cet1", 0.0)
        snap.setdefault("ciss", 0.0)
        snap.setdefault("funding_stress", 0.0)
        snap.setdefault("n_stressed", 0)
    return snapshots


def _generate_from_abm(
    n_steps: int = 500,
    progress_callback: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Run stochastic ABM and collect graph snapshots."""
    sim = sim_state.get_simulation()
    sim.reset()

    snapshots: List[Dict[str, Any]] = []

    snap = GNNPredictor.extract_graph_snapshot(sim)
    snap["time"] = 0
    snap.update(GNNPredictor.extract_features(sim, lambda: _spectral_from_sim(sim)))
    snap["_bank_cet1"] = {bid: b.state.get("CET1_ratio", 10.0)
                          for bid, b in sim.banks.items()}
    snapshots.append(snap)

    for step in range(1, n_steps + 1):
        sim.run_simulation(1)
        snap = GNNPredictor.extract_graph_snapshot(sim)
        snap["time"] = step
        snap.update(GNNPredictor.extract_features(sim, lambda: _spectral_from_sim(sim)))
        snap["_bank_cet1"] = {bid: b.state.get("CET1_ratio", 10.0)
                              for bid, b in sim.banks.items()}
        snapshots.append(snap)
        if progress_callback and step % 50 == 0:
            progress_callback(step, n_steps)

    return snapshots


def _spectral_from_sim(sim) -> Dict[str, Any]:
    """Compute spectral data from a simulation snapshot."""
    from scr_financial.network.spectral import (
        compute_laplacian, eigendecomposition, find_spectral_gap,
        analyze_spectral_properties,
    )
    adj = sim.get_adjacency_matrix()
    adj_sym = (adj + adj.T) / 2.0
    L = compute_laplacian(adj_sym, normalized=True)
    eigenvalues, eigenvectors = eigendecomposition(L)
    gap_idx, gap_size = find_spectral_gap(eigenvalues)
    props = analyze_spectral_properties(eigenvalues, eigenvectors)
    return {
        "algebraic_connectivity": float(props["algebraic_connectivity"]),
        "gap_size": float(gap_size),
        "gap_index": int(gap_idx),
        "spectral_radius": float(props["spectral_radius"]),
    }


def train_predictor(
    snapshots: List[Dict[str, Any]],
    seq_len: int = 10,
    hidden_dim: int = 64,
    num_gat_layers: int = 3,
    num_lstm_layers: int = 2,
    epochs: int = 200,
    lr: float = 3e-3,
    dropout: float = 0.1,
    progress_callback: Optional[Callable] = None,
):
    """Train a GNNPredictor on graph snapshots.

    Returns (predictor, train_metrics, test_metrics).
    """
    predictor = GNNPredictor(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_gat_layers=num_gat_layers,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout,
    )
    predictor.train(snapshots, epochs=epochs, lr=lr, progress_callback=progress_callback)
    return predictor, predictor.train_metrics, predictor.test_metrics


def compute_scg_reconstruction_accuracy() -> Dict[str, Any]:
    """Run the full SCG pipeline on the current adjacency and return reconstruction accuracy."""
    from scr_financial.network.coarse_graining import SpectralCoarseGraining
    adj = sim_state.get_simulation().get_adjacency_matrix()
    bank_ids = list(sim_state.get_simulation().banks.keys())
    scg = SpectralCoarseGraining.from_adjacency(adj, bank_ids)
    scg.coarse_grain()
    scg.contract_vertices()
    scg.rescale()
    return scg.compute_reconstruction_accuracy(time_steps=15)


def build_scg_vs_basel_comparison(
    feature_history: List[Dict[str, Any]],
) -> Dict[str, List]:
    """Build SCG risk score vs Basel + CoVaR + MES per step."""
    times, scg_risk, basel_stress = [], [], []

    for row in feature_history:
        t = row.get("time", 0)
        lam2 = row.get("lambda_2", 0.001)
        radius = row.get("spectral_radius", 1.0)
        risk = 1.0 - (lam2 / radius) if radius > 0 else 1.0
        times.append(t)
        scg_risk.append(max(0.0, min(1.0, risk)))
        basel_stress.append(int(row.get("n_stressed", 0)))

    delta_covar_series, mes_series = [], []
    window = 20
    bank_ids = list(feature_history[0].get("_bank_cet1", {}).keys()) if feature_history else []
    if bank_ids and all("_bank_cet1" in r for r in feature_history):
        bank_cet1_ts = {bid: [r["_bank_cet1"].get(bid, 10.0) for r in feature_history]
                        for bid in bank_ids}
        bank_returns = {bid: np.diff(ts) for bid, ts in bank_cet1_ts.items()}
        for t_idx in range(len(feature_history)):
            if t_idx < window + 1:
                delta_covar_series.append(0.0)
                mes_series.append(0.0)
                continue
            ret_window = {bid: rets[t_idx - window:t_idx]
                          for bid, rets in bank_returns.items()
                          if t_idx <= len(rets)}
            if not ret_window or any(len(v) < window for v in ret_window.values()):
                delta_covar_series.append(0.0)
                mes_series.append(0.0)
                continue
            all_rets = np.array(list(ret_window.values()))
            sys_rets = np.mean(all_rets, axis=0)
            dcovars = [compute_delta_covar(rets, sys_rets) for rets in ret_window.values()]
            mess = [compute_mes(rets, sys_rets) for rets in ret_window.values()]
            delta_covar_series.append(float(min(1.0, max(0.0, -np.mean(dcovars) * 10))))
            mes_series.append(float(min(1.0, max(0.0, abs(np.mean(mess)) * 5))))
    else:
        delta_covar_series = [0.0] * len(times)
        mes_series = [0.0] * len(times)

    return {
        "time": times, "scg_risk": scg_risk, "basel_stress": basel_stress,
        "delta_covar": delta_covar_series, "mes": mes_series,
    }
