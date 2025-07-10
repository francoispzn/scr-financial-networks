"""
Rolling backtesting engine for SCG vs traditional risk metrics.

Loads historical EBA snapshots, builds networks per quarter,
computes SCG metrics alongside CoVaR/MES/Basel, and evaluates
which model best predicts future bank distress.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scr_financial.network.coarse_graining import SpectralCoarseGraining
from scr_financial.network.spectral import (
    compute_laplacian,
    eigendecomposition,
    find_spectral_gap,
    analyze_spectral_properties,
)
from scr_financial.risk.metrics import (
    compute_delta_covar,
    compute_mes,
    compute_var,
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Rolling backtest comparing SCG, CoVaR, MES, and Basel risk indicators."""

    def __init__(self) -> None:
        self.snapshots: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []

    # ── Data loading ──────────────────────────────────────────────────────

    def load_snapshots(
        self,
        start_year: int = 2020,
        end_year: int = 2024,
    ) -> int:
        """Load quarterly EBA snapshots via the data loader.

        Returns the number of snapshots loaded.
        """
        from dashboard.data_loader import load_simulation_inputs, ALL_BANKS

        self.snapshots = []
        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                month = (quarter - 1) * 3 + 1
                snap_date = f"{year}-{month:02d}-01"
                try:
                    bd, nd, si = load_simulation_inputs(
                        start_date=f"{year}-01-01",
                        end_date=f"{year}-12-31",
                        bank_list=ALL_BANKS,
                        snapshot_date=snap_date,
                    )
                    self.snapshots.append({
                        "date": snap_date,
                        "bank_data": bd,
                        "network_data": nd,
                        "system_indicators": si,
                    })
                except Exception as e:
                    logger.debug("Skipping snapshot %s: %s", snap_date, e)

        logger.info("Loaded %d quarterly snapshots (%d–%d)", len(self.snapshots), start_year, end_year)
        return len(self.snapshots)

    # ── Metric computation ────────────────────────────────────────────────

    @staticmethod
    def _spectral_metrics(bank_data: Dict, network_data: Dict) -> Dict[str, float]:
        """Compute SCG spectral metrics from a snapshot."""
        from scr_financial.abm.simulation import BankingSystemSimulation
        sim = BankingSystemSimulation(bank_data, network_data, {}, stochastic=False)
        adj = sim.get_adjacency_matrix()
        adj_sym = (adj + adj.T) / 2.0
        L = compute_laplacian(adj_sym, normalized=True)
        eigenvalues, eigenvectors = eigendecomposition(L)
        gap_idx, gap_size = find_spectral_gap(eigenvalues)
        props = analyze_spectral_properties(eigenvalues, eigenvectors)

        lam2 = float(props["algebraic_connectivity"])
        radius = float(props["spectral_radius"])
        scg_risk = 1.0 - (lam2 / radius) if radius > 0 else 1.0

        return {
            "lambda_2": lam2,
            "spectral_gap": float(gap_size),
            "spectral_radius": radius,
            "scg_risk": max(0.0, min(1.0, scg_risk)),
        }

    @staticmethod
    def _bank_metrics(bank_data: Dict) -> Dict[str, float]:
        """Compute bank-level risk metrics."""
        cet1s = [d.get("CET1_ratio", 10.0) for d in bank_data.values()]
        return {
            "avg_cet1": float(np.mean(cet1s)),
            "min_cet1": float(np.min(cet1s)),
            "n_stressed": sum(1 for c in cet1s if c < 8.0),
            "n_insolvent": sum(1 for c in cet1s if c < 4.5),
        }

    # ── Rolling backtest ──────────────────────────────────────────────────

    def run_rolling(self, window_quarters: int = 4, step: int = 1) -> pd.DataFrame:
        """Run a rolling backtest across loaded snapshots.

        For each window, computes risk metrics at time t and checks
        whether stress materialised at t+step.

        Returns a DataFrame with per-quarter metrics.
        """
        if len(self.snapshots) < window_quarters + step:
            raise ValueError(
                f"Need >= {window_quarters + step} snapshots, have {len(self.snapshots)}"
            )

        self.results = []
        for i in range(window_quarters, len(self.snapshots) - step + 1):
            current = self.snapshots[i]
            future = self.snapshots[min(i + step, len(self.snapshots) - 1)]

            # Spectral metrics at current time
            spec = self._spectral_metrics(current["bank_data"], current["network_data"])

            # Bank health at current time
            bm_now = self._bank_metrics(current["bank_data"])

            # Bank health at future time (what we're trying to predict)
            bm_future = self._bank_metrics(future["bank_data"])

            # CET1 returns over the lookback window
            bank_ids = list(current["bank_data"].keys())
            cet1_history = []
            for j in range(max(0, i - window_quarters), i + 1):
                snap = self.snapshots[j]
                cet1_history.append({
                    bid: snap["bank_data"].get(bid, {}).get("CET1_ratio", 10.0)
                    for bid in bank_ids
                })

            # Compute rolling returns and risk metrics
            avg_dcovar, avg_mes = 0.0, 0.0
            if len(cet1_history) > 2:
                bank_returns = {}
                for bid in bank_ids:
                    ts = [h[bid] for h in cet1_history]
                    bank_returns[bid] = np.diff(ts)

                all_rets = np.array(list(bank_returns.values()))
                if all_rets.shape[1] > 0:
                    sys_rets = np.mean(all_rets, axis=0)
                    dcovars = [compute_delta_covar(r, sys_rets) for r in bank_returns.values()]
                    mess = [compute_mes(r, sys_rets) for r in bank_returns.values()]
                    avg_dcovar = float(np.mean(dcovars))
                    avg_mes = float(np.mean(mess))

            # Future stress indicator (target)
            stress_increased = bm_future["n_stressed"] > bm_now["n_stressed"]

            self.results.append({
                "date": current["date"],
                "future_date": future["date"],
                # Risk indicators (predictors)
                "scg_risk": spec["scg_risk"],
                "lambda_2": spec["lambda_2"],
                "spectral_gap": spec["spectral_gap"],
                "delta_covar": avg_dcovar,
                "mes": avg_mes,
                "basel_stress": bm_now["n_stressed"],
                "avg_cet1": bm_now["avg_cet1"],
                # Target
                "future_stressed": bm_future["n_stressed"],
                "future_min_cet1": bm_future["min_cet1"],
                "stress_increased": stress_increased,
            })

        df = pd.DataFrame(self.results)
        logger.info("Backtest complete: %d evaluation points", len(df))
        return df

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """Evaluate predictive accuracy of each risk model.

        Returns a dict per model with correlation to future stress,
        and hit rate (did high risk precede stress increase?).
        """
        if not self.results:
            raise ValueError("Run run_rolling() first.")

        df = pd.DataFrame(self.results)
        target = df["future_stressed"].values
        stress_events = df["stress_increased"].values

        models = {
            "SCG Risk": df["scg_risk"].values,
            "ΔCoVaR": -df["delta_covar"].values,  # negate: more negative = more risk
            "MES": -df["mes"].values,
            "Basel Stress": df["basel_stress"].values,
        }

        results = {}
        for name, signal in models.items():
            # Correlation with future stress count
            if np.std(signal) > 1e-10 and np.std(target) > 1e-10:
                corr = float(np.corrcoef(signal, target)[0, 1])
            else:
                corr = 0.0

            # Hit rate: proportion of times top-quartile risk precedes stress
            threshold = np.percentile(signal, 75)
            high_risk = signal >= threshold
            if high_risk.sum() > 0:
                hit_rate = float(stress_events[high_risk].mean())
            else:
                hit_rate = 0.0

            results[name] = {
                "correlation_future_stress": corr,
                "hit_rate": hit_rate,
                "mean_signal": float(np.mean(signal)),
                "std_signal": float(np.std(signal)),
            }

        return results

    def to_dataframe(self) -> pd.DataFrame:
        """Export results as a DataFrame."""
        return pd.DataFrame(self.results)
