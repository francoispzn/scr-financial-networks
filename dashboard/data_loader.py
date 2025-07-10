"""
Data loading bridge between the scr_financial data pipeline and the dashboard.

Uses DataPreprocessor / EBACollector / ECBCollector / MarketDataCollector
to build the dicts that BankingSystemSimulation expects.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scr_financial.data.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

# Full bank universe supported by the collectors
ALL_BANKS: List[str] = [
    "DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
    "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA",
]

BANK_LABELS: Dict[str, str] = {
    "DE_DBK":  "Deutsche Bank",
    "FR_BNP":  "BNP Paribas",
    "ES_SAN":  "Santander",
    "IT_UCG":  "UniCredit",
    "NL_ING":  "ING Group",
    "SE_NDA":  "Nordea",
    "CH_UBS":  "UBS",
    "UK_BARC": "Barclays",
    "UK_HSBC": "HSBC",
    "FR_ACA":  "Crédit Agricole",
}

BANK_COUNTRIES: Dict[str, str] = {
    "DE_DBK": "DE", "FR_BNP": "FR", "ES_SAN": "ES", "IT_UCG": "IT",
    "NL_ING": "NL", "SE_NDA": "SE", "CH_UBS": "CH", "UK_BARC": "GB",
    "UK_HSBC": "GB", "FR_ACA": "FR",
}


def load_simulation_inputs(
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    bank_list: Optional[List[str]] = None,
    snapshot_date: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Load bank data, network data, and system indicators from the data pipeline.

    Parameters
    ----------
    start_date, end_date : str
        Date range for data collection ('YYYY-MM-DD').
    bank_list : list of str, optional
        Bank IDs to include. Defaults to ALL_BANKS.
    snapshot_date : str, optional
        Date to take the snapshot at. Defaults to end_date.

    Returns
    -------
    bank_data : dict[bank_id -> state dict]
    network_data : dict[bank_id -> {target_id: weight}]
    system_indicators : dict
    """
    banks = bank_list or ALL_BANKS
    snap = pd.to_datetime(snapshot_date or end_date)

    logger.info("Loading data %s → %s  snapshot=%s  banks=%s",
                start_date, end_date, snap.date(), banks)

    preprocessor = DataPreprocessor(start_date, end_date, bank_list=banks)

    # ── Node data ────────────────────────────────────────────────────────────
    preprocessor.load_bank_node_data({
        "solvency":  "EBA_transparency",
        "liquidity": "EBA_aggregated",
    })

    # ── Edge data ────────────────────────────────────────────────────────────
    preprocessor.load_interbank_exposures(source="ECB_TARGET2")

    # ── System indicators ────────────────────────────────────────────────────
    preprocessor.load_system_indicators()

    # ── Extract snapshot ─────────────────────────────────────────────────────
    tp = preprocessor.get_data_for_timepoint(snap)

    bank_data = _build_bank_data(tp, banks)
    network_data = _build_network_data(tp, banks)
    system_indicators = _build_system_indicators(tp, preprocessor)

    logger.info("Loaded %d banks, %d network edges, %d system indicators",
                len(bank_data),
                sum(len(v) for v in network_data.values()),
                len(system_indicators))

    return bank_data, network_data, system_indicators


# ── Private helpers ──────────────────────────────────────────────────────────

def _build_bank_data(
    tp: Dict[str, Any],
    banks: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Merge solvency + liquidity node data into per-bank state dicts."""
    bank_data: Dict[str, Dict[str, Any]] = {}

    # Solvency DataFrame (CET1, total_assets, …)
    sol_df: Optional[pd.DataFrame] = tp["node_data"].get("solvency")
    # Liquidity DataFrame (LCR, NSFR, …)
    liq_df: Optional[pd.DataFrame] = tp["node_data"].get("liquidity")

    for bank_id in banks:
        state: Dict[str, Any] = {}

        if sol_df is not None and not (
            isinstance(sol_df, pd.DataFrame) and sol_df.empty
        ):
            row = (
                sol_df[sol_df["bank_id"] == bank_id]
                if isinstance(sol_df, pd.DataFrame)
                else None
            )
            if row is not None and not row.empty:
                r = row.iloc[0]
                state["CET1_ratio"] = float(r.get("CET1_ratio", 12.0))
                state["total_assets"] = float(r.get("total_assets", 1e12))
                state["risk_weighted_assets"] = float(
                    r.get("risk_weighted_assets", 4e11)
                )

        if liq_df is not None and not (
            isinstance(liq_df, pd.DataFrame) and liq_df.empty
        ):
            row = (
                liq_df[liq_df["bank_id"] == bank_id]
                if isinstance(liq_df, pd.DataFrame)
                else None
            )
            if row is not None and not row.empty:
                r = row.iloc[0]
                state["LCR"] = float(r.get("LCR", 130.0))
                state["NSFR"] = float(r.get("NSFR", 110.0))
                state["liquid_assets"] = float(r.get("liquid_assets", 1e11))

        # Provide sensible defaults for anything still missing
        state.setdefault("CET1_ratio", 12.0)
        state.setdefault("LCR", 130.0)
        state.setdefault("NSFR", 110.0)
        state.setdefault("total_assets", 1e12)
        state.setdefault("cash", state["total_assets"] * 0.12)
        state.setdefault("interbank_assets", state["total_assets"] * 0.18)
        state.setdefault("interbank_liabilities", state["total_assets"] * 0.16)

        bank_data[bank_id] = state

    return bank_data


def _build_network_data(
    tp: Dict[str, Any],
    banks: List[str],
) -> Dict[str, Dict[str, float]]:
    """Convert TARGET2 edge DataFrame into adjacency dict."""
    network: Dict[str, Dict[str, float]] = {b: {} for b in banks}
    bank_set = set(banks)

    edge_df = tp["edge_data"].get("interbank_exposures")
    if edge_df is None or (isinstance(edge_df, pd.DataFrame) and edge_df.empty):
        return network

    for _, row in edge_df.iterrows():
        src = row.get("source", "")
        tgt = row.get("target", "")
        w = float(row.get("weight", 0))
        if src in bank_set and tgt in bank_set and src != tgt and w > 0:
            network[src][tgt] = w

    return network


def _build_system_indicators(
    tp: Dict[str, Any],
    preprocessor: DataPreprocessor,
) -> Dict[str, Any]:
    """Extract CISS and funding stress from system data."""
    indicators: Dict[str, Any] = {}

    ciss_data = tp["system_data"].get("CISS")
    if ciss_data is not None:
        if isinstance(ciss_data, pd.DataFrame) and not ciss_data.empty:
            indicators["CISS"] = float(ciss_data["CISS"].iloc[-1])
        elif isinstance(ciss_data, pd.Series):
            indicators["CISS"] = float(ciss_data.get("CISS", 0.2))
        else:
            indicators["CISS"] = 0.2
    else:
        indicators["CISS"] = 0.2

    fs_data = tp["system_data"].get("funding_stress")
    if fs_data is not None:
        if isinstance(fs_data, pd.DataFrame) and not fs_data.empty:
            spread = fs_data["LIBOR_OIS_spread"].iloc[-1]
            indicators["funding_stress"] = min(1.0, float(spread) / 2.0)
        else:
            indicators["funding_stress"] = 0.1
    else:
        indicators["funding_stress"] = 0.1

    return indicators
