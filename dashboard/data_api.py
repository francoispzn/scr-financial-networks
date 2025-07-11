"""
Fast, API-based financial data fetcher for the SCR dashboard.

Data sources (no LLM required for any of these):

  yfinance     — bank stock prices, return correlations (→ adjacency matrix A),
                 market cap, balance sheet financials (total assets, equity)
  ECB SDW      — sovereign bond yields (IT, DE, FR, ES, NL, SE),
                 EUR/USD rate, ECB deposit facility rate
  FRED         — TED spread, VIX (systemic stress proxies)

Correlation-based edge weights
-------------------------------
As per the SCG proposal (§2.1), edges are defined by the Pearson correlation
of bank stock daily returns over a rolling window. This is real, daily data
that is:
  - Updated automatically every trading day
  - Directly computable without regulatory disclosures
  - Already used in the SCG literature (Mantegna 1999, Tumminello 2007)

All network fetches are parallelised via ThreadPoolExecutor for speed.
"""

from __future__ import annotations

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Bank universe ────────────────────────────────────────────────────────────
# Configurable via scr_financial.config.loader; defaults to eu_10.

from scr_financial.config.loader import load_universe

_default_universe = load_universe("eu_10")

# {bank_id: yahoo_ticker} — built from the config universe
BANK_TICKERS: Dict[str, str] = {
    b.id: b.ticker for b in _default_universe.banks
}
ALL_BANKS: List[str] = _default_universe.ids

# Sovereign yield ECB series (10Y) per country code
_ECB_YIELD_SERIES: Dict[str, str] = {
    "DE": "IRS/M.DE.L.L40.CI.0000.EUR.N.Z",
    "FR": "IRS/M.FR.L.L40.CI.0000.EUR.N.Z",
    "IT": "IRS/M.IT.L.L40.CI.0000.EUR.N.Z",
    "ES": "IRS/M.ES.L.L40.CI.0000.EUR.N.Z",
    "NL": "IRS/M.NL.L.L40.CI.0000.EUR.N.Z",
    "SE": "IRS/M.SE.L.L40.CI.0000.EUR.N.Z",
}
_ECB_BASE = "https://data-api.ecb.europa.eu/service/data"
_ECB_TIMEOUT = 10

# ── Helpers ──────────────────────────────────────────────────────────────────

def _ecb_csv(series_key: str, n_obs: int = 12) -> Optional[pd.DataFrame]:
    """Fetch a single ECB SDW series and return a DataFrame with TIME_PERIOD, OBS_VALUE."""
    url = f"{_ECB_BASE}/{series_key}?format=csvdata&lastNObservations={n_obs}"
    try:
        r = requests.get(url, timeout=_ECB_TIMEOUT,
                         headers={"Accept": "text/csv"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if "TIME_PERIOD" in df.columns and "OBS_VALUE" in df.columns:
            df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
            return df[["TIME_PERIOD", "OBS_VALUE"]].dropna()
        return None
    except Exception as exc:
        logger.debug("ECB fetch failed %s: %s", series_key, exc)
        return None


def _fetch_prices(bank_ids: List[str], period: str = "1y") -> pd.DataFrame:
    """Download adjusted close prices for a list of bank_ids in one yfinance call."""
    import yfinance as yf  # imported here to avoid hard dep at module level
    tickers = [BANK_TICKERS[b] for b in bank_ids if b in BANK_TICKERS]
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        prices = raw["Close"] if "Close" in raw else raw
        # Rename columns back to bank_ids
        rev = {v: k for k, v in BANK_TICKERS.items()}
        prices.columns = [rev.get(str(c), str(c)) for c in prices.columns]
        return prices
    except Exception as exc:
        logger.warning("yfinance price download failed: %s", exc)
        return pd.DataFrame()


# ── Core public functions ────────────────────────────────────────────────────

def fetch_correlation_adjacency(
    bank_ids: Optional[List[str]] = None,
    window_days: int = 252,
    min_corr: float = 0.3,
    pmfg: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Build the correlation-based adjacency matrix from daily bank stock returns.

    As per §2.1 of the SCG proposal: edges are Pearson correlations of returns.
    Weak edges (< min_corr) are removed (threshold filtering, §2.3).

    Parameters
    ----------
    bank_ids    : list of bank IDs to include (defaults to all 10)
    window_days : rolling return window in trading days
    min_corr    : threshold below which edges are zeroed out
    pmfg        : if True, apply Planar Maximally Filtered Graph (slower but cleaner)

    Returns
    -------
    dict  {source_id: {target_id: weight}}  — upper-triangular correlation weights
    """
    ids = bank_ids or ALL_BANKS

    prices = _fetch_prices(ids, period=f"{window_days + 50}d")
    if prices.empty:
        logger.warning("No price data — returning empty adjacency")
        return {b: {} for b in ids}

    returns = prices.pct_change().dropna()
    if len(returns) < 30:
        logger.warning("Insufficient return history (%d rows)", len(returns))
        return {b: {} for b in ids}

    # Use most recent window_days
    returns = returns.tail(window_days)
    corr = returns.corr()

    # Threshold filtering (§2.3)
    adj: Dict[str, Dict[str, float]] = {b: {} for b in ids}
    for i, src in enumerate(ids):
        for j, tgt in enumerate(ids):
            if i >= j:
                continue
            if src not in corr.index or tgt not in corr.index:
                continue
            w = float(corr.loc[src, tgt])
            if w >= min_corr:
                adj[src][tgt] = w
                adj[tgt][src] = w  # symmetric

    logger.info(
        "Correlation adjacency: %d banks, %d edges (min_corr=%.2f)",
        len(ids), sum(len(v) for v in adj.values()) // 2, min_corr,
    )
    return adj


def fetch_bank_market_features(
    bank_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch per-bank market and fundamental features from yfinance in parallel.

    Returns
    -------
    dict  {bank_id: {feature: value}}

    Features available:
      market_cap, total_assets, common_equity, roe, price_to_book, beta,
      shares_outstanding, latest_price, 1y_return, volatility_30d
    """
    import yfinance as yf
    ids = bank_ids or ALL_BANKS

    def _fetch_one(bid: str) -> Tuple[str, Dict[str, Any]]:
        ticker_sym = BANK_TICKERS.get(bid)
        if not ticker_sym:
            return bid, {}
        try:
            t = yf.Ticker(ticker_sym)
            info = t.info or {}

            feats: Dict[str, Any] = {}
            feats["market_cap"]    = info.get("marketCap")
            feats["roe"]           = info.get("returnOnEquity")
            feats["price_to_book"] = info.get("priceToBook")
            feats["beta"]          = info.get("beta")

            # Balance sheet — quarterly, most recent
            try:
                bs = t.quarterly_balance_sheet
                if bs is not None and not bs.empty:
                    latest = bs.iloc[:, 0]
                    feats["total_assets"]   = float(latest.get("Total Assets", np.nan))
                    feats["common_equity"]  = float(latest.get("Common Stock Equity", np.nan))
                    feats["total_debt"]     = float(latest.get("Total Debt", np.nan))
            except Exception:
                pass

            # 30-day volatility from price history
            try:
                hist = t.history(period="60d", progress=False)
                if not hist.empty and len(hist) >= 10:
                    rets = hist["Close"].pct_change().dropna()
                    feats["volatility_30d"] = float(rets.tail(30).std() * np.sqrt(252))
                    feats["latest_price"]   = float(hist["Close"].iloc[-1])
                    feats["1y_return"]      = float(
                        hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1
                        if len(hist) >= 200 else np.nan
                    )
            except Exception:
                pass

            return bid, feats
        except Exception as exc:
            logger.debug("yfinance fetch failed for %s: %s", bid, exc)
            return bid, {}

    results: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(len(ids), 6)) as pool:
        futures = {pool.submit(_fetch_one, bid): bid for bid in ids}
        for fut in as_completed(futures):
            bid, feats = fut.result()
            results[bid] = feats

    logger.info(
        "Market features fetched: %d banks, avg %.0f fields/bank",
        len(results),
        np.mean([len(v) for v in results.values()]) if results else 0,
    )
    return results


def fetch_sovereign_spreads() -> Dict[str, float]:
    """
    Fetch 10Y sovereign bond yields from ECB SDW and compute IT-DE spread
    as a systemic stress proxy.

    Returns
    -------
    dict  {country: latest_yield_pct, ..., 'IT_DE_spread': float, 'ES_DE_spread': float}
    """
    def _fetch_country(country: str, series: str) -> Tuple[str, Optional[float]]:
        df = _ecb_csv(series, n_obs=3)
        if df is not None and not df.empty:
            return country, float(df["OBS_VALUE"].iloc[-1])
        return country, None

    spreads: Dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_fetch_country, c, s): c
            for c, s in _ECB_YIELD_SERIES.items()
        }
        for fut in as_completed(futures):
            country, val = fut.result()
            if val is not None:
                spreads[country] = val

    if "IT" in spreads and "DE" in spreads:
        spreads["IT_DE_spread"] = spreads["IT"] - spreads["DE"]
    if "ES" in spreads and "DE" in spreads:
        spreads["ES_DE_spread"] = spreads["ES"] - spreads["DE"]

    logger.info("Sovereign spreads: %s", {k: f"{v:.3f}" for k, v in spreads.items()})
    return spreads


def fetch_system_indicators() -> Dict[str, float]:
    """
    Fetch system-level stress indicators from free public APIs in parallel.

    Returns a dict suitable for BankingSystemSimulation.system_indicators:
      CISS          — derived from IT-DE spread and bank volatility
      funding_stress — from bank stock volatility index
      sovereign_stress — IT-DE 10Y spread
      eurusd        — EUR/USD rate
    """
    indicators: Dict[str, float] = {}

    def _fetch_eurusd() -> Tuple[str, Optional[float]]:
        df = _ecb_csv("EXR/D.USD.EUR.SP00.A", n_obs=3)
        if df is not None and not df.empty:
            return "eurusd", float(df["OBS_VALUE"].iloc[-1])
        return "eurusd", None

    def _fetch_spreads() -> Dict[str, float]:
        return fetch_sovereign_spreads()

    def _fetch_bank_vol() -> Tuple[str, Optional[float]]:
        """Aggregate bank stock volatility as funding stress proxy."""
        try:
            import yfinance as yf
            # ALL_BANKS available at module level
            prices = _fetch_prices(ALL_BANKS, period="90d")
            if prices.empty:
                return "bank_vol", None
            rets = prices.pct_change().dropna()
            vol = float(rets.tail(30).std().mean() * np.sqrt(252))
            return "bank_vol", vol
        except Exception:
            return "bank_vol", None

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_fx    = pool.submit(_fetch_eurusd)
        f_sp    = pool.submit(_fetch_spreads)
        f_vol   = pool.submit(_fetch_bank_vol)

        _, eurusd = f_fx.result()
        spreads   = f_sp.result()
        _, vol    = f_vol.result()

    if eurusd is not None:
        indicators["eurusd"] = eurusd

    it_de = spreads.get("IT_DE_spread")
    es_de = spreads.get("ES_DE_spread")
    if it_de is not None:
        indicators["sovereign_stress"] = it_de
        # CISS proxy: normalise IT-DE spread to 0-1 range (0 = 0bps, 1 = 500bps+)
        indicators["CISS"] = min(1.0, max(0.0, it_de / 5.0))

    if vol is not None:
        indicators["bank_vol_annualised"] = vol
        # Funding stress: normalise bank vol (0 = 0%, 1 = 50%+ annualised)
        indicators["funding_stress"] = min(1.0, max(0.0, vol / 0.50))

    # Fallback if sovereign data unavailable
    indicators.setdefault("CISS", 0.2)
    indicators.setdefault("funding_stress", 0.1)

    logger.info("System indicators: %s", {k: f"{v:.4f}" for k, v in indicators.items()})
    return indicators


def fetch_all(
    bank_ids: Optional[List[str]] = None,
    correlation_window: int = 252,
    min_corr: float = 0.3,
) -> Dict[str, Any]:
    """
    Full parallel fetch: adjacency matrix + node features + system indicators.

    Returns
    -------
    dict with keys:
      adjacency      : {src: {tgt: weight}}
      bank_features  : {bank_id: {feature: value}}
      system         : {indicator: value}
      prices         : pd.DataFrame  (daily close prices)
      timestamp      : str  (UTC ISO)
    """
    ids = bank_ids or ALL_BANKS

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_adj  = pool.submit(fetch_correlation_adjacency, ids, correlation_window, min_corr)
        f_feat = pool.submit(fetch_bank_market_features, ids)
        f_sys  = pool.submit(fetch_system_indicators)

        adj   = f_adj.result()
        feats = f_feat.result()
        sys   = f_sys.result()

    # Retrieve raw prices for GNN time-series export
    prices = _fetch_prices(ids, period="2y")

    elapsed = time.time() - t0
    logger.info("fetch_all completed in %.1fs", elapsed)

    return {
        "adjacency":     adj,
        "bank_features": feats,
        "system":        sys,
        "prices":        prices,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }


def build_simulation_inputs_from_api(
    bank_ids: Optional[List[str]] = None,
    correlation_window: int = 252,
    min_corr: float = 0.3,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Build (bank_data, network_data, system_indicators) directly from market APIs.

    Same output format as ``data_loader.load_simulation_inputs`` — can be used
    as a drop-in replacement when live data is preferred over the EBA pipeline.
    """
    data = fetch_all(bank_ids, correlation_window, min_corr)
    feats = data["bank_features"]
    adj   = data["adjacency"]
    sys   = data["system"]

    bank_data: Dict[str, Any] = {}
    # ALL_BANKS available at module level
    for bid in (bank_ids or ALL_BANKS):
        f = feats.get(bid, {})
        total_assets = f.get("total_assets") or 1e12
        equity       = f.get("common_equity") or total_assets * 0.06

        # Derive regulatory-style ratios from market data where available
        cet1 = None
        if equity and total_assets:
            # CET1 ≈ equity / risk-weighted assets; RWA ≈ 35% of total assets for large banks
            rwa_est = total_assets * 0.35
            cet1 = float(equity / rwa_est * 100) if rwa_est > 0 else 13.0
        cet1 = max(6.0, min(25.0, cet1 or 13.0))  # clip to plausible range

        bank_data[bid] = {
            "CET1_ratio":              cet1,
            "LCR":                     130.0,    # not in yfinance; use default
            "NSFR":                    115.0,    # not in yfinance; use default
            "total_assets":            total_assets,
            "cash":                    total_assets * 0.10,
            "interbank_assets":        total_assets * 0.15,
            "interbank_liabilities":   total_assets * 0.14,
            "market_cap":              f.get("market_cap") or 0,
            "roe":                     (f.get("roe") or 0) * 100,
            "price_to_book":           f.get("price_to_book") or 1.0,
            "beta":                    f.get("beta") or 1.0,
            "volatility_30d":          f.get("volatility_30d") or 0.25,
        }

    system_indicators: Dict[str, Any] = {
        "CISS":           sys.get("CISS", 0.2),
        "funding_stress": sys.get("funding_stress", 0.1),
        "sovereign_stress": sys.get("sovereign_stress", 1.0),
        "bank_vol":       sys.get("bank_vol_annualised", 0.25),
    }

    logger.info(
        "API inputs ready: %d banks, %d edges, CISS=%.3f",
        len(bank_data),
        sum(len(v) for v in adj.values()) // 2,
        system_indicators["CISS"],
    )
    return bank_data, adj, system_indicators


# ── Daily graph snapshots for GNN training ──────────────────────────────────

def build_daily_graph_snapshots(
    bank_ids: Optional[List[str]] = None,
    lookback_years: int = 3,
    corr_window: int = 60,
    min_corr: float = 0.3,
    stride: int = 1,
    progress_callback: Optional[Any] = None,
    universe: Optional[str] = None,
    rmt_denoise: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build daily graph snapshots from historical market data for GNN training.

    Fetches multi-year daily prices once, then rolls through each trading day
    constructing:
      - Node features: [N, 5] per bank (volatility, return, log-price, beta_proxy, momentum)
      - Edge index + weight: from rolling correlation of returns
      - Spectral targets: lambda_2, spectral_gap, spectral_radius from the day's graph

    Parameters
    ----------
    lookback_years : int
        How many years of history to fetch (default 3 → ~750 trading days).
    corr_window : int
        Rolling window for correlation-based adjacency (trading days).
    min_corr : float
        Threshold for edge inclusion.
    stride : int
        Step between consecutive snapshots (1 = every day, 5 = weekly).
    progress_callback : callable(current, total)
        For UI progress updates.
    universe : str, optional
        Name of a bank universe from config (e.g. 'eu_10', 'eu_50').
        Overrides bank_ids when provided.
    rmt_denoise : bool
        If True, apply Marchenko-Pastur denoising to the correlation matrix
        before thresholding (uses constant method from scr_financial.network.rmt).

    Returns
    -------
    list of snapshot dicts compatible with GNNPredictor.
    """
    from scr_financial.network.spectral import (
        compute_laplacian, eigendecomposition, find_spectral_gap,
        analyze_spectral_properties,
    )

    # Resolve bank universe ------------------------------------------------
    if universe is not None:
        univ = load_universe(universe)
        ids = univ.ids
        # Temporarily patch BANK_TICKERS so _fetch_prices can resolve them
        _extra_tickers = {b.id: b.ticker for b in univ.banks}
        for k, v in _extra_tickers.items():
            if k not in BANK_TICKERS:
                BANK_TICKERS[k] = v
    else:
        ids = bank_ids or ALL_BANKS
    n_banks = len(ids)

    # Fetch full price history in one call
    prices = _fetch_prices(ids, period=f"{lookback_years * 365 + 60}d")
    if prices.empty or len(prices) < corr_window + 30:
        logger.warning("Insufficient price history for daily snapshots: %d rows", len(prices))
        return []

    returns = prices.pct_change().dropna()
    logger.info("Building daily snapshots: %d trading days, %d banks, corr_window=%d",
                len(returns), n_banks, corr_window)

    # Precompute rolling stats
    snapshots: List[Dict[str, Any]] = []
    valid_dates = returns.index[corr_window:]
    total = len(range(0, len(valid_dates), stride))

    for count, date_idx in enumerate(range(0, len(valid_dates), stride)):
        date = valid_dates[date_idx]
        window_end = corr_window + date_idx
        ret_window = returns.iloc[window_end - corr_window: window_end]

        # Correlation adjacency
        corr = ret_window.corr()

        # Optional RMT denoising (Marchenko-Pastur)
        if rmt_denoise:
            from scr_financial.network.rmt import denoise_correlation
            corr_vals = corr.reindex(index=ids, columns=ids).values
            corr_vals = np.nan_to_num(corr_vals, nan=0.0)
            np.fill_diagonal(corr_vals, 1.0)
            corr_vals = denoise_correlation(corr_vals, T=len(ret_window))
            corr = pd.DataFrame(corr_vals, index=ids, columns=ids)

        n = n_banks
        adj = np.zeros((n, n), dtype=np.float32)
        for i, src in enumerate(ids):
            for j, tgt in enumerate(ids):
                if i >= j or src not in corr.index or tgt not in corr.index:
                    continue
                w = float(corr.loc[src, tgt])
                if w >= min_corr:
                    adj[i, j] = w
                    adj[j, i] = w

        # Node features: [N, 5]
        # [vol_30d, mean_return_30d, log_price, beta_proxy, momentum_20d]
        node_feats = np.zeros((n, 5), dtype=np.float32)
        for i, bid in enumerate(ids):
            if bid in ret_window.columns:
                rets_i = ret_window[bid].values
                node_feats[i, 0] = float(np.std(rets_i) * np.sqrt(252))  # annualised vol
                node_feats[i, 1] = float(np.mean(rets_i) * 252)  # annualised return
                # Log price (from cumulative return)
                cum_ret = (1 + ret_window[bid]).prod()
                node_feats[i, 2] = float(np.log(max(cum_ret, 0.01)))
                # Beta proxy: covariance with market / var(market)
                mkt = ret_window.mean(axis=1).values
                cov = np.cov(rets_i, mkt)[0, 1] if len(rets_i) > 2 else 0
                var_mkt = np.var(mkt) if np.var(mkt) > 1e-10 else 1.0
                node_feats[i, 3] = float(cov / var_mkt)
                # 20-day momentum
                if len(rets_i) >= 20:
                    node_feats[i, 4] = float(np.sum(rets_i[-20:]))

        # Edge index / weight
        rows, cols = np.nonzero(adj)
        edge_index = np.stack([rows, cols], axis=0).astype(np.int64) if len(rows) > 0 \
            else np.zeros((2, 0), dtype=np.int64)
        edge_weight = adj[rows, cols] if len(rows) > 0 else np.zeros(0, dtype=np.float32)

        # Spectral targets
        adj_sym = (adj + adj.T) / 2.0
        L = compute_laplacian(adj_sym, normalized=True)
        eigenvalues, eigenvectors = eigendecomposition(L)
        gap_idx, gap_size = find_spectral_gap(eigenvalues)
        props = analyze_spectral_properties(eigenvalues, eigenvectors)

        lam2 = float(props["algebraic_connectivity"])
        gap = float(gap_size)
        radius = float(props["spectral_radius"])

        snapshots.append({
            "node_features": node_feats,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "targets": {"lambda_2": lam2, "spectral_gap": gap, "spectral_radius": radius},
            "lambda_2": lam2,
            "spectral_gap": gap,
            "spectral_radius": radius,
            "time": count,
            "date": str(date.date()) if hasattr(date, 'date') else str(date),
        })

        if progress_callback and (count + 1) % 50 == 0:
            progress_callback(count + 1, total)

    logger.info("Built %d daily graph snapshots (stride=%d)", len(snapshots), stride)
    return snapshots
