"""FX rate converter using ECB Statistical Data Warehouse (SDW).

Fetches daily exchange rates from the ECB SDW REST API and caches
results locally as JSON to avoid repeated network calls.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "scr_financial" / "fx"
_ECB_SDW_BASE = "https://sdw-wsrest.ecb.europa.eu/service/data"

# ECB dataset key for daily exchange rates (EXR)
# Dimensions: frequency.currency.currency_denom.exr_type.exr_suffix
# D = daily, . = all currencies, EUR = denominator, SP00 = spot, A = average
_EXR_FLOW = "EXR"


class FXConverter:
    """Fetch and cache ECB daily FX rates, convert arbitrary currency pairs via EUR cross.

    Usage
    -----
    >>> fx = FXConverter()
    >>> fx.fetch("USD", start="2020-01-01", end="2020-12-31")
    >>> rate = fx.get_rate("USD", "2020-06-15")
    >>> eur_amount = fx.convert(100.0, "USD", "EUR", "2020-06-15")
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # In-memory store: {currency: pd.Series} with DatetimeIndex -> float (units of currency per EUR)
        self._rates: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        currency: str,
        start: str = "2005-01-01",
        end: Optional[str] = None,
        force: bool = False,
    ) -> pd.Series:
        """Fetch daily EUR/<currency> rate from ECB SDW (or cache).

        Parameters
        ----------
        currency : str
            ISO 4217 code (e.g. 'USD', 'GBP', 'JPY').
        start, end : str
            ISO date strings bounding the period.
        force : bool
            If True, bypass cache and re-download.

        Returns
        -------
        pd.Series with DatetimeIndex and float values (units of *currency* per 1 EUR).
        """
        currency = currency.upper()
        if currency == "EUR":
            # trivial case
            idx = pd.bdate_range(start, end or datetime.today().strftime("%Y-%m-%d"))
            s = pd.Series(1.0, index=idx, name="EUR")
            self._rates["EUR"] = s
            return s

        end = end or datetime.today().strftime("%Y-%m-%d")
        cache_file = self._cache_path(currency, start, end)

        if not force and cache_file.exists():
            s = self._load_cache(cache_file)
            if s is not None:
                self._rates[currency] = s
                logger.info("Loaded %s rates from cache (%d obs)", currency, len(s))
                return s

        # Fetch from ECB SDW
        s = self._fetch_ecb(currency, start, end)
        self._rates[currency] = s
        self._save_cache(cache_file, s)
        logger.info("Fetched %s rates from ECB SDW (%d obs)", currency, len(s))
        return s

    def get_rate(self, currency: str, date: str) -> float:
        """Return EUR/<currency> rate for a given date (forward-filled for holidays)."""
        currency = currency.upper()
        if currency == "EUR":
            return 1.0
        if currency not in self._rates:
            raise KeyError(
                f"Currency '{currency}' not loaded. Call fetch() first."
            )
        s = self._rates[currency]
        ts = pd.Timestamp(date)
        # forward-fill to handle weekends / holidays
        s_ff = s.reindex(pd.date_range(s.index.min(), s.index.max())).ffill()
        if ts < s_ff.index.min() or ts > s_ff.index.max():
            raise ValueError(f"Date {date} outside loaded range for {currency}")
        return float(s_ff.loc[ts])

    def convert(
        self,
        amount: float,
        from_ccy: str,
        to_ccy: str,
        date: str,
    ) -> float:
        """Convert *amount* from one currency to another via EUR cross.

        Both currencies must have been previously fetched.
        """
        from_ccy, to_ccy = from_ccy.upper(), to_ccy.upper()
        if from_ccy == to_ccy:
            return amount
        # ECB quotes are X per 1 EUR
        rate_from = self.get_rate(from_ccy, date)  # from_ccy / EUR
        rate_to = self.get_rate(to_ccy, date)      # to_ccy / EUR
        # amount in from_ccy -> EUR -> to_ccy
        eur_amount = amount / rate_from
        return eur_amount * rate_to

    def ensure_currencies(
        self,
        currencies: set[str],
        start: str = "2005-01-01",
        end: Optional[str] = None,
    ) -> None:
        """Convenience: fetch all required currencies in one call."""
        for ccy in currencies:
            if ccy.upper() not in self._rates:
                self.fetch(ccy, start=start, end=end)

    # ------------------------------------------------------------------
    # ECB SDW fetching
    # ------------------------------------------------------------------

    def _fetch_ecb(self, currency: str, start: str, end: str) -> pd.Series:
        """Download daily EXR series from ECB SDW REST API."""
        key = f"D.{currency}.EUR.SP00.A"
        url = f"{_ECB_SDW_BASE}/{_EXR_FLOW}/{key}"
        params = {
            "startPeriod": start,
            "endPeriod": end,
            "format": "csvdata",
        }
        headers = {"Accept": "text/csv"}

        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()

        df = pd.read_csv(
            pd.io.common.StringIO(resp.text),
            usecols=["TIME_PERIOD", "OBS_VALUE"],
        )
        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
        df = df.set_index("TIME_PERIOD").sort_index()
        s = df["OBS_VALUE"].astype(float)
        s.name = currency
        return s

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _cache_path(self, currency: str, start: str, end: str) -> Path:
        return self.cache_dir / f"ecb_exr_{currency}_{start}_{end}.json"

    def _save_cache(self, path: Path, series: pd.Series) -> None:
        data = {
            "currency": series.name,
            "dates": series.index.strftime("%Y-%m-%d").tolist(),
            "values": series.tolist(),
            "cached_at": datetime.utcnow().isoformat(),
        }
        path.write_text(json.dumps(data))

    def _load_cache(self, path: Path) -> Optional[pd.Series]:
        try:
            data = json.loads(path.read_text())
            # expire cache after 24 h
            cached_at = datetime.fromisoformat(data["cached_at"])
            if datetime.utcnow() - cached_at > timedelta(hours=24):
                return None
            idx = pd.to_datetime(data["dates"])
            s = pd.Series(data["values"], index=idx, name=data["currency"])
            return s
        except Exception:
            logger.warning("Corrupt cache file %s; will re-fetch.", path)
            return None
