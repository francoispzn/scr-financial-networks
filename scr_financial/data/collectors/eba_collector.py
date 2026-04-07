"""
EBA data collector for financial network analysis.

This module provides the EBACollector class for retrieving data from the
European Banking Authority (EBA) transparency exercises and stress tests.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Solvency parameters per bank used by sample data generators.
# Keys are bank identifiers; values hold base CET1 ratio and T1 leverage ratio.
_BANK_SOLVENCY_PARAMS: dict[str, dict[str, float]] = {
    "DE_DBK": {"base_cet1": 13.2, "base_t1_leverage": 4.5},   # Deutsche Bank
    "FR_BNP": {"base_cet1": 12.8, "base_t1_leverage": 4.7},   # BNP Paribas
    "ES_SAN": {"base_cet1": 11.9, "base_t1_leverage": 5.1},   # Santander
    "IT_UCG": {"base_cet1": 13.5, "base_t1_leverage": 5.3},   # UniCredit
    "NL_ING": {"base_cet1": 14.2, "base_t1_leverage": 4.9},   # ING
    "SE_NDA": {"base_cet1": 16.1, "base_t1_leverage": 5.5},   # Nordea
    "CH_UBS": {"base_cet1": 14.5, "base_t1_leverage": 5.2},   # UBS
    "UK_BARC": {"base_cet1": 13.8, "base_t1_leverage": 4.8},  # Barclays
    "UK_HSBC": {"base_cet1": 15.2, "base_t1_leverage": 5.5},  # HSBC
    "FR_ACA": {"base_cet1": 13.1, "base_t1_leverage": 4.6},   # Credit Agricole
}

# Liquidity parameters per bank used by sample data generators.
# Keys are bank identifiers; values hold base LCR and NSFR ratios.
_BANK_LIQUIDITY_PARAMS: dict[str, dict[str, float]] = {
    "DE_DBK": {"base_lcr": 145.0, "base_nsfr": 115.0},   # Deutsche Bank
    "FR_BNP": {"base_lcr": 152.0, "base_nsfr": 118.0},   # BNP Paribas
    "ES_SAN": {"base_lcr": 158.0, "base_nsfr": 112.0},   # Santander
    "IT_UCG": {"base_lcr": 165.0, "base_nsfr": 110.0},   # UniCredit
    "NL_ING": {"base_lcr": 138.0, "base_nsfr": 115.0},   # ING
    "SE_NDA": {"base_lcr": 172.0, "base_nsfr": 120.0},   # Nordea
    "CH_UBS": {"base_lcr": 168.0, "base_nsfr": 117.0},   # UBS
    "UK_BARC": {"base_lcr": 156.0, "base_nsfr": 113.0},  # Barclays
    "UK_HSBC": {"base_lcr": 160.0, "base_nsfr": 116.0},  # HSBC
    "FR_ACA": {"base_lcr": 149.0, "base_nsfr": 114.0},   # Credit Agricole
}

_DEFAULT_BASE_SOLVENCY: dict[str, float] = {"base_cet1": 12.0, "base_t1_leverage": 4.5}
_DEFAULT_BASE_LIQUIDITY: dict[str, float] = {"base_lcr": 150.0, "base_nsfr": 110.0}


class EBACollector:
    """
    Collector for European Banking Authority (EBA) data.

    This class provides methods for retrieving data from EBA transparency
    exercises and stress tests, which include key bank metrics like CET1 ratio,
    leverage ratio, and liquidity coverage ratio.
    """

    def __init__(self) -> None:
        """Initialize the EBA data collector."""
        self.base_url = "https://www.eba.europa.eu/sites/default/documents/files/"
        self.transparency_url = self.base_url + "transparency-exercise/"
        self.stress_test_url = self.base_url + "eu-wide-stress-testing/"
        self._rng = np.random.default_rng(seed=42)

    def __repr__(self) -> str:
        """Return a string representation of this collector.

        Returns:
            str: Class name and base URL.
        """
        return f"{self.__class__.__name__}(base_url={self.base_url!r})"

    def collect_transparency_data(
        self,
        start_date: str,
        end_date: str,
        bank_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Collect data from EBA transparency exercises.

        Args:
            start_date: Start date for data collection in 'YYYY-MM-DD' format.
            end_date: End date for data collection in 'YYYY-MM-DD' format.
            bank_list: List of bank identifiers to include.

        Returns:
            DataFrame containing bank solvency metrics.
        """
        # TODO: replace with real EBA API call when credentials are available

        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Determine which transparency exercises to include based on dates
        exercises = []

        if start <= pd.to_datetime("2020-12-31") <= end:
            exercises.append("2020")
        if start <= pd.to_datetime("2021-12-31") <= end:
            exercises.append("2021")
        if start <= pd.to_datetime("2022-12-31") <= end:
            exercises.append("2022")
        if start <= pd.to_datetime("2023-12-31") <= end:
            exercises.append("2023")
        if start <= pd.to_datetime("2024-12-31") <= end:
            exercises.append("2024")

        # If no exercises match the date range, use sample data
        if not exercises:
            return self._get_sample_transparency_data(start_date, end_date, bank_list)

        # Initialize empty DataFrame to store results
        all_data = pd.DataFrame()

        # Collect data from each exercise
        for exercise in exercises:
            try:
                # In a real implementation, this would download from EBA website
                # For now, we'll use sample data
                exercise_data = self._get_sample_transparency_data(
                    start_date, end_date, bank_list
                )
                exercise_data["exercise"] = exercise
                all_data = pd.concat([all_data, exercise_data])
            except Exception as exc:
                logger.error(
                    "Error collecting %s transparency data: %s", exercise, exc
                )

        # Filter by bank list if provided
        if bank_list is not None and not all_data.empty:
            all_data = all_data[all_data["bank_id"].isin(bank_list)]

        return all_data

    def collect_aggregated_data(
        self,
        start_date: str,
        end_date: str,
        bank_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Collect aggregated data from EBA.

        Args:
            start_date: Start date for data collection in 'YYYY-MM-DD' format.
            end_date: End date for data collection in 'YYYY-MM-DD' format.
            bank_list: List of bank identifiers to include.

        Returns:
            DataFrame containing bank liquidity metrics.
        """
        # TODO: replace with real EBA API call when credentials are available

        # For now, use sample data
        return self._get_sample_aggregated_data(start_date, end_date, bank_list)

    def _get_sample_transparency_data(
        self,
        start_date: str,
        end_date: str,
        bank_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate sample transparency exercise data for testing.

        Args:
            start_date: Start date for data collection in 'YYYY-MM-DD' format.
            end_date: End date for data collection in 'YYYY-MM-DD' format.
            bank_list: List of bank identifiers to include.

        Returns:
            DataFrame containing sample bank solvency metrics.
        """
        # Default bank list if none provided
        if bank_list is None:
            bank_list = [
                "DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
                "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA",
            ]

        # Generate dates within the range
        date_range = pd.date_range(start=start_date, end=end_date, freq="Q")

        # Create sample data
        data = []

        for bank_id in bank_list:
            params = _BANK_SOLVENCY_PARAMS.get(bank_id, _DEFAULT_BASE_SOLVENCY)
            base_cet1 = params["base_cet1"]
            base_t1_leverage = params["base_t1_leverage"]

            # Add some random variation over time
            for date in date_range:
                # Add noise and slight trend
                time_factor = (date - pd.to_datetime(start_date)).days / 365
                cet1 = base_cet1 + 0.5 * time_factor + self._rng.normal(0, 0.3)
                t1_leverage = (
                    base_t1_leverage + 0.2 * time_factor + self._rng.normal(0, 0.15)
                )

                # Add row to data
                data.append(
                    {
                        "date": date,
                        "bank_id": bank_id,
                        "CET1_ratio": round(cet1, 2),
                        "Tier1_leverage_ratio": round(t1_leverage, 2),
                        "total_assets": round(
                            self._rng.uniform(500, 2000) * 1e9, 0
                        ),  # in EUR
                        "risk_weighted_assets": round(
                            self._rng.uniform(200, 800) * 1e9, 0
                        ),  # in EUR
                    }
                )

        return pd.DataFrame(data)

    def _get_sample_aggregated_data(
        self,
        start_date: str,
        end_date: str,
        bank_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate sample aggregated data for testing.

        Args:
            start_date: Start date for data collection in 'YYYY-MM-DD' format.
            end_date: End date for data collection in 'YYYY-MM-DD' format.
            bank_list: List of bank identifiers to include.

        Returns:
            DataFrame containing sample bank liquidity metrics.
        """
        # Default bank list if none provided
        if bank_list is None:
            bank_list = [
                "DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
                "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA",
            ]

        # Generate dates within the range (semi-annual)
        date_range = pd.date_range(start=start_date, end=end_date, freq="6M")

        # Create sample data
        data = []

        for bank_id in bank_list:
            params = _BANK_LIQUIDITY_PARAMS.get(bank_id, _DEFAULT_BASE_LIQUIDITY)
            base_lcr = params["base_lcr"]
            base_nsfr = params["base_nsfr"]

            # Add some random variation over time
            for date in date_range:
                # Add noise and slight trend
                time_factor = (date - pd.to_datetime(start_date)).days / 365
                lcr = base_lcr + 5 * time_factor + self._rng.normal(0, 5)
                nsfr = base_nsfr + 2 * time_factor + self._rng.normal(0, 3)

                # Add row to data
                data.append(
                    {
                        "date": date,
                        "bank_id": bank_id,
                        "LCR": round(lcr, 1),
                        "NSFR": round(nsfr, 1),
                        "liquid_assets": round(
                            self._rng.uniform(100, 500) * 1e9, 0
                        ),  # in EUR
                        "net_cash_outflows": round(
                            self._rng.uniform(50, 300) * 1e9, 0
                        ),  # in EUR
                    }
                )

        return pd.DataFrame(data)
