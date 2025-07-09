"""
Data preprocessor for financial network analysis.

This module provides the DataPreprocessor class for loading, cleaning,
and normalizing financial data from various sources.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Union

from .collectors.eba_collector import EBACollector
from .collectors.ecb_collector import ECBCollector
from .collectors.market_collector import MarketDataCollector

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles loading, cleaning, and normalizing financial data from various sources.

    Args:
        start_date: Start date for data collection in 'YYYY-MM-DD' format.
        end_date: End date for data collection in 'YYYY-MM-DD' format.
        bank_list: List of bank identifiers to include in the analysis.

    Attributes:
        node_data: Dictionary containing bank-specific attributes.
        edge_data: Dictionary containing interbank network data.
        system_data: Dictionary containing system-wide indicators.

    Raises:
        ValueError: If start_date or end_date cannot be parsed as dates, or if
            end_date is earlier than start_date.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        bank_list: Optional[List[str]] = None,
    ) -> None:
        """Initialize the data preprocessor with date range and optional bank list."""
        # Validate and parse dates
        try:
            parsed_start = pd.to_datetime(start_date)
        except Exception as exc:
            raise ValueError(
                f"start_date '{start_date}' could not be parsed as a date: {exc}"
            ) from exc

        try:
            parsed_end = pd.to_datetime(end_date)
        except Exception as exc:
            raise ValueError(
                f"end_date '{end_date}' could not be parsed as a date: {exc}"
            ) from exc

        if parsed_end < parsed_start:
            raise ValueError(
                f"end_date '{end_date}' must be >= start_date '{start_date}'."
            )

        self.start_date = start_date
        self.end_date = end_date
        self.bank_list = bank_list

        # Initialize data collectors
        self.eba_collector = EBACollector()
        self.ecb_collector = ECBCollector()
        self.market_collector = MarketDataCollector()

        # Initialize data containers
        self.node_data: Dict[str, pd.DataFrame] = {}
        self.edge_data: Dict[str, pd.DataFrame] = {}
        self.system_data: Dict[str, pd.DataFrame] = {}

        # Initialize scalers
        self.node_scalers: Dict[str, StandardScaler] = {}
        self.edge_scalers: Dict[str, StandardScaler] = {}

    def load_bank_node_data(
        self, data_sources: Dict[str, str]
    ) -> Dict[str, pd.DataFrame]:
        """Load bank-specific attributes from specified sources.

        Args:
            data_sources: Dictionary mapping attribute categories to data sources.

        Returns:
            Dictionary of loaded bank attributes.
        """
        logger.info(
            "Loading bank node data for sources: %s", list(data_sources.keys())
        )
        for category, source in data_sources.items():
            if source == "EBA_transparency":
                data = self.eba_collector.collect_transparency_data(
                    self.start_date, self.end_date, self.bank_list
                )
            elif source == "EBA_aggregated":
                data = self.eba_collector.collect_aggregated_data(
                    self.start_date, self.end_date, self.bank_list
                )
            elif source == "NYU_VLAB":
                data = self.market_collector.collect_srisk_data(
                    self.start_date, self.end_date, self.bank_list
                )
            else:
                raise ValueError(f"Unknown data source: {source}")

            self.node_data[category] = data

        return self.node_data

    def load_interbank_exposures(
        self, source: str = "ECB_TARGET2"
    ) -> Dict[str, pd.DataFrame]:
        """Load interbank network data from specified source.

        Args:
            source: Source for interbank exposure data, by default 'ECB_TARGET2'.

        Returns:
            Dictionary of interbank exposure data.
        """
        logger.info("Loading interbank exposures from source: %s", source)
        if source == "ECB_TARGET2":
            data = self.ecb_collector.collect_target2_data(
                self.start_date, self.end_date, self.bank_list
            )
        elif source == "BCBS_GSIB":
            data = self.ecb_collector.collect_gsib_data(
                self.start_date, self.end_date, self.bank_list
            )
        else:
            raise ValueError(f"Unknown data source: {source}")

        self.edge_data["interbank_exposures"] = data

        return self.edge_data

    def load_system_indicators(self) -> Dict[str, pd.DataFrame]:
        """Load system-wide financial indicators.

        Returns:
            Dictionary of system-wide indicators.
        """
        logger.info("Loading system-wide indicators")

        # Load ECB CISS data
        ciss_data = self.ecb_collector.collect_ciss_data(
            self.start_date, self.end_date
        )
        self.system_data["CISS"] = ciss_data

        # Load BIS credit-to-GDP gap
        credit_gap_data = self.market_collector.collect_bis_data(
            self.start_date, self.end_date, "credit_to_GDP_gap"
        )
        self.system_data["credit_to_GDP_gap"] = credit_gap_data

        # Load funding stress indicators
        funding_stress = self.market_collector.collect_funding_stress_data(
            self.start_date, self.end_date
        )
        self.system_data["funding_stress"] = funding_stress

        return self.system_data

    def normalize_edge_weights(
        self, method: str = "degree"
    ) -> Dict[str, pd.DataFrame]:
        """Apply normalization to edge weights.

        Args:
            method: Normalization method, by default 'degree'.

        Returns:
            Dictionary of normalized edge weights.
        """
        normalized_edge_data: Dict[str, pd.DataFrame] = {}

        for key, data in self.edge_data.items():
            if method == "degree":
                # Normalize by node degrees
                # TODO: vectorise by groupby(date) for large datasets
                for date in data.index.unique():
                    adj_matrix = data.loc[date].pivot(
                        index="source", columns="target", values="weight"
                    ).fillna(0)

                    # Calculate out-degrees
                    out_degrees = adj_matrix.sum(axis=1)
                    in_degrees = adj_matrix.sum(axis=0)

                    # Avoid division by zero
                    out_degrees = out_degrees.replace(0, 1)
                    in_degrees = in_degrees.replace(0, 1)

                    # Normalize
                    norm_matrix = adj_matrix.div(np.sqrt(out_degrees), axis=0)
                    norm_matrix = norm_matrix.div(np.sqrt(in_degrees), axis=1)

                    # Convert back to long format
                    norm_df = norm_matrix.stack().reset_index()
                    norm_df.columns = pd.Index(["source", "target", "weight"])
                    norm_df["date"] = date

                    if key not in normalized_edge_data:
                        normalized_edge_data[key] = norm_df
                    else:
                        normalized_edge_data[key] = pd.concat(
                            [normalized_edge_data[key], norm_df]
                        )

            elif method == "standardize":
                # Apply standardization
                scaler = StandardScaler()
                for date in data.index.unique():
                    date_data = data.loc[date].copy()
                    date_data["weight"] = scaler.fit_transform(date_data[["weight"]])

                    if key not in normalized_edge_data:
                        normalized_edge_data[key] = date_data
                    else:
                        normalized_edge_data[key] = pd.concat(
                            [normalized_edge_data[key], date_data]
                        )

                self.edge_scalers[key] = scaler

            else:
                raise ValueError(f"Unknown normalization method: {method}")

        self.edge_data = normalized_edge_data
        return self.edge_data

    def filter_network(
        self, method: str = "threshold", threshold: float = 0.05
    ) -> Dict[str, pd.DataFrame]:
        """Apply filtering techniques to reduce noise in the network.

        Args:
            method: Filtering method, by default 'threshold'.
            threshold: Threshold value for filtering, by default 0.05.

        Returns:
            Dictionary of filtered edge data.
        """
        filtered_edge_data: Dict[str, pd.DataFrame] = {}

        for key, data in self.edge_data.items():
            if method == "threshold":
                # Remove edges below threshold
                filtered_data = data[data["weight"] > threshold].copy()
                filtered_edge_data[key] = filtered_data
            else:
                raise ValueError(f"Unknown filtering method: {method}")

        self.edge_data = filtered_edge_data
        return self.edge_data

    def align_timescales(self) -> None:
        """Align data with different frequencies to a common timeline."""
        # Determine common dates across all datasets
        all_dates: set = set()

        # Collect all unique dates from node data
        for category, data in self.node_data.items():
            if isinstance(data.index, pd.DatetimeIndex):
                all_dates.update(data.index)
            elif "date" in data.columns:
                all_dates.update(pd.to_datetime(data["date"]))

        # Collect all unique dates from edge data
        for key, data in self.edge_data.items():
            if "date" in data.columns:
                all_dates.update(pd.to_datetime(data["date"]))

        # Collect all unique dates from system data
        for key, data in self.system_data.items():
            if isinstance(data.index, pd.DatetimeIndex):
                all_dates.update(data.index)
            elif "date" in data.columns:
                all_dates.update(pd.to_datetime(data["date"]))

        # Convert to sorted list
        common_dates = sorted(list(all_dates))

        # Align node data
        for category, data in self.node_data.items():
            if isinstance(data.index, pd.DatetimeIndex):
                self.node_data[category] = data.reindex(common_dates, method="ffill")
            elif "date" in data.columns:
                data["date"] = pd.to_datetime(data["date"])
                data = data.set_index("date")
                self.node_data[category] = (
                    data.reindex(common_dates, method="ffill").reset_index()
                )

        # Align system data
        for key, data in self.system_data.items():
            if isinstance(data.index, pd.DatetimeIndex):
                self.system_data[key] = data.reindex(common_dates, method="ffill")
            elif "date" in data.columns:
                data["date"] = pd.to_datetime(data["date"])
                data = data.set_index("date")
                self.system_data[key] = (
                    data.reindex(common_dates, method="ffill").reset_index()
                )

    def _find_closest_date(
        self,
        dates: pd.DatetimeIndex,
        target: pd.Timestamp,
    ) -> Optional[pd.Timestamp]:
        """Return the most recent date in *dates* that is <= *target*.

        Args:
            dates: Sequence of timestamps to search.
            target: Upper bound timestamp.

        Returns:
            The closest previous timestamp, or None if no date is <= target.
        """
        prev_dates = [d for d in dates if d <= target]
        return max(prev_dates) if prev_dates else None

    def get_data_for_timepoint(
        self, time_point: Union[str, datetime]
    ) -> Dict[str, Dict]:
        """Get all data for a specific time point.

        Args:
            time_point: Time point for which to retrieve data.

        Returns:
            Dictionary containing node_data, edge_data, and system_data for the
            specified time point.
        """
        if isinstance(time_point, str):
            time_point = pd.to_datetime(time_point)

        result: Dict[str, Dict] = {
            "node_data": {},
            "edge_data": {},
            "system_data": {},
        }

        # Extract node data
        for category, data in self.node_data.items():
            if isinstance(data.index, pd.DatetimeIndex):
                if time_point in data.index:
                    result["node_data"][category] = data.loc[time_point]
                else:
                    closest = self._find_closest_date(data.index, time_point)
                    if closest is not None:
                        result["node_data"][category] = data.loc[closest]
            elif "date" in data.columns:
                date_data = data[data["date"] == time_point]
                if not date_data.empty:
                    result["node_data"][category] = date_data
                else:
                    closest = self._find_closest_date(
                        pd.DatetimeIndex(pd.to_datetime(data["date"])), time_point
                    )
                    if closest is not None:
                        result["node_data"][category] = data[
                            data["date"] == closest
                        ]

        # Extract edge data
        for key, data in self.edge_data.items():
            if "date" in data.columns:
                date_data = data[data["date"] == time_point]
                if not date_data.empty:
                    result["edge_data"][key] = date_data
                else:
                    closest = self._find_closest_date(
                        pd.DatetimeIndex(pd.to_datetime(data["date"])), time_point
                    )
                    if closest is not None:
                        result["edge_data"][key] = data[data["date"] == closest]

        # Extract system data
        for key, data in self.system_data.items():
            if isinstance(data.index, pd.DatetimeIndex):
                if time_point in data.index:
                    result["system_data"][key] = data.loc[time_point]
                else:
                    closest = self._find_closest_date(data.index, time_point)
                    if closest is not None:
                        result["system_data"][key] = data.loc[closest]
            elif "date" in data.columns:
                date_data = data[data["date"] == time_point]
                if not date_data.empty:
                    result["system_data"][key] = date_data
                else:
                    closest = self._find_closest_date(
                        pd.DatetimeIndex(pd.to_datetime(data["date"])), time_point
                    )
                    if closest is not None:
                        result["system_data"][key] = data[
                            data["date"] == closest
                        ]

        return result
