"""
Data preprocessor for financial network analysis.

This module provides the DataPreprocessor class for loading, cleaning,
and normalizing financial data from various sources.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Union

from .collectors.eba_collector import EBACollector
from .collectors.ecb_collector import ECBCollector
from .collectors.market_collector import MarketDataCollector


class DataPreprocessor:
    """
    Handles loading, cleaning, and normalizing financial data from various sources.
    
    Parameters
    ----------
    start_date : str
        Start date for data collection in 'YYYY-MM-DD' format
    end_date : str
        End date for data collection in 'YYYY-MM-DD' format
    bank_list : list of str, optional
        List of bank identifiers to include in the analysis
    
    Attributes
    ----------
    node_data : dict
        Dictionary containing bank-specific attributes
    edge_data : dict
        Dictionary containing interbank network data
    system_data : dict
        Dictionary containing system-wide indicators
    """
    
    def __init__(self, start_date: str, end_date: str, bank_list: Optional[List[str]] = None):
        """Initialize the data preprocessor with date range and optional bank list."""
        self.start_date = start_date
        self.end_date = end_date
        self.bank_list = bank_list
        
        # Initialize data collectors
        self.eba_collector = EBACollector()
        self.ecb_collector = ECBCollector()
        self.market_collector = MarketDataCollector()
        
        # Initialize data containers
        self.node_data = {}
        self.edge_data = {}
        self.system_data = {}
        
        # Initialize scalers
        self.node_scalers = {}
        self.edge_scalers = {}
    
    def load_bank_node_data(self, data_sources: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Load bank-specific attributes from specified sources.
        
        Parameters
        ----------
        data_sources : dict
            Dictionary mapping attribute categories to data sources
            
        Returns
        -------
        dict
            Dictionary of loaded bank attributes
        """
        for category, source in data_sources.items():
            if source == 'EBA_transparency':
                data = self.eba_collector.collect_transparency_data(
                    self.start_date, 
                    self.end_date, 
                    self.bank_list
                )
            elif source == 'EBA_aggregated':
                data = self.eba_collector.collect_aggregated_data(
                    self.start_date, 
                    self.end_date, 
                    self.bank_list
                )
            elif source == 'NYU_VLAB':
                data = self.market_collector.collect_srisk_data(
                    self.start_date, 
                    self.end_date, 
                    self.bank_list
                )
            else:
                raise ValueError(f"Unknown data source: {source}")
            
            self.node_data[category] = data
        
        return self.node_data
    
    def load_interbank_exposures(self, source: str = 'ECB_TARGET2') -> Dict[str, pd.DataFrame]:
        """
        Load interbank network data from specified source.
        
        Parameters
        ----------
        source : str, optional
            Source for interbank exposure data, by default 'ECB_TARGET2'
            
        Returns
        -------
        dict
            Dictionary of interbank exposure data
        """
        if source == 'ECB_TARGET2':
            data = self.ecb_collector.collect_target2_data(
                self.start_date, 
                self.end_date, 
                self.bank_list
            )
        elif source == 'BCBS_GSIB':
            data = self.ecb_collector.collect_gsib_data(
                self.start_date, 
                self.end_date, 
                self.bank_list
            )
        else:
            raise ValueError(f"Unknown data source: {source}")
        
        self.edge_data['interbank_exposures'] = data
        
        return self.edge_data
    
    def load_system_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Load system-wide financial indicators.
        
        Returns
        -------
        dict
            Dictionary of system-wide indicators
        """
        # Load ECB CISS data
        ciss_data = self.ecb_collector.collect_ciss_data(
            self.start_date, 
            self.end_date
        )
        self.system_data['CISS'] = ciss_data
        
        # Load BIS credit-to-GDP gap
        credit_gap_data = self.market_collector.collect_bis_data(
            self.start_date, 
            self.end_date, 
            'credit_to_GDP_gap'
        )
        self.system_data['credit_to_GDP_gap'] = credit_gap_data
        
        # Load funding stress indicators
        funding_stress = self.market_collector.collect_funding_stress_data(
            self.start_date, 
            self.end_date
        )
        self.system_data['funding_stress'] = funding_stress
        
        return self.system_data
    
    def normalize_edge_weights(self, method: str = 'degree') -> Dict[str, pd.DataFrame]:
        """
        Apply normalization to edge weights.
        
        Parameters
        ----------
        method : str, optional
            Normalization method, by default 'degree'
            
        Returns
        -------
        dict
            Dictionary of normalized edge weights
        """
        normalized_edge_data = {}
        
        for key, data in self.edge_data.items():
            if method == 'degree':
                # Normalize by node degrees
                for date in data.index.unique():
                    adj_matrix = data.loc[date].pivot(
                        index='source', 
                        columns='target', 
                        values='weight'
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
                    norm_df.columns = ['source', 'target', 'weight']
                    norm_df['date'] = date
                    
                    if key not in normalized_edge_data:
                        normalized_edge_data[key] = norm_df
                    else:
                        normalized_edge_data[key] = pd.concat([normalized_edge_data[key], norm_df])
                        
            elif method == 'standardize':
                # Apply standardization
                scaler = StandardScaler()
                for date in data.index.unique():
                    date_data = data.loc[date].copy()
                    date_data['weight'] = scaler.fit_transform(date_data[['weight']])
                    
                    if key not in normalized_edge_data:
                        normalized_edge_data[key] = date_data
                    else:
                        normalized_edge_data[key] = pd.concat([normalized_edge_data[key], date_data])
                
                self.edge_scalers[key] = scaler
            
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        
        self.edge_data = normalized_edge_data
        return self.edge_data
    
    def filter_network(self, method: str = 'threshold', threshold: float = 0.05) -> Dict[str, pd.DataFrame]:
        """
        Apply filtering techniques to reduce noise in the network.
        
        Parameters
        ----------
        method : str, optional
            Filtering method, by default 'threshold'
        threshold : float, optional
            Threshold value for filtering, by default 0.05
            
        Returns
        -------
        dict
            Dictionary of filtered edge data
        """
        filtered_edge_data = {}
        
        for key, data in self.edge_data.items():
            if method == 'threshold':
                # Remove edges below threshold
                filtered_data = data[data['weight'] > threshold].copy()
                filtered_edge_data[key] = filtered_data
            else:
                raise ValueError(f"Unknown filtering method: {method}")
        
        self.edge_data = filtered_edge_data
        return self.edge_data
    
    def align_timescales(self) -> None:
        """
        Align data with different frequencies to a common timeline.
        """
        # Determine common dates across all datasets
        all_dates = set()
        
        # Collect all unique dates from node data
        for category, data in self.node_data.items():
            if isinstance(data.index, pd.DatetimeIndex):
                all_dates.update(data.index)
            elif 'date' in data.columns:
                all_dates.update(pd.to_datetime(data['date']))
        
        # Collect all unique dates from edge data
        for key, data in self.edge_data.items():
            if 'date' in data.columns:
                all_dates.update(pd.to_datetime(data['date']))
        
        # Collect all unique dates from system data
        for key, data in self.system_data.items():
            if isinstance(data.index, pd.DatetimeIndex):
                all_dates.update(data.index)
            elif 'date' in data.columns:
                all_dates.update(pd.to_datetime(data['date']))
        
        # Convert to sorted list
        common_dates = sorted(list(all_dates))
        
        # Align node data
        for category, data in self.node_data.items():
            if isinstance(data.index, pd.DatetimeIndex):
                self.node_data[category] = data.reindex(common_dates, method='ffill')
            elif 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date')
                self.node_data[category] = data.reindex(common_dates, method='ffill').reset_index()
        
        # Align system data
        for key, data in self.system_data.items():
            if isinstance(data.index, pd.DatetimeIndex):
                self.system_data[key] = data.reindex(common_dates, method='ffill')
            elif 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date')
                self.system_data[key] = data.reindex(common_dates, method='ffill').reset_index()
    
    def get_data_for_timepoint(self, time_point: Union[str, datetime]) -> Dict[str, Dict]:
        """
        Get all data for a specific time point.
        
        Parameters
        ----------
        time_point : str or datetime
            Time point for which to retrieve data
            
        Returns
        -------
        dict
            Dictionary containing node_data, edge_data, and system_data for the specified time point
        """
        if isinstance(time_point, str):
            time_point = pd.to_datetime(time_point)
        
        result = {
            'node_data': {},
            'edge_data': {},
            'system_data': {}
        }
        
        # Extract node data
        for category, data in self.node_data.items():
            if isinstance(data.index, pd.DatetimeIndex):
                if time_point in data.index:
                    result['node_data'][category] = data.loc[time_point]
                else:
                    # Find closest previous date
                    prev_dates = [d for d in data.index if d <= time_point]
                    if prev_dates:
                        closest_date = max(prev_dates)
                        result['node_data'][category] = data.loc[closest_date]
            elif 'date' in data.columns:
                date_data = data[data['date'] == time_point]
                if not date_data.empty:
                    result['node_data'][category] = date_data
                else:
                    # Find closest previous date
                    prev_dates = [d for d in pd.to_datetime(data['date']) if d <= time_point]
                    if prev_dates:
                        closest_date = max(prev_dates)
                        result['node_data'][category] = data[data['date'] == closest_date]
        
        # Extract edge data
        for key, data in self.edge_data.items():
            if 'date' in data.columns:
                date_data = data[data['date'] == time_point]
                if not date_data.empty:
                    result['edge_data'][key] = date_data
                else:
                    # Find closest previous date
                    prev_dates = [d for d in pd.to_datetime(data['date']) if d <= time_point]
                    if prev_dates:
                        closest_date = max(prev_dates)
                        result['edge_data'][key] = data[data['date'] == closest_date]
        
        # Extract system data
        for key, data in self.system_data.items():
            if isinstance(data.index, pd.DatetimeIndex):
                if time_point in data.index:
                    result['system_data'][key] = data.loc[time_point]
                else:
                    # Find closest previous date
                    prev_dates = [d for d in data.index if d <= time_point]
                    if prev_dates:
                        closest_date = max(prev_dates)
                        result['system_data'][key] = data.loc[closest_date]
            elif 'date' in data.columns:
                date_data = data[data['date'] == time_point]
                if not date_data.empty:
                    result['system_data'][key] = date_data
                else:
                    # Find closest previous date
                    prev_dates = [d for d in pd.to_datetime(data['date']) if d <= time_point]
                    if prev_dates:
                        closest_date = max(prev_dates)
                        result['system_data'][key] = data[data['date'] == closest_date]
        
        return result
