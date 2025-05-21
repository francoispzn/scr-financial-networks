"""
Market data collector for financial network analysis.

This module provides the MarketDataCollector class for retrieving market data
such as CDS spreads, equity prices, and other market indicators.
"""

import pandas as pd
import numpy as np
import requests
import io
from typing import List, Optional
from datetime import datetime


class MarketDataCollector:
    """
    Collector for market data.
    
    This class provides methods for retrieving market data such as CDS spreads,
    equity prices, SRISK, and other market indicators from various sources.
    """
    
    def __init__(self):
        """Initialize the market data collector."""
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.vlab_base_url = "https://vlab.stern.nyu.edu/api/v2"
        self.bis_base_url = "https://www.bis.org/statistics/api/data"
    
    def collect_cds_data(
        self, 
        start_date: str, 
        end_date: str, 
        bank_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Collect CDS spread data.
        
        Parameters
        ----------
        start_date : str
            Start date for data collection in 'YYYY-MM-DD' format
        end_date : str
            End date for data collection in 'YYYY-MM-DD' format
        bank_list : list of str, optional
            List of bank identifiers to include
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing CDS spreads
        """
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # For now, use sample data
        return self._get_sample_cds_data(start_date, end_date, bank_list)
    
    def collect_srisk_data(
        self, 
        start_date: str, 
        end_date: str, 
        bank_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Collect SRISK data from NYU V-Lab.
        
        Parameters
        ----------
        start_date : str
            Start date for data collection in 'YYYY-MM-DD' format
        end_date : str
            End date for data collection in 'YYYY-MM-DD' format
        bank_list : list of str, optional
            List of bank identifiers to include
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing SRISK values
        """
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # For now, use sample data
        return self._get_sample_srisk_data(start_date, end_date, bank_list)
    
    def collect_equity_data(
        self, 
        start_date: str, 
        end_date: str, 
        bank_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Collect equity price data.
        
        Parameters
        ----------
        start_date : str
            Start date for data collection in 'YYYY-MM-DD' format
        end_date : str
            End date for data collection in 'YYYY-MM-DD' format
        bank_list : list of str, optional
            List of bank identifiers to include
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing equity prices
        """
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # For now, use sample data
        return self._get_sample_equity_data(start_date, end_date, bank_list)
    
    def collect_funding_stress_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Collect funding stress indicators.
        
        Parameters
        ----------
        start_date : str
            Start date for data collection in 'YYYY-MM-DD' format
        end_date : str
            End date for data collection in 'YYYY-MM-DD' format
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing funding stress indicators
        """
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # For now, use sample data
        return self._get_sample_funding_stress_data(start_date, end_date)
    
    def collect_bis_data(self, start_date: str, end_date: str, indicator: str) -> pd.DataFrame:
        """
        Collect data from Bank for International Settlements (BIS).
        
        Parameters
        ----------
        start_date : str
            Start date for data collection in 'YYYY-MM-DD' format
        end_date : str
            End date for data collection in 'YYYY-MM-DD' format
        indicator : str
            Indicator to collect (e.g., 'credit_to_GDP_gap')
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing BIS data
        """
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # For now, use sample data
        if indicator == 'credit_to_GDP_gap':
            return self._get_sample_credit_gap_data(start_date, end_date)
        elif indicator == 'house_price_gap':
            return self._get_sample_house_price_gap_data(start_date, end_date)
        else:
            raise ValueError(f"Unknown BIS indicator: {indicator}")
    
    def _get_sample_cds_data(
        self, 
        start_date: str, 
        end_date: str, 
        bank_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate sample CDS data for testing.
        
        Parameters
        ----------
        start_date : str
            Start date for data collection in 'YYYY-MM-DD' format
        end_date : str
            End date for data collection in 'YYYY-MM-DD' format
        bank_list : list of str, optional
            List of bank identifiers to include
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing sample CDS spreads
        """
        # Default bank list if none provided
        if bank_list is None:
            bank_list = [
                "DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
                "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA"
            ]
        
        # Generate dates within the range (daily)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Create sample data
        data = []
        
        # Base CDS spreads for each bank
        base_cds = {
            "DE_DBK": 85,     # Deutsche Bank
            "FR_BNP": 70,     # BNP Paribas
            "ES_SAN": 90,     # Santander
            "IT_UCG": 100,    # UniCredit
            "NL_ING": 65,     # ING
            "SE_NDA": 55,     # Nordea
            "CH_UBS": 60,     # UBS
            "UK_BARC": 80,    # Barclays
            "UK_HSBC": 65,    # HSBC
            "FR_ACA": 75      # Credit Agricole
        }
        
        for date in date_range:
            # Add some stress events
            if "2008-09" in str(date) or "2008-10" in str(date):  # Financial crisis
                stress_factor = 5.0
            elif "2011-08" in str(date) or "2011-09" in str(date):  # European debt crisis
                stress_factor = 3.0
            elif "2020-03" in str(date) or "2020-04" in str(date):  # COVID-19
                stress_factor = 4.0
            else:
                stress_factor = 1.0
            
            for bank_id in bank_list:
                # Get base CDS
                base = base_cds.get(bank_id, 75)
                
                # Add noise and stress factor
                cds = base * stress_factor * (1 + np.random.normal(0, 0.05))
                
                data.append({
                    'date': date,
                    'bank_id': bank_id,
                    'CDS_5yr': round(cds, 1)
                })
        
        return pd.DataFrame(data)
    
    def _get_sample_srisk_data(
        self, 
        start_date: str, 
        end_date: str, 
        bank_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate sample SRISK data for testing.
        
        Parameters
        ----------
        start_date : str
            Start date for data collection in 'YYYY-MM-DD' format
        end_date : str
            End date for data collection in 'YYYY-MM-DD' format
        bank_list : list of str, optional
            List of bank identifiers to include
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing sample SRISK values
        """
        # Default bank list if none provided
        if bank_list is None:
            bank_list = [
                "DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
                "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA"
            ]
        
        # Generate dates within the range (weekly)
        date_range = pd.date_range(start=start_date, end=end_date, freq='W')
        
        # Create sample data
        data = []
        
        # Base SRISK for each bank (in billions EUR)
        base_srisk = {
            "DE_DBK": 35,     # Deutsche Bank
            "FR_BNP": 40,     # BNP Paribas
            "ES_SAN": 30,     # Santander
            "IT_UCG": 28,     # UniCredit
            "NL_ING": 22,     # ING
            "SE_NDA": 15,     # Nordea
            "CH_UBS": 25,     # UBS
            "UK_BARC": 32,    # Barclays
            "UK_HSBC": 38,    # HSBC
            "FR_ACA": 27      # Credit Agricole
        }
        
        for date in date_range:
            # Add some stress events
            if "2008-09" in str(date) or "2008-10" in str(date):  # Financial crisis
                stress_factor = 2.5
            elif "2011-08" in str(date) or "2011-09" in str(date):  # European debt crisis
                stress_factor = 1.8
            elif "2020-03" in str(date) or "2020-04" in str(date):  # COVID-19
                stress_factor = 2.0
            else:
                stress_factor = 1.0
            
            for bank_id in bank_list:
                # Get base SRISK
                base = base_srisk.get(bank_id, 25)
                
                # Add noise and stress factor
                srisk = base * stress_factor * (1 + np.random.normal(0, 0.08))
                
                # Convert to billions EUR
                srisk = srisk * 1e9
                
                data.append({
                    'date': date,
                    'bank_id': bank_id,
                    'SRISK': round(srisk, 0)
                })
        
        return pd.DataFrame(data)
    
    def _get_sample_equity_data(
        self, 
        start_date: str, 
        end_date: str, 
        bank_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate sample equity price data for testing.
        
        Parameters
        ----------
        start_date : str
            Start date for data collection in 'YYYY-MM-DD' format
        end_date : str
            End date for data collection in 'YYYY-MM-DD' format
        bank_list : list of str, optional
            List of bank identifiers to include
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing sample equity prices
        """
        # Default bank list if none provided
        if bank_list is None:
            bank_list = [
                "DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
                "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA"
            ]
        
        # Generate dates within the range (daily)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Create sample data
        data = []
        
        # Base equity prices for each bank (in EUR)
        base_price = {
            "DE_DBK": 12.5,    # Deutsche Bank
            "FR_BNP": 55.0,    # BNP Paribas
            "ES_SAN": 3.8,     # Santander
            "IT_UCG": 15.2,    # UniCredit
            "NL_ING": 12.8,    # ING
            "SE_NDA": 10.5,    # Nordea
            "CH_UBS": 18.2,    # UBS
            "UK_BARC": 2.1,    # Barclays
            "UK_HSBC": 6.5,    # HSBC
            "FR_ACA": 11.8     # Credit Agricole
        }
        
        # Generate price series with random walk
        for bank_id in bank_list:
            # Get base price
            price = base_price.get(bank_id, 10.0)
            
            # Generate price series with random walk
            prices = [price]
            for i in range(1, len(date_range)):
                # Add some stress events
                if "2008-09" in str(date_range[i]) or "2008-10" in str(date_range[i]):  # Financial crisis
                    drift = -0.01
                    volatility = 0.03
                elif "2011-08" in str(date_range[i]) or "2011-09" in str(date_range[i]):  # European debt crisis
                    drift = -0.005
                    volatility = 0.025
                elif "2020-03" in str(date_range[i]) or "2020-04" in str(date_range[i]):  # COVID-19
                    drift = -0.015
                    volatility = 0.035
                else:
                    drift = 0.0002
                    volatility = 0.015
                
                # Random walk with drift
                change = drift + np.random.normal(0, volatility)
                new_price = prices[-1] * (1 + change)
                # Ensure price doesn't go below 0.1
                prices.append(max(0.1, new_price))
            
            # Add to data
            for i, date in enumerate(date_range):
                data.append({
                    'date': date,
                    'bank_id': bank_id,
                    'price': round(prices[i], 2),
                    'volume': int(np.random.uniform(1000000, 10000000))
                })
        
        return pd.DataFrame(data)
    
    def _get_sample_funding_stress_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate sample funding stress indicators for testing.
        
        Parameters
        ----------
        start_date : str
            Start date for data collection in 'YYYY-MM-DD' format
        end_date : str
            End date for data collection in 'YYYY-MM-DD' format
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing sample funding stress indicators
        """
        # Generate dates within the range (daily)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Create sample data
        data = []
        
        # Base values
        base_libor_ois = 0.15  # LIBOR-OIS spread in percentage points
        base_gc_tbill = 0.08   # GC Repo - T-Bill spread in percentage points
        
        for date in date_range:
            # Add some stress events
            if "2008-09" in str(date) or "2008-10" in str(date):  # Financial crisis
                stress_factor = 5.0
            elif "2011-08" in str(date) or "2011-09" in str(date):  # European debt crisis
                stress_factor = 3.0
            elif "2020-03" in str(date) or "2020-04" in str(date):  # COVID-19
                stress_factor = 4.0
            else:
                stress_factor = 1.0
            
            # Add noise and stress factor
            libor_ois = base_libor_ois * stress_factor * (1 + np.random.normal(0, 0.1))
            gc_tbill = base_gc_tbill * stress_factor * (1 + np.random.normal(0, 0.12))
            
            data.append({
                'date': date,
                'LIBOR_OIS_spread': round(libor_ois, 3),
                'GC_Repo_TBill_spread': round(gc_tbill, 3)
            })
        
        return pd.DataFrame(data)
    
    def _get_sample_credit_gap_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate sample credit-to-GDP gap data for testing.
        
        Parameters
        ----------
        start_date : str
            Start date for data collection in 'YYYY-MM-DD' format
        end_date : str
            End date for data collection in 'YYYY-MM-DD' format
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing sample credit-to-GDP gap data
        """
        # Generate dates within the range (quarterly)
        date_range = pd.date_range(start=start_date, end=end_date, freq='Q')
        
        # Create sample data
        data = []
        
        # Base values for different countries
        base_gaps = {
            'USA': 2.5,
            'EUR': 3.2,
            'GBR': 1.8,
            'CHE': 4.5,
            'JPN': -0.5
        }
        
        for date in date_range:
            # Time factor for trend
            time_factor = (date - pd.to_datetime(start_date)).days / 365
            
            for country, base_gap in base_gaps.items():
                # Add trend and noise
                if "2008" in str(date) or "2009" in str(date):  # Financial crisis
                    gap = base_gap + 5.0 + 0.5 * time_factor + np.random.normal(0, 1.0)
                elif "2011" in str(date) or "2012" in str(date):  # European debt crisis
                    gap = base_gap + 3.0 + 0.5 * time_factor + np.random.normal(0, 0.8)
                else:
                    gap = base_gap + 0.5 * time_factor + np.random.normal(0, 0.5)
                
                data.append({
                    'date': date,
                    'country': country,
                    'credit_to_GDP_gap': round(gap, 2)
                })
        
        return pd.DataFrame(data)
    
    def _get_sample_house_price_gap_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate sample house price gap data for testing.
        
        Parameters
        ----------
        start_date : str
            Start date for data collection in 'YYYY-MM-DD' format
        end_date : str
            End date for data collection in 'YYYY-MM-DD' format
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing sample house price gap data
        """
        # Generate dates within the range (quarterly)
        date_range = pd.date_range(start=start_date, end=end_date, freq='Q')
        
        # Create sample data
        data = []
        
        # Base values for different countries
        base_gaps = {
            'USA': 3.5,
            'EUR': 2.8,
            'GBR': 4.2,
            'CHE': 5.5,
            'JPN': -1.5
        }
        
        for date in date_range:
            # Time factor for trend
            time_factor = (date - pd.to_datetime(start_date)).days / 365
            
            for country, base_gap in base_gaps.items():
                # Add trend and noise
                if "2008" in str(date) or "2009" in str(date):  # Financial crisis
                    gap = base_gap - 4.0 + 0.3 * time_factor + np.random.normal(0, 1.2)
                elif "2011" in str(date) or "2012" in str(date):  # European debt crisis
                    gap = base_gap - 2.0 + 0.3 * time_factor + np.random.normal(0, 1.0)
                else:
                    gap = base_gap + 0.3 * time_factor + np.random.normal(0, 0.7)
                
                data.append({
                    'date': date,
                    'country': country,
                    'house_price_gap': round(gap, 2)
                })
        
        return pd.DataFrame(data)
