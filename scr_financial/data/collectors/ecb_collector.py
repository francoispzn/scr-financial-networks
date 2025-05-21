"""
ECB data collector for financial network analysis.

This module provides the ECBCollector class for retrieving data from the
European Central Bank (ECB) Statistical Data Warehouse and other ECB sources.
"""

import pandas as pd
import numpy as np
import requests
import io
from typing import List, Optional
from datetime import datetime


class ECBCollector:
    """
    Collector for European Central Bank (ECB) data.
    
    This class provides methods for retrieving data from ECB sources, including
    TARGET2 interbank payment data, CISS systemic stress indicator, and G-SIB data.
    """
    
    def __init__(self):
        """Initialize the ECB data collector."""
        self.sdw_base_url = "https://sdw.ecb.europa.eu/quickview.do?SERIES_KEY="
        self.target2_base_url = "https://www.ecb.europa.eu/paym/target/target2/profuse/download/html/"
    
    def collect_target2_data(
        self, 
        start_date: str, 
        end_date: str, 
        bank_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Collect TARGET2 interbank payment data.
        
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
            DataFrame containing interbank payment volumes
        """
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # For now, use sample data
        return self._get_sample_target2_data(start_date, end_date, bank_list)
    
    def collect_ciss_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Collect ECB Composite Indicator of Systemic Stress (CISS) data.
        
        Parameters
        ----------
        start_date : str
            Start date for data collection in 'YYYY-MM-DD' format
        end_date : str
            End date for data collection in 'YYYY-MM-DD' format
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing CISS values
        """
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # For now, use sample data
        return self._get_sample_ciss_data(start_date, end_date)
    
    def collect_gsib_data(
        self, 
        start_date: str, 
        end_date: str, 
        bank_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Collect G-SIB (Global Systemically Important Bank) indicator data.
        
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
            DataFrame containing G-SIB indicators
        """
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # For now, use sample data
        return self._get_sample_gsib_data(start_date, end_date, bank_list)
    
    def _get_sample_target2_data(
        self, 
        start_date: str, 
        end_date: str, 
        bank_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate sample TARGET2 data for testing.
        
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
            DataFrame containing sample interbank payment volumes
        """
        # Default bank list if none provided
        if bank_list is None:
            bank_list = [
                "DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
                "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA"
            ]
        
        # Generate dates within the range (monthly)
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Create sample data
        data = []
        
        # Base exposure matrix (symmetric)
        base_exposures = {
            "DE_DBK": {"FR_BNP": 15, "ES_SAN": 8, "IT_UCG": 12, "NL_ING": 10, "SE_NDA": 5, "CH_UBS": 7, "UK_BARC": 9, "UK_HSBC": 11, "FR_ACA": 6},
            "FR_BNP": {"DE_DBK": 15, "ES_SAN": 14, "IT_UCG": 9, "NL_ING": 8, "SE_NDA": 4, "CH_UBS": 6, "UK_BARC": 10, "UK_HSBC": 12, "FR_ACA": 18},
            "ES_SAN": {"DE_DBK": 8, "FR_BNP": 14, "IT_UCG": 13, "NL_ING": 7, "SE_NDA": 3, "CH_UBS": 5, "UK_BARC": 8, "UK_HSBC": 9, "FR_ACA": 7},
            "IT_UCG": {"DE_DBK": 12, "FR_BNP": 9, "ES_SAN": 13, "NL_ING": 6, "SE_NDA": 4, "CH_UBS": 5, "UK_BARC": 7, "UK_HSBC": 8, "FR_ACA": 6},
            "NL_ING": {"DE_DBK": 10, "FR_BNP": 8, "ES_SAN": 7, "IT_UCG": 6, "SE_NDA": 9, "CH_UBS": 8, "UK_BARC": 6, "UK_HSBC": 7, "FR_ACA": 5},
            "SE_NDA": {"DE_DBK": 5, "FR_BNP": 4, "ES_SAN": 3, "IT_UCG": 4, "NL_ING": 9, "CH_UBS": 7, "UK_BARC": 4, "UK_HSBC": 5, "FR_ACA": 3},
            "CH_UBS": {"DE_DBK": 7, "FR_BNP": 6, "ES_SAN": 5, "IT_UCG": 5, "NL_ING": 8, "SE_NDA": 7, "UK_BARC": 9, "UK_HSBC": 10, "FR_ACA": 4},
            "UK_BARC": {"DE_DBK": 9, "FR_BNP": 10, "ES_SAN": 8, "IT_UCG": 7, "NL_ING": 6, "SE_NDA": 4, "CH_UBS": 9, "UK_HSBC": 15, "FR_ACA": 7},
            "UK_HSBC": {"DE_DBK": 11, "FR_BNP": 12, "ES_SAN": 9, "IT_UCG": 8, "NL_ING": 7, "SE_NDA": 5, "CH_UBS": 10, "UK_BARC": 15, "FR_ACA": 8},
            "FR_ACA": {"DE_DBK": 6, "FR_BNP": 18, "ES_SAN": 7, "IT_UCG": 6, "NL_ING": 5, "SE_NDA": 3, "CH_UBS": 4, "UK_BARC": 7, "UK_HSBC": 8}
        }
        
        for date in date_range:
            # Time factor for trend
            time_factor = (date - pd.to_datetime(start_date)).days / 365
            
            for source in bank_list:
                for target in bank_list:
                    if source != target:
                        # Get base exposure with some time trend and noise
                        if source in base_exposures and target in base_exposures[source]:
                            base = base_exposures[source][target]
                            # Add trend (increasing over time) and noise
                            exposure = base * (1 + 0.1 * time_factor) * (1 + np.random.normal(0, 0.15))
                            # Convert to billions of EUR
                            exposure = exposure * 1e9
                            
                            data.append({
                                'date': date,
                                'source': source,
                                'target': target,
                                'weight': round(exposure, 0)
                            })
        
        return pd.DataFrame(data)
    
    def _get_sample_ciss_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate sample CISS data for testing.
        
        Parameters
        ----------
        start_date : str
            Start date for data collection in 'YYYY-MM-DD' format
        end_date : str
            End date for data collection in 'YYYY-MM-DD' format
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing sample CISS values
        """
        # Generate dates within the range (weekly)
        date_range = pd.date_range(start=start_date, end=end_date, freq='W')
        
        # Create sample data
        data = []
        
        # Base CISS value
        base_ciss = 0.15
        
        # Add some random variation and events
        for date in date_range:
            # Time factor
            time_factor = (date - pd.to_datetime(start_date)).days / 365
            
            # Add some stress events
            if "2008-09" in str(date) or "2008-10" in str(date):  # Financial crisis
                ciss = 0.8 + np.random.normal(0, 0.05)
            elif "2011-08" in str(date) or "2011-09" in str(date):  # European debt crisis
                ciss = 0.6 + np.random.normal(0, 0.05)
            elif "2020-03" in str(date) or "2020-04" in str(date):  # COVID-19
                ciss = 0.7 + np.random.normal(0, 0.05)
            else:
                # Normal times with slight trend and noise
                ciss = base_ciss + 0.02 * time_factor + np.random.normal(0, 0.03)
            
            # Ensure CISS is between 0 and 1
            ciss = max(0, min(1, ciss))
            
            data.append({
                'date': date,
                'CISS': round(ciss, 4)
            })
        
        return pd.DataFrame(data)
    
    def _get_sample_gsib_data(
        self, 
        start_date: str, 
        end_date: str, 
        bank_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate sample G-SIB data for testing.
        
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
            DataFrame containing sample G-SIB indicators
        """
        # Default bank list if none provided
        if bank_list is None:
            bank_list = [
                "DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
                "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA"
            ]
        
        # Generate dates within the range (annual)
        date_range = pd.date_range(start=start_date, end=end_date, freq='A')
        
        # Create sample data
        data = []
        
        # Base G-SIB scores for each bank
        base_scores = {
            "DE_DBK": 210,    # Deutsche Bank
            "FR_BNP": 230,    # BNP Paribas
            "ES_SAN": 190,    # Santander
            "IT_UCG": 170,    # UniCredit
            "NL_ING": 160,    # ING
            "SE_NDA": 120,    # Nordea
            "CH_UBS": 200,    # UBS
            "UK_BARC": 180,   # Barclays
            "UK_HSBC": 250,   # HSBC
            "FR_ACA": 170     # Credit Agricole
        }
        
        # Base cross-jurisdictional claims and liabilities
        base_cross_claims = {
            "DE_DBK": 450,    # Deutsche Bank
            "FR_BNP": 480,    # BNP Paribas
            "ES_SAN": 320,    # Santander
            "IT_UCG": 280,    # UniCredit
            "NL_ING": 250,    # ING
            "SE_NDA": 180,    # Nordea
            "CH_UBS": 420,    # UBS
            "UK_BARC": 350,   # Barclays
            "UK_HSBC": 520,   # HSBC
            "FR_ACA": 290     # Credit Agricole
        }
        
        # Base intra-financial assets and liabilities
        base_intra_assets = {
            "DE_DBK": 380,    # Deutsche Bank
            "FR_BNP": 410,    # BNP Paribas
            "ES_SAN": 290,    # Santander
            "IT_UCG": 250,    # UniCredit
            "NL_ING": 220,    # ING
            "SE_NDA": 160,    # Nordea
            "CH_UBS": 370,    # UBS
            "UK_BARC": 320,   # Barclays
            "UK_HSBC": 450,   # HSBC
            "FR_ACA": 260     # Credit Agricole
        }
        
        for date in date_range:
            # Time factor for trend
            time_factor = (date - pd.to_datetime(start_date)).days / 365
            
            for bank_id in bank_list:
                # Get base values
                base_score = base_scores.get(bank_id, 150)
                base_cross = base_cross_claims.get(bank_id, 300)
                base_intra = base_intra_assets.get(bank_id, 250)
                
                # Add trend and noise
                gsib_score = base_score * (1 + 0.05 * time_factor) * (1 + np.random.normal(0, 0.05))
                cross_claims = base_cross * (1 + 0.08 * time_factor) * (1 + np.random.normal(0, 0.08))
                intra_assets = base_intra * (1 + 0.06 * time_factor) * (1 + np.random.normal(0, 0.07))
                
                # Convert to billions of EUR
                cross_claims = cross_claims * 1e9
                intra_assets = intra_assets * 1e9
                
                data.append({
                    'date': date,
                    'bank_id': bank_id,
                    'gsib_score': round(gsib_score, 0),
                    'cross_jurisdictional_claims': round(cross_claims, 0),
                    'intra_financial_assets': round(intra_assets, 0)
                })
        
        return pd.DataFrame(data)
