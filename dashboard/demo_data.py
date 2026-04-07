"""
Demo data for the SCR Financial Networks dashboard.

Provides a realistic 6-bank European interbank system for demonstration purposes.
All figures are illustrative and based on publicly reported regulatory ratios.
"""

# ── Bank initial states ──────────────────────────────────────────────────────
# Keys mirror those expected by BankAgent / BankingSystemSimulation.

BANK_DATA: dict = {
    "DE_DBK": {
        "name": "Deutsche Bank",
        "country": "DE",
        "CET1_ratio": 13.7,
        "LCR": 148,
        "NSFR": 119,
        "total_assets": 1.32e12,
        "cash": 1.8e11,
        "interbank_assets": 2.4e11,
        "interbank_liabilities": 2.1e11,
    },
    "FR_BNP": {
        "name": "BNP Paribas",
        "country": "FR",
        "CET1_ratio": 13.1,
        "LCR": 142,
        "NSFR": 116,
        "total_assets": 2.67e12,
        "cash": 3.2e11,
        "interbank_assets": 4.1e11,
        "interbank_liabilities": 3.8e11,
    },
    "FR_SG": {
        "name": "Société Générale",
        "country": "FR",
        "CET1_ratio": 13.5,
        "LCR": 155,
        "NSFR": 121,
        "total_assets": 1.43e12,
        "cash": 1.9e11,
        "interbank_assets": 2.2e11,
        "interbank_liabilities": 1.9e11,
    },
    "ES_SAN": {
        "name": "Banco Santander",
        "country": "ES",
        "CET1_ratio": 12.3,
        "LCR": 158,
        "NSFR": 122,
        "total_assets": 1.81e12,
        "cash": 2.1e11,
        "interbank_assets": 2.8e11,
        "interbank_liabilities": 2.5e11,
    },
    "IT_UCG": {
        "name": "UniCredit",
        "country": "IT",
        "CET1_ratio": 16.1,
        "LCR": 162,
        "NSFR": 130,
        "total_assets": 7.90e11,
        "cash": 9.5e10,
        "interbank_assets": 1.2e11,
        "interbank_liabilities": 1.0e11,
    },
    "NL_ING": {
        "name": "ING Group",
        "country": "NL",
        "CET1_ratio": 14.8,
        "LCR": 145,
        "NSFR": 125,
        "total_assets": 9.95e11,
        "cash": 1.3e11,
        "interbank_assets": 1.8e11,
        "interbank_liabilities": 1.5e11,
    },
}

# ── Interbank exposure network (directed: lender → {borrower: amount €}) ────
NETWORK_DATA: dict = {
    "DE_DBK": {"FR_BNP": 4.5e10, "FR_SG": 2.0e10, "NL_ING": 3.0e10},
    "FR_BNP": {"DE_DBK": 4.5e10, "ES_SAN": 5.0e10, "IT_UCG": 2.5e10, "NL_ING": 3.5e10},
    "FR_SG":  {"DE_DBK": 2.0e10, "FR_BNP": 3.0e10, "IT_UCG": 1.5e10},
    "ES_SAN": {"FR_BNP": 5.0e10, "IT_UCG": 2.0e10, "NL_ING": 2.5e10},
    "IT_UCG": {"FR_BNP": 2.5e10, "FR_SG": 1.5e10, "ES_SAN": 2.0e10},
    "NL_ING": {"DE_DBK": 3.0e10, "FR_BNP": 3.5e10, "ES_SAN": 2.5e10},
}

# ── Initial system-wide indicators ──────────────────────────────────────────
SYSTEM_INDICATORS: dict = {
    "CISS": 0.18,          # Composite Indicator of Systemic Stress (ECB)
    "funding_stress": 0.12,
    "VIX": 18.5,
    "EONIA_spread": 0.05,
}

# ── Pre-built shock scenarios ────────────────────────────────────────────────
SHOCK_SCENARIOS: dict = {
    "mild_credit": {
        "label": "Mild Credit Shock",
        "description": "Moderate credit quality deterioration across all banks.",
        "params": {bank: {"CET1_ratio": -1.5} for bank in BANK_DATA},
    },
    "severe_credit": {
        "label": "Severe Credit Shock",
        "description": "Severe credit losses; two weakest banks approach insolvency threshold.",
        "params": {
            "DE_DBK": {"CET1_ratio": -3.0},
            "FR_BNP": {"CET1_ratio": -3.5},
            "FR_SG":  {"CET1_ratio": -2.0},
            "ES_SAN": {"CET1_ratio": -4.0},
            "IT_UCG": {"CET1_ratio": -2.5},
            "NL_ING": {"CET1_ratio": -2.0},
        },
    },
    "liquidity_crunch": {
        "label": "Liquidity Crunch",
        "description": "Sudden liquidity squeeze; all LCR ratios fall sharply.",
        "params": {bank: {"LCR": -35} for bank in BANK_DATA},
    },
    "sovereign_stress": {
        "label": "Sovereign Stress",
        "description": "Southern European sovereign spread widening.",
        "params": {
            "ES_SAN": {"CET1_ratio": -2.5, "LCR": -20},
            "IT_UCG": {"CET1_ratio": -3.0, "LCR": -25},
            "system": {"CISS": 0.25, "funding_stress": 0.15},
        },
    },
    "systemic_crisis": {
        "label": "Systemic Crisis",
        "description": "Full system stress: credit losses, liquidity crunch, sentiment collapse.",
        "params": {
            **{bank: {"CET1_ratio": -4.0, "LCR": -50} for bank in BANK_DATA},
            "system": {"CISS": 0.60, "funding_stress": 0.45},
        },
    },
}
