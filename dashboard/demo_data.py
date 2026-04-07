"""
Pre-built shock scenarios for the SCR Financial Networks dashboard.

These scenarios are applied on top of whatever data has been loaded
from the pipeline — no bank states are hardcoded here.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Banks in the default universe
_ALL_BANKS: List[str] = [
    "DE_DBK", "FR_BNP", "ES_SAN", "IT_UCG", "NL_ING",
    "SE_NDA", "CH_UBS", "UK_BARC", "UK_HSBC", "FR_ACA",
]

SHOCK_SCENARIOS: Dict[str, Any] = {
    "mild_credit": {
        "label": "Mild Credit Shock",
        "description": "Moderate credit quality deterioration across all banks (−1.5 pp CET1).",
        "params": {bank: {"CET1_ratio": -1.5} for bank in _ALL_BANKS},
    },
    "severe_credit": {
        "label": "Severe Credit Shock",
        "description": "Severe credit losses across all banks (−3 to −4 pp CET1).",
        "params": {
            "DE_DBK": {"CET1_ratio": -3.0},
            "FR_BNP": {"CET1_ratio": -3.5},
            "ES_SAN": {"CET1_ratio": -4.0},
            "IT_UCG": {"CET1_ratio": -3.0},
            "NL_ING": {"CET1_ratio": -2.5},
            "SE_NDA": {"CET1_ratio": -2.0},
            "CH_UBS": {"CET1_ratio": -2.5},
            "UK_BARC": {"CET1_ratio": -3.5},
            "UK_HSBC": {"CET1_ratio": -2.0},
            "FR_ACA": {"CET1_ratio": -3.5},
            "FR_SG":  {"CET1_ratio": -3.0},
        },
    },
    "liquidity_crunch": {
        "label": "Liquidity Crunch",
        "description": "Sudden liquidity squeeze — all LCR ratios fall −35 pp.",
        "params": {bank: {"LCR": -35} for bank in _ALL_BANKS},
    },
    "sovereign_stress": {
        "label": "Sovereign Stress",
        "description": "Southern European sovereign spread widening hits Santander and UniCredit.",
        "params": {
            "ES_SAN": {"CET1_ratio": -2.5, "LCR": -20},
            "IT_UCG": {"CET1_ratio": -3.0, "LCR": -25},
            "system": {"CISS": 0.25, "funding_stress": 0.15},
        },
    },
    "systemic_crisis": {
        "label": "Systemic Crisis",
        "description": "Full system stress: credit losses + liquidity crunch + sentiment collapse.",
        "params": {
            **{bank: {"CET1_ratio": -4.0, "LCR": -50} for bank in _ALL_BANKS},
            "system": {"CISS": 0.60, "funding_stress": 0.45},
        },
    },
}
