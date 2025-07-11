"""Pre-defined crisis and stress period windows for financial analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CrisisPeriod:
    label: str
    start: str          # ISO date  YYYY-MM-DD
    end: str            # ISO date  YYYY-MM-DD
    description: str


CRISIS_PERIODS: Dict[str, CrisisPeriod] = {
    # Global Financial Crisis (GFC)
    "gfc": CrisisPeriod(
        label="Global Financial Crisis",
        start="2007-07-01",
        end="2009-03-31",
        description="Sub-prime mortgage collapse, Lehman Brothers failure, global credit freeze.",
    ),
    "gfc_acute": CrisisPeriod(
        label="GFC Acute Phase",
        start="2008-09-01",
        end="2009-03-31",
        description="Lehman bankruptcy through market trough.",
    ),
    # European sovereign debt crisis
    "euro_sovereign": CrisisPeriod(
        label="European Sovereign Debt Crisis",
        start="2010-04-01",
        end="2012-07-31",
        description="Greek, Irish, Portuguese bailouts; 'whatever it takes' speech July 2012.",
    ),
    "euro_sovereign_acute": CrisisPeriod(
        label="Euro Sovereign Acute",
        start="2011-07-01",
        end="2012-01-31",
        description="Italian/Spanish spread blow-out, MF Global, LTRO anticipation.",
    ),
    # COVID-19
    "covid": CrisisPeriod(
        label="COVID-19 Market Crash",
        start="2020-02-20",
        end="2020-06-30",
        description="Pandemic sell-off and initial recovery.",
    ),
    "covid_acute": CrisisPeriod(
        label="COVID-19 Acute Phase",
        start="2020-02-20",
        end="2020-03-23",
        description="Fastest bear market in history.",
    ),
    # 2023 banking stress
    "banking_stress_2023": CrisisPeriod(
        label="2023 Banking Stress (SVB / CS)",
        start="2023-03-01",
        end="2023-05-31",
        description="SVB, Signature Bank failures; Credit Suisse forced merger with UBS.",
    ),
    # Taper tantrum
    "taper_tantrum": CrisisPeriod(
        label="Taper Tantrum",
        start="2013-05-01",
        end="2013-09-30",
        description="Bond sell-off after Bernanke hints at QE tapering.",
    ),
    # China devaluation / commodity crash
    "china_deval_2015": CrisisPeriod(
        label="China Devaluation / Commodity Crash",
        start="2015-06-01",
        end="2016-02-28",
        description="Chinese equity bubble burst, CNY devaluation, oil price collapse.",
    ),
    # Brexit vote
    "brexit": CrisisPeriod(
        label="Brexit Referendum",
        start="2016-06-15",
        end="2016-07-15",
        description="UK votes to leave the EU; sharp GBP and banking sell-off.",
    ),
    # 2022 rate shock
    "rate_shock_2022": CrisisPeriod(
        label="2022 Rate Shock",
        start="2022-01-01",
        end="2022-10-31",
        description="Aggressive Fed/ECB rate hikes; UK gilt crisis; bond-equity correlation flip.",
    ),
    # Calm / baseline periods (useful for comparison)
    "calm_2017": CrisisPeriod(
        label="2017 Low-Vol Baseline",
        start="2017-01-01",
        end="2017-12-31",
        description="Historically low VIX year; useful calm-period baseline.",
    ),
    "calm_2019": CrisisPeriod(
        label="2019 Pre-COVID Baseline",
        start="2019-01-01",
        end="2019-12-31",
        description="Steady risk-on rally before pandemic.",
    ),
}


def get_crisis(name: str) -> CrisisPeriod:
    """Retrieve a crisis period by key."""
    if name not in CRISIS_PERIODS:
        raise KeyError(
            f"Unknown crisis period '{name}'. Available: {list(CRISIS_PERIODS.keys())}"
        )
    return CRISIS_PERIODS[name]


def list_crises() -> list[str]:
    """Return all available crisis period keys."""
    return list(CRISIS_PERIODS.keys())
