"""Load bank universe definitions from YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
_UNIVERSES_FILE = _CONFIG_DIR / "bank_universes.yaml"


@dataclass(frozen=True)
class Bank:
    id: str
    ticker: str
    name: str
    country: str
    currency: str
    region: str


@dataclass
class BankUniverse:
    name: str
    banks: List[Bank] = field(default_factory=list)

    # convenience accessors ------------------------------------------------
    @property
    def tickers(self) -> List[str]:
        return [b.ticker for b in self.banks]

    @property
    def ids(self) -> List[str]:
        return [b.id for b in self.banks]

    @property
    def currencies(self) -> set:
        return {b.currency for b in self.banks}

    def filter_region(self, region: str) -> "BankUniverse":
        return BankUniverse(
            name=f"{self.name}_{region}",
            banks=[b for b in self.banks if b.region == region],
        )


def _load_yaml() -> dict:
    with open(_UNIVERSES_FILE, "r") as f:
        return yaml.safe_load(f)


def list_universes() -> List[str]:
    """Return available universe names."""
    return list(_load_yaml().keys())


def load_universe(name: str) -> BankUniverse:
    """Load a named bank universe from the YAML config.

    Parameters
    ----------
    name : str
        One of the keys in bank_universes.yaml (e.g. 'eu_10', 'eu_50', 'global_100').

    Returns
    -------
    BankUniverse
    """
    data = _load_yaml()
    if name not in data:
        raise KeyError(
            f"Unknown universe '{name}'. Available: {list(data.keys())}"
        )
    banks = [Bank(**entry) for entry in data[name]]
    return BankUniverse(name=name, banks=banks)
