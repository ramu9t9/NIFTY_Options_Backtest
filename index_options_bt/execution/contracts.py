"""
Contracts and selection interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional, Protocol, Sequence

from ..data.models import MarketSnapshot
from ..strategy.base import Intent
from ..config.schemas import SelectorConfig


CP = Literal["C", "P"]
Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class OptionContract:
    """Normalized option contract representation."""

    symbol: str
    expiry: date
    strike: int
    cp: CP
    multiplier: int = 1


@dataclass(frozen=True)
class SelectedLeg:
    """A selected leg produced by a selector."""

    contract: OptionContract
    side: Side
    qty: int
    tag: Optional[str] = None


class ContractSelector(Protocol):
    """Selector protocol: map (snapshot,intent) -> concrete option legs."""

    def select(
        self,
        snapshot: MarketSnapshot,
        intent: Intent,
        selector_cfg: SelectorConfig,
    ) -> Sequence[SelectedLeg]:
        ...


