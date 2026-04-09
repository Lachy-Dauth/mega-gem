"""Card definitions and deck factories for MegaGem."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Color(Enum):
    BLUE = "Blue"
    GREEN = "Green"
    PINK = "Pink"
    PURPLE = "Purple"
    YELLOW = "Yellow"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class GemCard:
    color: Color

    def __str__(self) -> str:
        return f"{self.color} gem"


@dataclass(frozen=True)
class AuctionCard:
    """Base for auction cards. Use a concrete subclass."""


@dataclass(frozen=True)
class TreasureCard(AuctionCard):
    gems: int  # 1 or 2

    def __str__(self) -> str:
        return f"Treasure ({self.gems} gem{'s' if self.gems != 1 else ''})"


@dataclass(frozen=True)
class LoanCard(AuctionCard):
    amount: int  # 10 or 20

    def __str__(self) -> str:
        return f"Loan ({self.amount} coins)"


@dataclass(frozen=True)
class InvestCard(AuctionCard):
    amount: int  # 5 or 10

    def __str__(self) -> str:
        return f"Invest ({self.amount} coins)"


def make_gem_deck() -> list[GemCard]:
    """30 gem cards: 6 of each of the 5 colors."""
    return [GemCard(color) for color in Color for _ in range(6)]


def make_auction_deck() -> list[AuctionCard]:
    """25 auction cards per the rules:

    - 17 Treasure: 12× 1-gem, 5× 2-gem
    - 4 Loan: 2× 10-coin, 2× 20-coin
    - 4 Invest: 2× 5-coin, 2× 10-coin
    """
    deck: list[AuctionCard] = []
    deck.extend(TreasureCard(1) for _ in range(12))
    deck.extend(TreasureCard(2) for _ in range(5))
    deck.extend(LoanCard(10) for _ in range(2))
    deck.extend(LoanCard(20) for _ in range(2))
    deck.extend(InvestCard(5) for _ in range(2))
    deck.extend(InvestCard(10) for _ in range(2))
    return deck
