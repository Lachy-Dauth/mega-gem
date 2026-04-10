"""Game state dataclasses for MegaGem.

Pure data — nothing here knows about the CLI or about Player implementations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .cards import AuctionCard, Color, GemCard, InvestCard, LoanCard
from .missions import MissionCard

if TYPE_CHECKING:
    from .players import Player


@dataclass
class PlayerState:
    name: str
    is_human: bool = False
    hand: list[GemCard] = field(default_factory=list)
    coins: int = 0
    collection_gems: Counter = field(default_factory=Counter)
    completed_missions: list[MissionCard] = field(default_factory=list)
    loans: list[LoanCard] = field(default_factory=list)
    investments: list[tuple[InvestCard, int]] = field(default_factory=list)


@dataclass
class GameState:
    player_states: list[PlayerState]
    players: list["Player"]
    value_chart: str
    value_display: Counter = field(default_factory=Counter)
    gem_deck: list[GemCard] = field(default_factory=list)
    revealed_gems: list[GemCard] = field(default_factory=list)
    auction_deck: list[AuctionCard] = field(default_factory=list)
    active_missions: list[MissionCard] = field(default_factory=list)
    last_winner_idx: int | None = None
    round_number: int = 0

    @property
    def num_players(self) -> int:
        return len(self.player_states)
