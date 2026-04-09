"""RandomAI — baseline floor for the AI zoo."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from ..cards import AuctionCard, GemCard
from ..engine import max_legal_bid
from .base import Player

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


class RandomAI(Player):
    """Picks bids and gem reveals uniformly at random over legal options."""

    def __init__(self, name: str, seed: int | None = None) -> None:
        super().__init__(name)
        self._rng = random.Random(seed)

    def choose_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> int:
        cap = max_legal_bid(my_state, auction)
        return self._rng.randint(0, cap)

    def choose_gem_to_reveal(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
    ) -> GemCard:
        return self._rng.choice(my_state.hand)
