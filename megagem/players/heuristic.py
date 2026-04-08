"""HeuristicAI — greedy estimator that beats RandomAI most of the time.

This is the GA's fitness opponent. Uses a fixed ``0.75`` discount applied
to a uniform-share display estimator. See :mod:`megagem.players.helpers`
for the value-estimation math.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING

from ..cards import AuctionCard, GemCard, InvestCard, LoanCard, TreasureCard
from ..engine import max_legal_bid
from ..value_charts import value_for
from .base import Player
from .helpers import (
    _expected_avg_treasure_value,
    _remaining_supply,
    _treasure_value,
)

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


class HeuristicAI(Player):
    """Greedy heuristic player. Beats RandomAI in head-to-head simulations."""

    DISCOUNT = 0.75  # bid this fraction of estimated value for treasures

    def __init__(self, name: str, seed: int | None = None) -> None:
        super().__init__(name)
        self._rng = random.Random(seed)

    # --- Bidding ----------------------------------------------------------

    def choose_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> int:
        cap = max_legal_bid(my_state, auction)
        if cap == 0:
            return 0

        if isinstance(auction, TreasureCard):
            value = _treasure_value(auction, public_state, my_state)
            target = int(value * self.DISCOUNT)
            # Floor: keep at least a tiny reserve so we're not broke.
            reserve = self._reserve_for_future(public_state)
            spendable = max(0, my_state.coins - reserve)
            bid = min(target, spendable, cap)
            return max(0, bid)

        if isinstance(auction, InvestCard):
            # Investments return their face value at end-of-game on top of the
            # locked bid — strictly positive cash flow. Bid surplus cash so we
            # don't starve future treasure bids.
            reserve = self._reserve_for_future(public_state)
            surplus = max(0, my_state.coins - reserve)
            # Always sneak in a token bid even if reserve is tight: free coins.
            bid = min(surplus, cap)
            if bid == 0 and cap > 0:
                bid = 1
            return bid

        if isinstance(auction, LoanCard):
            # Loans are net-negative cash flow (you pay back the full face).
            # Only useful as leverage when you have no coins AND there are
            # still meaningful gems to win.
            if my_state.coins >= 5:
                return 0
            if _remaining_supply(public_state) < 3:
                return 0
            return min(auction.amount, cap)

        return 0

    def _reserve_for_future(self, public_state: "GameState") -> int:
        """Coins to keep aside for upcoming treasure auctions."""
        gems_left = _remaining_supply(public_state)
        # Roughly: half the remaining gems are likely worth bidding on.
        future_treasures = max(0, gems_left // 2)
        avg_value = _expected_avg_treasure_value(public_state, public_state.player_states[0])
        # Use a fraction of avg so we don't over-reserve.
        return int(future_treasures * avg_value * 0.2)

    # --- Debug rationale --------------------------------------------------

    def explain_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> list[str]:
        reserve = self._reserve_for_future(public_state)
        spendable = max(0, my_state.coins - reserve)
        lines = [
            f"reserve={reserve}  spendable={spendable}  discount={self.DISCOUNT:.2f}",
        ]
        if isinstance(auction, TreasureCard):
            value = _treasure_value(auction, public_state, my_state)
            lines.append(f"treasure_value_estimate={value}")
        return lines

    # --- Reveal -----------------------------------------------------------

    def choose_gem_to_reveal(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
    ) -> GemCard:
        chart = public_state.value_chart
        display = public_state.value_display

        # Tally collection holdings per color across all players.
        my_holding = my_state.collection_gems
        opp_holding: Counter = Counter()
        for ps in public_state.player_states:
            if ps is my_state:
                continue
            opp_holding.update(ps.collection_gems)

        # Score each unique color present in our hand.
        best_score: float | None = None
        best_card: GemCard | None = None
        for card in my_state.hand:
            color = card.color
            current = display.get(color, 0)
            delta = value_for(chart, current + 1) - value_for(chart, current)
            relative = my_holding.get(color, 0) - opp_holding.get(color, 0)
            net_benefit = delta * relative
            # Tie-breaker: prefer revealing a color we hold least of.
            tiebreaker = -my_holding.get(color, 0)
            score = (net_benefit, tiebreaker)
            if best_score is None or score > best_score:
                best_score = score
                best_card = card

        return best_card if best_card is not None else my_state.hand[0]
