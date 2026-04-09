"""HypergeometricAI — probabilistic value-display estimator.

HeuristicAI's point estimator collapses opponents' unknown gems into a
single mean count per color and then evaluates the value chart at that
point. That throws away the full distribution and is wrong for any
non-linear chart — most obviously chart E, which peaks at 3 gems and
then crashes. The fix is to compute the full hypergeometric distribution
of ``final_display[c]`` for each color and take ``E[chart_value(X)]``
instead of ``chart_value(E[X])``.

The bidding policy, reserve formula, and reveal logic are intentionally
*unchanged* from HeuristicAI — only the value estimator is upgraded so
this step can be evaluated in isolation. We deliberately do NOT subclass
HeuristicAI to avoid any coupling with the value-estimation plumbing.
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
    _hyper_avg_treasure_value,
    _hyper_treasure_value,
    _remaining_supply,
)

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


class HypergeometricAI(Player):
    """Heuristic player whose value estimator uses the full hypergeometric
    distribution over the hidden gem pool.

    Compared to ``HeuristicAI``, this AI computes
    ``E[chart_value(final_display[c])]`` rather than
    ``chart_value(E[final_display[c]])``. For monotonic charts that is a
    small accuracy bump; for non-monotonic charts (especially chart E,
    which peaks at 3 gems) it is a significant one.

    The bidding policy, reserve formula, and reveal logic are intentionally
    *unchanged* from HeuristicAI — only the value estimator is upgraded so
    that this step can be evaluated in isolation. We deliberately do NOT
    subclass HeuristicAI: the file is being edited in parallel and the
    standalone class avoids any coupling to that work.
    """

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
            value = _hyper_treasure_value(auction, public_state, my_state)
            target = int(value * self.DISCOUNT)
            reserve = self._reserve_for_future(public_state, my_state)
            spendable = max(0, my_state.coins - reserve)
            bid = min(target, spendable, cap)
            return max(0, bid)

        if isinstance(auction, InvestCard):
            # Investments return their face value at end-of-game on top of
            # the locked bid — strictly positive cash flow. Bid surplus cash
            # so we don't starve future treasure bids.
            reserve = self._reserve_for_future(public_state, my_state)
            surplus = max(0, my_state.coins - reserve)
            bid = min(surplus, cap)
            if bid == 0 and cap > 0:
                bid = 1
            return bid

        if isinstance(auction, LoanCard):
            # Loans are net-negative cash flow. Only useful when cash-poor
            # AND there's still meaningful gem supply left to win.
            if my_state.coins >= 5:
                return 0
            if _remaining_supply(public_state) < 3:
                return 0
            return min(auction.amount, cap)

        return 0

    def _reserve_for_future(
        self, public_state: "GameState", my_state: "PlayerState"
    ) -> int:
        """Coins to keep aside for upcoming treasure auctions."""
        gems_left = _remaining_supply(public_state)
        future_treasures = max(0, gems_left // 2)
        # Use *this* player's view of remaining-treasure value — using a
        # fixed seat would skew non-zero-seat bidders.
        avg_value = _hyper_avg_treasure_value(public_state, my_state)
        return int(future_treasures * avg_value * 0.2)

    # --- Debug rationale --------------------------------------------------

    def explain_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> list[str]:
        reserve = self._reserve_for_future(public_state, my_state)
        spendable = max(0, my_state.coins - reserve)
        lines = [
            f"reserve={reserve}  spendable={spendable}  discount={self.DISCOUNT:.2f}",
        ]
        if isinstance(auction, TreasureCard):
            value = _hyper_treasure_value(auction, public_state, my_state)
            lines.append(f"treasure_value_estimate={value:.1f}")
        return lines

    # --- Reveal -----------------------------------------------------------

    def choose_gem_to_reveal(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
    ) -> GemCard:
        chart = public_state.value_chart
        display = public_state.value_display

        my_holding = my_state.collection_gems
        opp_holding: Counter = Counter()
        for ps in public_state.player_states:
            if ps is my_state:
                continue
            opp_holding.update(ps.collection_gems)

        best_score: tuple[int, int] | None = None
        best_card: GemCard | None = None
        for card in my_state.hand:
            color = card.color
            current = display.get(color, 0)
            delta = value_for(chart, current + 1) - value_for(chart, current)
            relative = my_holding.get(color, 0) - opp_holding.get(color, 0)
            net_benefit = delta * relative
            tiebreaker = -my_holding.get(color, 0)
            score = (net_benefit, tiebreaker)
            if best_score is None or score > best_score:
                best_score = score
                best_card = card

        return best_card if best_card is not None else my_state.hand[0]
