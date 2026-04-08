"""HyperAdaptiveAI — AdaptiveHeuristicAI fed by the hypergeometric estimator.

Synthesises both upgrades — the better per-gem value estimate AND the
state-dependent discount — into a single player. Everything else
(discount weights, loan thresholds, reveal logic) is inherited from
``AdaptiveHeuristicAI``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..cards import AuctionCard, InvestCard, LoanCard, TreasureCard
from ..engine import max_legal_bid
from .adaptive_heuristic import AdaptiveHeuristicAI
from .helpers import (
    _format_discount_features,
    _hyper_avg_treasure_value,
    _hyper_compute_discount_features,
    _hyper_treasure_value,
    _remaining_supply,
)

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


class HyperAdaptiveAI(AdaptiveHeuristicAI):
    """AdaptiveHeuristicAI fed by the hypergeometric value estimator.

    Inherits the linear-model discount weights, loan thresholds, and reveal
    logic from ``AdaptiveHeuristicAI``; replaces the value-estimation
    plumbing (treasure value, average per-gem value, EV remaining) with the
    ``_hyper_*`` versions so the bidder reasons about the same numbers the
    HypergeometricAI's tests verified.

    The two effects this combines:

    1. ``_hyper_treasure_value`` — better per-gem coin estimate, especially
       on non-monotonic charts where ``E[chart(X)] ≠ chart(E[X])``.
    2. ``AdaptiveHeuristicAI.discount_rate`` — state-dependent bid sizing
       instead of a fixed 0.75 fraction.
    """

    def _reserve_for_future(self, public_state: "GameState") -> int:
        """Hyper-aware reserve. Same shape as HeuristicAI's, hyper avg."""
        gems_left = _remaining_supply(public_state)
        future_treasures = max(0, gems_left // 2)
        avg_value = _hyper_avg_treasure_value(
            public_state, public_state.player_states[0]
        )
        return int(future_treasures * avg_value * 0.2)

    def choose_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> int:
        cap = max_legal_bid(my_state, auction)
        if cap == 0:
            return 0

        features = _hyper_compute_discount_features(public_state, my_state)
        discount = self.discount_rate(features)

        reserve = self._reserve_for_future(public_state)
        spendable = max(0, my_state.coins - reserve)

        if isinstance(auction, TreasureCard):
            value = _hyper_treasure_value(auction, public_state, my_state)
            target = int(value * discount)
            bid = min(target, spendable, cap)
            return max(0, bid)

        if isinstance(auction, InvestCard):
            bid = min(int(spendable * discount), cap)
            if bid == 0 and cap > 0:
                bid = 1
            return bid

        if isinstance(auction, LoanCard):
            if features.my_cash_ratio >= self.LOAN_CASH_RATIO_MAX:
                return 0
            if discount < self.LOAN_DISCOUNT_MIN:
                return 0
            return min(int(auction.amount * discount), cap)

        return 0

    # --- Debug rationale --------------------------------------------------

    def explain_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> list[str]:
        features = _hyper_compute_discount_features(public_state, my_state)
        reserve = self._reserve_for_future(public_state)
        spendable = max(0, my_state.coins - reserve)
        lines = [
            _format_discount_features(features),
            f"reserve={reserve}  spendable={spendable}  "
            f"discount={self.discount_rate(features):.2f}",
        ]
        if isinstance(auction, TreasureCard):
            value = _hyper_treasure_value(auction, public_state, my_state)
            lines.append(f"treasure_value_estimate={value:.1f}")
        return lines
