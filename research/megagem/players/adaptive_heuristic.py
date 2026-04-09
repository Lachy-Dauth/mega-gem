"""AdaptiveHeuristicAI — HeuristicAI with a linear-model bid-sizing discount.

Replaces HeuristicAI's fixed ``0.75`` discount with
``clamp(bias + Σ wᵢ·featureᵢ, 0, 1)`` over five state features. Value
estimation and reveal logic are inherited unchanged from ``HeuristicAI``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..cards import AuctionCard, InvestCard, LoanCard, TreasureCard
from ..engine import max_legal_bid
from .helpers import (
    _DiscountFeatures,
    _compute_discount_features,
    _format_discount_features,
    _treasure_value,
)
from .heuristic import HeuristicAI

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


class AdaptiveHeuristicAI(HeuristicAI):
    """HeuristicAI variant whose bid-sizing discount is a linear function of
    game-state features instead of a hard-coded constant.

    Inherits choose_gem_to_reveal and the value-estimation helpers from
    HeuristicAI; only choose_bid is overridden.
    """

    # Linear model: discount = clamp(BIAS + Σ W_i * feature_i, 0, 1).
    # Hand-tuned by benchmark sweep against RandomAI. These live as class
    # attributes so a tuned subclass can override them cleanly.
    BIAS = 0.70
    W_PROGRESS = 0.25
    W_MY_CASH = 0.35
    W_AVG_CASH = -0.10
    W_TOP_CASH = -0.15
    W_VARIANCE = -0.05

    # Loan policy thresholds.
    LOAN_CASH_RATIO_MAX = 0.5
    LOAN_DISCOUNT_MIN = 0.5

    def discount_rate(self, features: _DiscountFeatures) -> float:
        raw = (
            self.BIAS
            + self.W_PROGRESS * features.progress
            + self.W_MY_CASH * features.my_cash_ratio
            + self.W_AVG_CASH * features.avg_cash_ratio
            + self.W_TOP_CASH * features.top_cash_ratio
            + self.W_VARIANCE * features.variance
        )
        return max(0.0, min(1.0, raw))

    def choose_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> int:
        cap = max_legal_bid(my_state, auction)
        if cap == 0:
            return 0

        features = _compute_discount_features(public_state, my_state)
        discount = self.discount_rate(features)

        # Reserve cash for upcoming auctions (inherited from HeuristicAI).
        reserve = self._reserve_for_future(public_state, my_state)
        spendable = max(0, my_state.coins - reserve)

        if isinstance(auction, TreasureCard):
            value = _treasure_value(auction, public_state, my_state)
            target = int(value * discount)
            bid = min(target, spendable, cap)
            return max(0, bid)

        if isinstance(auction, InvestCard):
            # Whole face value is profit; the discount scales how much of
            # the surplus-over-reserve we're willing to lock up. Always grab
            # at least 1 if legal — free coins are free coins.
            bid = min(int(spendable * discount), cap)
            if bid == 0 and cap > 0:
                bid = 1
            return bid

        if isinstance(auction, LoanCard):
            # Only borrow when (a) we're cash-poor relative to remaining gem
            # value AND (b) the model thinks aggressive bidding is warranted.
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
        features = _compute_discount_features(public_state, my_state)
        reserve = self._reserve_for_future(public_state, my_state)
        spendable = max(0, my_state.coins - reserve)
        lines = [
            _format_discount_features(features),
            f"reserve={reserve}  spendable={spendable}  "
            f"discount={self.discount_rate(features):.2f}",
        ]
        if isinstance(auction, TreasureCard):
            value = _treasure_value(auction, public_state, my_state)
            lines.append(f"treasure_value_estimate={value}")
        return lines
