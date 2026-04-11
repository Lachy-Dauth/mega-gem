"""HyperAdaptiveSplitAI — three independent linear bid models.

Motivation: a single state-feature → discount mapping is forced to give
the same answer for treasures (positive cash flow per unit value),
invests (free money), and loans (net-negative cash flow). Splitting the
model into three heads with their own biases and weights lets each
auction type have its own "be aggressive when …" rule. The three heads
share the same five game-state features so that the existing
:func:`_hyper_compute_discount_features` can be reused unchanged.

This is the class that :mod:`scripts.evolve_hyper_adaptive` tunes via a
small genetic algorithm. The defaults below are sane starting points;
the GA replaces them with evolved values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..cards import AuctionCard, InvestCard, LoanCard, TreasureCard
from ..engine import max_legal_bid
from .helpers import (
    _DiscountFeatures,
    _format_discount_features,
    _hyper_avg_treasure_value,
    _hyper_compute_discount_features,
    _hyper_treasure_value,
    _remaining_supply,
)
from .heuristic import HeuristicAI

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


class _BidModel:
    """Bias + 5 weights for one auction-type linear model.

    The discount is ``clamp(bias + Σ w_i * feature_i, 0, 1)`` over the same
    five features ``_DiscountFeatures`` carries. Stored as a plain class
    with __slots__ rather than a dataclass to keep it dependency-free and
    cheap to construct (the GA builds many of these per generation).
    """

    __slots__ = (
        "bias",
        "w_progress",
        "w_my_cash",
        "w_avg_cash",
        "w_top_cash",
        "w_variance",
    )

    def __init__(
        self,
        bias: float,
        w_progress: float,
        w_my_cash: float,
        w_avg_cash: float,
        w_top_cash: float,
        w_variance: float,
    ) -> None:
        self.bias = bias
        self.w_progress = w_progress
        self.w_my_cash = w_my_cash
        self.w_avg_cash = w_avg_cash
        self.w_top_cash = w_top_cash
        self.w_variance = w_variance

    def discount(self, features: _DiscountFeatures) -> float:
        raw = (
            self.bias
            + self.w_progress * features.progress
            + self.w_my_cash * features.my_cash_ratio
            + self.w_avg_cash * features.avg_cash_ratio
            + self.w_top_cash * features.top_cash_ratio
            + self.w_variance * features.variance
        )
        return max(0.0, min(1.0, raw))

    def _clone(self) -> "_BidModel":
        """Return an independent copy. Used to keep the class-level
        ``DEFAULT_*`` constants from being shared across AI instances —
        a per-instance tweak to ``treasure_model.bias`` would otherwise
        leak into every other AI built from defaults."""
        return _BidModel(
            self.bias,
            self.w_progress,
            self.w_my_cash,
            self.w_avg_cash,
            self.w_top_cash,
            self.w_variance,
        )


class HyperAdaptiveSplitAI(HeuristicAI):
    """Three independent linear discount models over hypergeometric value estimates.

    Inherits ``choose_gem_to_reveal`` from ``HeuristicAI``. Overrides
    ``choose_bid`` with three per-auction-type ``_BidModel`` heads and
    ``_reserve_for_future`` with a hypergeometric-aware reserve.

    Constructor accepts optional ``treasure``/``invest``/``loan`` model
    overrides; the GA driver builds instances via ``from_weights`` so the
    individuals it manipulates can stay as plain Python lists.
    """

    # Sensible starting points. Treasure copies HyperAdaptive's tuned
    # weights; invest is mildly aggressive (free money); loan is
    # conservative (net-negative cash flow). The GA replaces these.
    DEFAULT_TREASURE = _BidModel(0.70, 0.25, 0.35, -0.10, -0.15, -0.05)
    DEFAULT_INVEST = _BidModel(0.80, 0.10, 0.20, -0.05, -0.05, 0.00)
    DEFAULT_LOAN = _BidModel(0.10, 0.05, -0.40, 0.10, 0.10, -0.05)

    # Number of constants in a flat weights vector. 3 heads × (1 bias + 5
    # weights) = 18. The GA uses this to size individuals.
    NUM_WEIGHTS = 18

    def __init__(
        self,
        name: str,
        *,
        treasure: _BidModel | None = None,
        invest: _BidModel | None = None,
        loan: _BidModel | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(name, seed=seed)
        # Always copy the class-level defaults so two AIs that fall back
        # to the same default don't share a mutable model object.
        self.treasure_model = treasure if treasure is not None else self.DEFAULT_TREASURE._clone()
        self.invest_model = invest if invest is not None else self.DEFAULT_INVEST._clone()
        self.loan_model = loan if loan is not None else self.DEFAULT_LOAN._clone()

    @classmethod
    def from_weights(
        cls,
        name: str,
        weights: list[float],
        *,
        seed: int | None = None,
    ) -> "HyperAdaptiveSplitAI":
        """Build from a flat 18-element weights list.

        Order is ``[treasure_bias, *treasure_w5, invest_bias, *invest_w5,
        loan_bias, *loan_w5]`` so the GA driver can keep individuals as
        plain Python lists. Each 6-element block matches the ``_BidModel``
        constructor argument order.
        """
        if len(weights) != cls.NUM_WEIGHTS:
            raise ValueError(
                f"Expected {cls.NUM_WEIGHTS} weights, got {len(weights)}"
            )
        t = _BidModel(*weights[0:6])
        i = _BidModel(*weights[6:12])
        l = _BidModel(*weights[12:18])
        return cls(name, treasure=t, invest=i, loan=l, seed=seed)

    @classmethod
    def flatten_defaults(cls) -> list[float]:
        """Return the class-level ``DEFAULT_*`` constants as a flat 18-vector.

        The inverse of :meth:`from_weights`: feeding this result back
        through ``from_weights`` reconstructs the class-default AI. The
        unified GA tuner uses this as its fallback for individual #0
        when no saved weights file exists yet.
        """
        t = cls.DEFAULT_TREASURE
        i = cls.DEFAULT_INVEST
        l = cls.DEFAULT_LOAN
        return [
            t.bias, t.w_progress, t.w_my_cash, t.w_avg_cash, t.w_top_cash, t.w_variance,
            i.bias, i.w_progress, i.w_my_cash, i.w_avg_cash, i.w_top_cash, i.w_variance,
            l.bias, l.w_progress, l.w_my_cash, l.w_avg_cash, l.w_top_cash, l.w_variance,
        ]

    def _reserve_for_future(
        self, public_state: "GameState", my_state: "PlayerState"
    ) -> int:
        """Hyper-aware reserve. Same shape as HeuristicAI's, hyper avg."""
        gems_left = _remaining_supply(public_state)
        future_treasures = max(0, gems_left // 2)
        avg_value = _hyper_avg_treasure_value(public_state, my_state)
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
        reserve = self._reserve_for_future(public_state, my_state)
        spendable = max(0, my_state.coins - reserve)

        if isinstance(auction, TreasureCard):
            d = self.treasure_model.discount(features)
            value = _hyper_treasure_value(auction, public_state, my_state)
            return max(0, min(int(value * d), spendable, cap))

        if isinstance(auction, InvestCard):
            d = self.invest_model.discount(features)
            bid = min(int(spendable * d), cap)
            # Always grab a token bid if legal — invest payouts are
            # strictly positive cash flow, so 1 coin in is never wrong.
            if bid == 0 and cap > 0:
                bid = 1
            return bid

        if isinstance(auction, LoanCard):
            d = self.loan_model.discount(features)
            # No hard threshold gate — d ≈ 0 already produces bid 0 for
            # unfavourable states. The loan model owns the entire decision.
            return min(int(auction.amount * d), cap)

        return 0

    # --- Debug rationale --------------------------------------------------

    def explain_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> list[str]:
        features = _hyper_compute_discount_features(public_state, my_state)
        reserve = self._reserve_for_future(public_state, my_state)
        spendable = max(0, my_state.coins - reserve)
        td = self.treasure_model.discount(features)
        idd = self.invest_model.discount(features)
        ld = self.loan_model.discount(features)
        # Mark which head fires for the current auction so the active row
        # is unambiguous in the debug printout.
        kind = (
            "treasure" if isinstance(auction, TreasureCard)
            else "invest" if isinstance(auction, InvestCard)
            else "loan" if isinstance(auction, LoanCard)
            else None
        )
        marker = {"treasure": ("◀", "  ", "  "),
                  "invest":   ("  ", "◀", "  "),
                  "loan":     ("  ", "  ", "◀")}.get(kind, ("  ", "  ", "  "))
        lines = [
            _format_discount_features(features),
            f"reserve={reserve}  spendable={spendable}",
            f"heads:  treasure={td:.2f}{marker[0]}  "
            f"invest={idd:.2f}{marker[1]}  loan={ld:.2f}{marker[2]}",
        ]
        if isinstance(auction, TreasureCard):
            value = _hyper_treasure_value(auction, public_state, my_state)
            lines.append(f"treasure_value_estimate={value:.1f}")
        return lines
