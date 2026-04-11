"""Evo3AI — Evo2 with opponent-pricing awareness.

Identical to :class:`Evo2AI` except that each head gains two new features:

* ``mean_delta`` — weighted mean of ``max(opponent bids) − baseline_bid``
  over the rounds this AI has already played, where ``baseline_bid`` is
  what Evo3 *would have bid* using the default delta values ``(0.0, 1.0)``
  (i.e. the Evo2-like bid, ignoring any history).
* ``std_delta``  — weighted standard deviation of the same quantity.

Using the default-deltas bid as the baseline (rather than Evo3's actual
bid) removes the feedback loop that would otherwise make the feature
definition depend on the AI's current response to its own history — the
baseline is a stable function of the state and the fixed model weights.

The "weighted" bit: when computing the features for a given head (e.g. the
treasure head on a treasure auction), observations from the matching
category are counted with weight 4 and observations from other categories
with weight 1. The rationale is that loan bidding tells you less about
how opponents price treasures than previous treasures do — 4× was the
weighting the user specified.

On the first call the history is empty, so the defaults ``(0.0, 1.0)``
are used. During :meth:`choose_bid` the AI caches the default-deltas
baseline bid on ``self._last_default_bid``; after each round resolves
the engine calls :meth:`Evo3AI.observe_round`, which pulls that cache
out and appends ``(category, max_opp_bid − baseline_bid)`` to the
history.

Weight layout (flat 25-element vector, the form the GA produces):

    treasure (9): bias, w_rounds, w_my, w_avg, w_top, w_ev, w_std,
                  w_mean_delta, w_std_delta
    invest   (8): bias, w_rounds, w_my, w_avg, w_top, w_amount,
                  w_mean_delta, w_std_delta
    loan     (8): bias, w_rounds, w_my, w_avg, w_top, w_amount,
                  w_mean_delta, w_std_delta

Seed defaults are the Evo2 defaults with zeros for the two new weights,
so a freshly constructed Evo3AI with the class defaults reproduces Evo2
behaviour exactly when the history is empty — and starts drifting once
enough rounds have been observed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..cards import (
    AuctionCard,
    InvestCard,
    LoanCard,
    TreasureCard,
)
from ..engine import max_legal_bid
from .base_evo import BaseEvoAI
from .evo2 import (
    _compute_evo2_features,
    _Evo2Features,
    _treasure_value_stats,
)

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


# Category tags for the opponent-history log. Kept as short strings so
# the history list is cheap to pickle / inspect from tests.
_CAT_TREASURE = "treasure"
_CAT_INVEST = "invest"
_CAT_LOAN = "loan"

# Matching-category weight multiplier. The user's spec: 4× for the
# current category, 1× for the others.
_MATCH_WEIGHT = 4.0
_OTHER_WEIGHT = 1.0

# Defaults when there's no history yet.
_DEFAULT_MEAN_DELTA = 0.0
_DEFAULT_STD_DELTA = 1.0


def _category_of(auction: AuctionCard) -> str | None:
    if isinstance(auction, TreasureCard):
        return _CAT_TREASURE
    if isinstance(auction, InvestCard):
        return _CAT_INVEST
    if isinstance(auction, LoanCard):
        return _CAT_LOAN
    return None


def _weighted_delta_stats(
    history: list[tuple[str, float]],
    current_category: str,
) -> tuple[float, float]:
    """Weighted mean and std of the opponent-delta history.

    Observations whose category matches ``current_category`` are counted
    with weight ``_MATCH_WEIGHT``; all others with ``_OTHER_WEIGHT``.
    Returns ``(_DEFAULT_MEAN_DELTA, _DEFAULT_STD_DELTA)`` when the
    history is empty — the spec's "first go" defaults.
    """
    if not history:
        return _DEFAULT_MEAN_DELTA, _DEFAULT_STD_DELTA

    total_w = 0.0
    total_x = 0.0
    total_x2 = 0.0
    for cat, delta in history:
        w = _MATCH_WEIGHT if cat == current_category else _OTHER_WEIGHT
        total_w += w
        total_x += w * delta
        total_x2 += w * delta * delta

    if total_w <= 0.0:
        return _DEFAULT_MEAN_DELTA, _DEFAULT_STD_DELTA

    mean = total_x / total_w
    var = max(0.0, total_x2 / total_w - mean * mean)
    return mean, math.sqrt(var)


# ---------------------------------------------------------------------------
# Linear bid models. Sibling classes to the Evo2 ones; each gains a
# ``w_mean_delta`` / ``w_std_delta`` weight pair and a 2-extra-arg ``bid``.
# ---------------------------------------------------------------------------


class _Evo3TreasureModel:
    """1 bias + 4 shared + 2 EV/std + 2 opponent-delta = 9 weights."""

    __slots__ = (
        "bias",
        "w_rounds",
        "w_my",
        "w_avg",
        "w_top",
        "w_ev",
        "w_std",
        "w_mean_delta",
        "w_std_delta",
    )

    def __init__(
        self,
        bias: float,
        w_rounds: float,
        w_my: float,
        w_avg: float,
        w_top: float,
        w_ev: float,
        w_std: float,
        w_mean_delta: float,
        w_std_delta: float,
    ) -> None:
        self.bias = bias
        self.w_rounds = w_rounds
        self.w_my = w_my
        self.w_avg = w_avg
        self.w_top = w_top
        self.w_ev = w_ev
        self.w_std = w_std
        self.w_mean_delta = w_mean_delta
        self.w_std_delta = w_std_delta

    def bid(
        self,
        f: _Evo2Features,
        ev: float,
        std: float,
        mean_delta: float,
        std_delta: float,
    ) -> float:
        return (
            self.bias
            + self.w_rounds * f.rounds_remaining
            + self.w_my * f.my_coins
            + self.w_avg * f.avg_opp_coins
            + self.w_top * f.top_opp_coins
            + self.w_ev * ev
            + self.w_std * std
            + self.w_mean_delta * mean_delta
            + self.w_std_delta * std_delta
        )


class _Evo3InvestModel:
    """1 bias + 4 shared + 1 amount + 2 opponent-delta = 8 weights."""

    __slots__ = (
        "bias",
        "w_rounds",
        "w_my",
        "w_avg",
        "w_top",
        "w_amount",
        "w_mean_delta",
        "w_std_delta",
    )

    def __init__(
        self,
        bias: float,
        w_rounds: float,
        w_my: float,
        w_avg: float,
        w_top: float,
        w_amount: float,
        w_mean_delta: float,
        w_std_delta: float,
    ) -> None:
        self.bias = bias
        self.w_rounds = w_rounds
        self.w_my = w_my
        self.w_avg = w_avg
        self.w_top = w_top
        self.w_amount = w_amount
        self.w_mean_delta = w_mean_delta
        self.w_std_delta = w_std_delta

    def bid(
        self,
        f: _Evo2Features,
        amount: int,
        mean_delta: float,
        std_delta: float,
    ) -> float:
        return (
            self.bias
            + self.w_rounds * f.rounds_remaining
            + self.w_my * f.my_coins
            + self.w_avg * f.avg_opp_coins
            + self.w_top * f.top_opp_coins
            + self.w_amount * amount
            + self.w_mean_delta * mean_delta
            + self.w_std_delta * std_delta
        )


class _Evo3LoanModel:
    """1 bias + 4 shared + 1 amount + 2 opponent-delta = 8 weights.

    Structurally identical to :class:`_Evo3InvestModel` but kept as its
    own class so flat-vector slicing in the GA is type-clear.
    """

    __slots__ = (
        "bias",
        "w_rounds",
        "w_my",
        "w_avg",
        "w_top",
        "w_amount",
        "w_mean_delta",
        "w_std_delta",
    )

    def __init__(
        self,
        bias: float,
        w_rounds: float,
        w_my: float,
        w_avg: float,
        w_top: float,
        w_amount: float,
        w_mean_delta: float,
        w_std_delta: float,
    ) -> None:
        self.bias = bias
        self.w_rounds = w_rounds
        self.w_my = w_my
        self.w_avg = w_avg
        self.w_top = w_top
        self.w_amount = w_amount
        self.w_mean_delta = w_mean_delta
        self.w_std_delta = w_std_delta

    def bid(
        self,
        f: _Evo2Features,
        amount: int,
        mean_delta: float,
        std_delta: float,
    ) -> float:
        return (
            self.bias
            + self.w_rounds * f.rounds_remaining
            + self.w_my * f.my_coins
            + self.w_avg * f.avg_opp_coins
            + self.w_top * f.top_opp_coins
            + self.w_amount * amount
            + self.w_mean_delta * mean_delta
            + self.w_std_delta * std_delta
        )


# ---------------------------------------------------------------------------
# Evo3AI itself.
# ---------------------------------------------------------------------------


class Evo3AI(BaseEvoAI):
    """Evo2 with opponent-pricing awareness.

    Per-round lifecycle:

    1. ``choose_bid`` reads the current opponent-delta history, computes
       weighted ``(mean_delta, std_delta)`` for the auction's category,
       feeds them into the active head alongside the Evo2 features, and
       returns ``int(bid)`` clamped to ``[0, cap]``. It also computes a
       *baseline bid* by running the same head with the default delta
       values ``(0.0, 1.0)``, clamped identically, and stashes that on
       ``self._last_default_bid``.
    2. After all players have bid and the engine has resolved the round,
       the engine calls :meth:`observe_round`. Evo3 pulls the max
       opponent bid from ``result["bids"]`` and appends
       ``(category, max_opp − self._last_default_bid)`` to
       ``_opp_history``. The baseline is the default-deltas bid, not
       the actual bid Evo3 submitted, so the delta measurement does
       not depend on Evo3's learned response to its own history.

    The history persists for the lifetime of the player instance, so
    across-game persistence is *not* provided — each game starts fresh.
    """

    NUM_WEIGHTS = 25  # 9 (treasure) + 8 (invest) + 8 (loan)

    # Seeded from the Evo2 defaults with zeros for the two new weights
    # on each head. A fresh Evo3AI with these defaults behaves exactly
    # like the default Evo2AI when the opponent history is empty.
    DEFAULT_TREASURE = _Evo3TreasureModel(
        bias=0.9671062444221764,
        w_rounds=-0.0906995616980441,
        w_my=0.07804979550128198,
        w_avg=0.05375147152736104,
        w_top=-0.04247465810129918,
        w_ev=0.32783828473034604,
        w_std=-0.011838494331700117,
        w_mean_delta=0.0,
        w_std_delta=0.0,
    )
    DEFAULT_INVEST = _Evo3InvestModel(
        bias=1.908464547879478,
        w_rounds=0.4300303741599258,
        w_my=-0.1201852409204779,
        w_avg=-0.28421403664160627,
        w_top=0.3149361220138405,
        w_amount=0.07219353469220569,
        w_mean_delta=0.0,
        w_std_delta=0.0,
    )
    DEFAULT_LOAN = _Evo3LoanModel(
        bias=-0.4139242208454687,
        w_rounds=-0.31190499765072527,
        w_my=0.13966251262722051,
        w_avg=0.12135141558388368,
        w_top=-0.0669196243751372,
        w_amount=0.36349000133503273,
        w_mean_delta=0.0,
        w_std_delta=0.0,
    )

    def __init__(
        self,
        name: str,
        *,
        treasure: _Evo3TreasureModel | None = None,
        invest: _Evo3InvestModel | None = None,
        loan: _Evo3LoanModel | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(name, seed=seed)
        self.treasure_model = (
            treasure if treasure is not None else self.DEFAULT_TREASURE
        )
        self.invest_model = (
            invest if invest is not None else self.DEFAULT_INVEST
        )
        self.loan_model = loan if loan is not None else self.DEFAULT_LOAN
        # ``self._opp_history`` (log of ``(category, max_opp_bid
        # − baseline_bid)`` per round) and ``self._last_default_bid``
        # (scratch cache populated at the end of choose_bid with the
        # default-deltas baseline bid) are initialized by
        # :class:`BaseEvoAI`. ``observe_round`` below reads and clears
        # the cache so a stale value can't leak across rounds.

    @classmethod
    def from_weights(
        cls,
        name: str,
        weights: list[float],
        *,
        seed: int | None = None,
    ) -> "Evo3AI":
        """Build from a flat 25-element weights list.

        Layout: ``[treasure(9), invest(8), loan(8)]``.
        """
        if len(weights) != cls.NUM_WEIGHTS:
            raise ValueError(
                f"Expected {cls.NUM_WEIGHTS} weights, got {len(weights)}"
            )
        return cls(
            name,
            treasure=_Evo3TreasureModel(*weights[0:9]),
            invest=_Evo3InvestModel(*weights[9:17]),
            loan=_Evo3LoanModel(*weights[17:25]),
            seed=seed,
        )

    @classmethod
    def flatten_defaults(cls) -> list[float]:
        """Return the class-level ``DEFAULT_*`` constants as a flat 25-vector.

        The inverse of :meth:`from_weights`: feeding this result back
        through ``from_weights`` reconstructs the class-default AI. The
        unified GA tuner uses this as its fallback for individual #0
        when no saved weights file exists yet.
        """
        t = cls.DEFAULT_TREASURE
        i = cls.DEFAULT_INVEST
        l = cls.DEFAULT_LOAN
        return [
            t.bias, t.w_rounds, t.w_my, t.w_avg, t.w_top,
            t.w_ev, t.w_std, t.w_mean_delta, t.w_std_delta,
            i.bias, i.w_rounds, i.w_my, i.w_avg, i.w_top, i.w_amount,
            i.w_mean_delta, i.w_std_delta,
            l.bias, l.w_rounds, l.w_my, l.w_avg, l.w_top, l.w_amount,
            l.w_mean_delta, l.w_std_delta,
        ]

    # ------------------------------------------------------------------
    # Post-round observation: grow the opponent-delta history.
    # ------------------------------------------------------------------
    def observe_round(
        self,
        public_state: "GameState",
        my_idx: int,
        result: dict,
    ) -> None:
        # Pop the baseline cache unconditionally so a round we decide
        # to skip (unknown auction kind, no opponents) can't leak its
        # stale baseline into the next observation.
        baseline = self._last_default_bid
        self._last_default_bid = None

        auction = result.get("auction")
        cat = _category_of(auction) if auction is not None else None
        if cat is None:
            return
        if baseline is None:
            # choose_bid wasn't called (or didn't cache a baseline) this
            # round — skip the observation rather than fall back to the
            # actual bid, so the delta definition stays consistent.
            return
        bids = result.get("bids") or []
        opp_bids = [b for i, b in enumerate(bids) if i != my_idx]
        if not opp_bids:
            return
        max_opp = max(opp_bids)
        self._opp_history.append((cat, float(max_opp - baseline)))

    # ------------------------------------------------------------------
    # Bid selection.
    # ------------------------------------------------------------------
    def choose_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> int:
        cap = max_legal_bid(my_state, auction)
        if cap == 0:
            self._last_default_bid = 0
            return 0

        f = _compute_evo2_features(public_state, my_state)

        if isinstance(auction, TreasureCard):
            ev, std = _treasure_value_stats(auction, public_state, my_state)
            mean_delta, std_delta = _weighted_delta_stats(
                self._opp_history, _CAT_TREASURE
            )
            actual_raw = self.treasure_model.bid(
                f, ev, std, mean_delta, std_delta
            )
            # Baseline: same features, same weights, but the
            # opponent-delta inputs pinned to the ``(0.0, 1.0)`` defaults.
            # This is what the AI would bid if it had no history yet.
            default_raw = self.treasure_model.bid(
                f, ev, std, _DEFAULT_MEAN_DELTA, _DEFAULT_STD_DELTA
            )
            actual_bid = max(0, min(int(actual_raw), cap))
            self._last_default_bid = max(0, min(int(default_raw), cap))
            return actual_bid

        if isinstance(auction, InvestCard):
            mean_delta, std_delta = _weighted_delta_stats(
                self._opp_history, _CAT_INVEST
            )
            actual_raw = self.invest_model.bid(
                f, auction.amount, mean_delta, std_delta
            )
            default_raw = self.invest_model.bid(
                f, auction.amount, _DEFAULT_MEAN_DELTA, _DEFAULT_STD_DELTA
            )
            actual_bid = max(0, min(int(actual_raw), cap))
            default_bid = max(0, min(int(default_raw), cap))
            # Free money — the token-bid-if-zero rule applies to both
            # the actual bid (to avoid skipping free invests) and the
            # baseline (so the recorded baseline matches what choose_bid
            # would actually return for an empty history).
            if actual_bid == 0 and cap > 0:
                actual_bid = 1
            if default_bid == 0 and cap > 0:
                default_bid = 1
            self._last_default_bid = default_bid
            return actual_bid

        if isinstance(auction, LoanCard):
            mean_delta, std_delta = _weighted_delta_stats(
                self._opp_history, _CAT_LOAN
            )
            actual_raw = self.loan_model.bid(
                f, auction.amount, mean_delta, std_delta
            )
            default_raw = self.loan_model.bid(
                f, auction.amount, _DEFAULT_MEAN_DELTA, _DEFAULT_STD_DELTA
            )
            actual_bid = max(0, min(int(actual_raw), cap))
            self._last_default_bid = max(0, min(int(default_raw), cap))
            return actual_bid

        self._last_default_bid = 0
        return 0

    # ------------------------------------------------------------------
    # Debug rationale.
    # ------------------------------------------------------------------
    def explain_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> list[str]:
        f = _compute_evo2_features(public_state, my_state)
        invest_amount = (
            auction.amount if isinstance(auction, InvestCard) else 5
        )
        loan_amount = (
            auction.amount if isinstance(auction, LoanCard) else 10
        )
        treasure_ev = 0.0
        treasure_std = 0.0
        if isinstance(auction, TreasureCard):
            treasure_ev, treasure_std = _treasure_value_stats(
                auction, public_state, my_state
            )

        t_md, t_sd = _weighted_delta_stats(self._opp_history, _CAT_TREASURE)
        i_md, i_sd = _weighted_delta_stats(self._opp_history, _CAT_INVEST)
        l_md, l_sd = _weighted_delta_stats(self._opp_history, _CAT_LOAN)

        tb = self.treasure_model.bid(f, treasure_ev, treasure_std, t_md, t_sd)
        ib = self.invest_model.bid(f, invest_amount, i_md, i_sd)
        lb = self.loan_model.bid(f, loan_amount, l_md, l_sd)

        kind = (
            "treasure" if isinstance(auction, TreasureCard)
            else "invest" if isinstance(auction, InvestCard)
            else "loan" if isinstance(auction, LoanCard)
            else None
        )
        marker = {
            "treasure": ("◀", "  ", "  "),
            "invest":   ("  ", "◀", "  "),
            "loan":     ("  ", "  ", "◀"),
        }.get(kind, ("  ", "  ", "  "))

        lines = [
            f"features:  rounds_left={f.rounds_remaining:.2f}  "
            f"my_coins={f.my_coins:.0f}  avg_opp={f.avg_opp_coins:.1f}  "
            f"top_opp={f.top_opp_coins:.0f}",
            f"heads (raw bid):  treasure={tb:+.1f}{marker[0]}  "
            f"invest={ib:+.1f}{marker[1]}  loan={lb:+.1f}{marker[2]}",
            f"opp-delta (history={len(self._opp_history)}):  "
            f"treasure=({t_md:+.1f},{t_sd:.1f})  "
            f"invest=({i_md:+.1f},{i_sd:.1f})  "
            f"loan=({l_md:+.1f},{l_sd:.1f})",
        ]
        if isinstance(auction, TreasureCard):
            lines.append(
                f"treasure:  ev={treasure_ev:.1f}  std={treasure_std:.2f}"
            )
        return lines
