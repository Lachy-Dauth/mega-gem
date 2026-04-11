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

from typing import TYPE_CHECKING

from ..cards import (
    AuctionCard,
    InvestCard,
    LoanCard,
    TreasureCard,
)
from ..engine import max_legal_bid
from .base_evo import (
    BaseEvoAI,
    _CAT_INVEST,
    _CAT_LOAN,
    _CAT_TREASURE,
    _DEFAULT_MEAN_DELTA,
    _DEFAULT_STD_DELTA,
    _MATCH_WEIGHT,
    _OTHER_WEIGHT,
    _category_of,
    _weighted_delta_stats,
)
from .evo2 import (
    _compute_evo2_features,
    _Evo2Features,
    _treasure_value_stats,
)

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


# The opponent-pricing category tags (``_CAT_*``), the matching-category
# weight constants (``_MATCH_WEIGHT`` / ``_OTHER_WEIGHT``), the
# ``(0.0, 1.0)`` default-delta fallbacks, and the ``_category_of`` /
# ``_weighted_delta_stats`` helpers now live in :mod:`.base_evo` so
# both :class:`Evo3AI` and :class:`Evo4AI` can share them without
# evo4 having to reach into evo3. The re-exports above preserve the
# historical import path — tests and external callers can still write
# ``from megagem.players.evo3 import _CAT_TREASURE``.
__all__ = [
    "Evo3AI",
    "_CAT_TREASURE",
    "_CAT_INVEST",
    "_CAT_LOAN",
    "_DEFAULT_MEAN_DELTA",
    "_DEFAULT_STD_DELTA",
    "_Evo3InvestModel",
    "_Evo3LoanModel",
    "_Evo3TreasureModel",
    "_MATCH_WEIGHT",
    "_OTHER_WEIGHT",
    "_category_of",
    "_weighted_delta_stats",
]


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
    # Post-round observation: grow the opponent-delta history via the
    # shared :meth:`BaseEvoAI._record_opp_delta` helper.
    # ------------------------------------------------------------------
    def observe_round(
        self,
        public_state: "GameState",
        my_idx: int,
        result: dict,
    ) -> None:
        self._record_opp_delta(my_idx, result)

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
            actual_bid, default_bid = self._bid_amount_auction_with_delta(
                self.invest_model,
                f,
                auction.amount,
                cap,
                _CAT_INVEST,
                apply_token_rule=True,
            )
            self._last_default_bid = default_bid
            return actual_bid

        if isinstance(auction, LoanCard):
            actual_bid, default_bid = self._bid_amount_auction_with_delta(
                self.loan_model,
                f,
                auction.amount,
                cap,
                _CAT_LOAN,
                apply_token_rule=False,
            )
            self._last_default_bid = default_bid
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
