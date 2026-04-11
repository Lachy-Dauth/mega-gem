"""Evo4AI — Evo3 with bid-signal gem inference AND opponent-bid prediction.

Identical to :class:`Evo3AI` on every head except treasure. The treasure
head gains two new ideas, both tuneable by the GA:

1. **Per-color signal.** Evo4 watches how much opponents bid on
   *treasures* versus its own baseline bid, and attributes any excess to
   the colors of the gems that were on offer. A persistent per-color
   ``_color_signal`` dictionary accumulates that evidence ("opponents
   consistently bid above expectation on treasures containing Blue →
   they probably already hold Blue gems in hand → the final value
   display will skew toward Blue"). When Evo4 evaluates a pending
   treasure, it pushes the hypergeometric chart-index expectation for
   each color by ``color_bias_influence × color_signal[color]`` before
   the treasure-value chart lookup. Spec in the user's own words: "if
   two people bet 8 for treasure and one bets 5 you can assume the two
   have a gem of that colour".

2. **Opponent-bid prediction.** For every seat that isn't us, Evo4 runs
   a small internal Evo2-style treasure head from that seat's POV — its
   own coins become ``my_coins``, everyone else (including us) becomes
   the ``avg_opp_coins`` / ``top_opp_coins`` bucket — feeding in the
   same ``(ev, std)`` we computed for the treasure as a proxy for
   their estimate. Taking ``max`` and ``mean`` across the per-opponent
   predicted bids gives two new features, ``opp_max`` and ``opp_avg``,
   that the treasure head can weight via ``w_opp_max`` / ``w_opp_avg``.
   **The internal-predictor weights are themselves evolvable** (they
   live in the flat weights vector and the GA tunes them alongside the
   outer-head weights), so the GA gets to decide how to model
   opponents rather than inheriting a fixed Evo2 snapshot.

Weight layout (flat 35-element vector, the form the GA produces)::

    treasure             (11): bias, w_rounds, w_my, w_avg, w_top,
                               w_ev, w_std, w_mean_delta, w_std_delta,
                               w_opp_max, w_opp_avg
    invest                (8): bias, w_rounds, w_my, w_avg, w_top,
                               w_amount, w_mean_delta, w_std_delta
    loan                  (8): bias, w_rounds, w_my, w_avg, w_top,
                               w_amount, w_mean_delta, w_std_delta
    color_bias_influence  (1): scalar, scales the per-color index shift
    internal_evo2_treasure(7): bias, w_rounds, w_my, w_avg, w_top,
                               w_ev, w_std — Evo2-style head used per
                               opponent seat to predict their bid

Seed defaults are the Evo3 class defaults with zeros for
``w_opp_max`` / ``w_opp_avg`` / ``color_bias_influence``, and the
Evo2 class defaults for the internal predictor. A freshly constructed
Evo4AI with these defaults reproduces Evo3 behaviour exactly until
the GA lights up any of the new weights — ``opp_max`` / ``opp_avg``
are multiplied by zero, ``color_bias_influence`` is zero, so none of
the new features can leak into the bid.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import TYPE_CHECKING

from ..cards import (
    AuctionCard,
    Color,
    GemCard,
    InvestCard,
    LoanCard,
    TreasureCard,
)
from ..engine import max_legal_bid
from ..value_charts import value_for
from .base import Player
from .evo2 import (
    _compute_evo2_features,
    _Evo2Features,
    _expected_rounds_remaining,
    _mission_probability_delta,
    _TreasureModel as _Evo2TreasureModel,
)
from .evo3 import (
    _CAT_INVEST,
    _CAT_LOAN,
    _CAT_TREASURE,
    _DEFAULT_MEAN_DELTA,
    _DEFAULT_STD_DELTA,
    _category_of,
    _Evo3InvestModel,
    _Evo3LoanModel,
    _Evo3TreasureModel,
    _weighted_delta_stats,
)
from .helpers import (
    _hyper_hidden_distribution,
    _mission_completion_bonus,
    _mission_progress_bonus,
)

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


# Canonical color order, cached once so ``observe_round`` / the EV
# loop don't churn the enum metaclass on every call.
_COLORS: tuple[Color, ...] = tuple(Color)


def _empty_color_signal() -> dict[Color, float]:
    """Fresh per-color signal dict — every color starts at 0.0."""
    return {c: 0.0 for c in _COLORS}


# ---------------------------------------------------------------------------
# Bias-adjusted treasure value stats.
# ---------------------------------------------------------------------------


def _biased_per_color_value_stats(
    state: "GameState",
    my_state: "PlayerState",
    chart: str,
    index_shift: dict[Color, float],
) -> dict[Color, tuple[float, float]]:
    """Per-color ``(EV, var)`` with the chart index shifted by color.

    Drop-in replacement for :func:`megagem.players.evo2._per_color_value_stats`
    that takes an ``index_shift`` mapping. For each term in the
    hypergeometric distribution of ``final_display[c]``, the chart
    lookup is performed at ``count + index_shift[c]`` instead of
    ``count``. The shifted index is linearly interpolated between the
    two adjacent integer chart entries, so small shifts produce smooth
    gradients (important for the GA — a hard ``round()`` would quantize
    away everything below ±0.5).

    With an all-zero ``index_shift`` this function is numerically
    identical to the Evo2 helper: ``count`` is integer, ``frac = 0``,
    so ``v = value_for(chart, count)``. That guarantees Evo4 with a
    zero ``color_bias_influence`` bids exactly like Evo3 regardless of
    whether the color signal has accumulated.
    """
    distributions = _hyper_hidden_distribution(state, my_state)
    stats: dict[Color, tuple[float, float]] = {}
    for color, dist in distributions.items():
        shift = index_shift.get(color, 0.0)
        ev = 0.0
        ev2 = 0.0
        for count, p in dist.items():
            adjusted = count + shift
            if adjusted <= 0.0:
                v = float(value_for(chart, 0))
            elif adjusted >= 5.0:
                v = float(value_for(chart, 5))
            else:
                lo = int(adjusted)  # floor, adjusted ≥ 0
                frac = adjusted - lo
                v_lo = value_for(chart, lo)
                v_hi = value_for(chart, lo + 1)
                v = v_lo * (1.0 - frac) + v_hi * frac
            ev += p * v
            ev2 += p * v * v
        var = max(0.0, ev2 - ev * ev)
        stats[color] = (ev, var)
    return stats


def _treasure_value_stats_biased(
    auction: TreasureCard,
    state: "GameState",
    my_state: "PlayerState",
    index_shift: dict[Color, float],
) -> tuple[float, float]:
    """``(EV, std)`` of winning this treasure, with a per-color index shift.

    Structurally identical to :func:`megagem.players.evo2._treasure_value_stats`
    — it just routes through :func:`_biased_per_color_value_stats` for
    the gem-value terms. Mission bonuses (hard completion, soft
    progress, and probability delta) are applied unchanged so the
    bias can't accidentally warp mission accounting.
    """
    gems_for_sale = state.revealed_gems[: min(auction.gems, len(state.revealed_gems))]
    if not gems_for_sale:
        return 0.0, 0.0

    stats = _biased_per_color_value_stats(
        state, my_state, state.value_chart, index_shift
    )
    color_counts = Counter(g.color for g in gems_for_sale)

    ev = 0.0
    var = 0.0
    for color, n in color_counts.items():
        mean, v = stats[color]
        ev += n * mean
        var += (n * n) * v  # same color: perfect correlation

    extra = Counter(g.color for g in gems_for_sale)
    ev += _mission_completion_bonus(my_state, state.active_missions, extra)
    ev += _mission_progress_bonus(my_state, state.active_missions, extra)
    ev += _mission_probability_delta(auction, state, my_state)

    return ev, math.sqrt(var)


# ---------------------------------------------------------------------------
# Evo4 treasure head — Evo3 + 2 opponent-predicted-bid features.
# ---------------------------------------------------------------------------


class _Evo4TreasureModel(_Evo3TreasureModel):
    """Evo3 treasure head + ``w_opp_max`` + ``w_opp_avg`` = 11 weights.

    Structurally an extension of :class:`_Evo3TreasureModel`: inherits
    the 9 Evo3 slots, appends two for the opponent-bid-prediction
    aggregates. The ``bid`` method's signature grows by two trailing
    scalars (``opp_max``, ``opp_avg``) and composes the parent's bid
    with the new linear terms. With ``w_opp_max = w_opp_avg = 0`` this
    is bit-for-bit a 9-weight Evo3 head, which is how the defaults
    reproduce Evo3 behaviour.
    """

    __slots__ = ("w_opp_max", "w_opp_avg")

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
        w_opp_max: float,
        w_opp_avg: float,
    ) -> None:
        super().__init__(
            bias,
            w_rounds,
            w_my,
            w_avg,
            w_top,
            w_ev,
            w_std,
            w_mean_delta,
            w_std_delta,
        )
        self.w_opp_max = w_opp_max
        self.w_opp_avg = w_opp_avg

    def bid(
        self,
        f: _Evo2Features,
        ev: float,
        std: float,
        mean_delta: float,
        std_delta: float,
        opp_max: float = 0.0,
        opp_avg: float = 0.0,
    ) -> float:
        return (
            super().bid(f, ev, std, mean_delta, std_delta)
            + self.w_opp_max * opp_max
            + self.w_opp_avg * opp_avg
        )


# ---------------------------------------------------------------------------
# Opponent-bid prediction: run an internal Evo2 head from each opponent's POV.
# ---------------------------------------------------------------------------


def _predict_opponent_treasure_bids(
    auction: TreasureCard,
    state: "GameState",
    my_state: "PlayerState",
    ev: float,
    std: float,
    rounds_remaining: float,
    internal_model: _Evo2TreasureModel,
) -> tuple[float, float]:
    """``(max, avg)`` of predicted opponent bids for this treasure.

    For each opponent seat, builds an ``_Evo2Features`` *from their
    POV* — their own coins become ``my_coins``, the remaining seats
    (including us) form the ``avg_opp_coins`` / ``top_opp_coins``
    bucket — runs ``internal_model.bid(...)`` on it, clamps to their
    own legal cap, and returns the max and mean across the predicted
    bids. Returns ``(0.0, 0.0)`` when there are no opponents.

    **EV/std are reused from our own computation** as a proxy for the
    opponent's estimate. Strictly, each opponent sees a different
    hypergeometric hidden distribution (they know their own hand,
    which is hidden from us), but recomputing a full per-opponent
    distribution inside every bid is too expensive and the treasure
    EV is dominated by public signals (the value chart and revealed
    gems) anyway. The GA can learn whatever systematic correction is
    needed via the internal model's bias / ev / std weights.

    ``rounds_remaining`` is passed in rather than recomputed because
    it's a pure state function — the same number for every seat — and
    the outer ``choose_bid`` has already computed it once.
    """
    n = len(state.player_states)
    if n <= 1:
        return 0.0, 0.0

    coins = [ps.coins for ps in state.player_states]
    predicted: list[float] = []
    for i, opp_state in enumerate(state.player_states):
        if opp_state is my_state:
            continue
        # Features from seat i's POV: their coins vs everyone else's.
        others_coins = [coins[j] for j in range(n) if j != i]
        if not others_coins:
            continue
        f_opp = _Evo2Features(
            rounds_remaining=rounds_remaining,
            my_coins=float(coins[i]),
            avg_opp_coins=sum(others_coins) / len(others_coins),
            top_opp_coins=float(max(others_coins)),
        )
        raw = internal_model.bid(f_opp, ev, std)
        # Cap: treasures can't be paid with a loan, so the opponent's
        # legal cap is exactly their coin pile. Route through
        # ``max_legal_bid`` anyway so any future cap changes in the
        # engine propagate automatically.
        cap = max_legal_bid(opp_state, auction)
        predicted.append(max(0.0, min(float(raw), float(cap))))

    if not predicted:
        return 0.0, 0.0
    return max(predicted), sum(predicted) / len(predicted)


# ---------------------------------------------------------------------------
# Evo4AI itself.
# ---------------------------------------------------------------------------


class Evo4AI(Player):
    """Evo3 with bid-signal-driven color probability adjustment.

    Per-round lifecycle:

    1. :meth:`choose_bid` computes a per-color chart-index shift from
       the current color signal scaled by ``color_bias_influence``,
       feeds it into :func:`_treasure_value_stats_biased`, and runs
       the treasure head's linear formula on the biased ``(ev, std)``.
       It also caches a *baseline bid* — the bid Evo4 would produce
       with a zero shift AND the default ``(0.0, 1.0)`` opp-delta
       inputs — on ``self._last_default_bid``.
    2. After the engine resolves the round, :meth:`observe_round`
       reads the cached baseline, computes ``max_opp_bid − baseline``,
       appends it to the Evo3-style ``_opp_history`` list, and — if
       the auction was a treasure — distributes that delta across the
       gems that were on offer, adding a per-color share into
       ``_color_signal``.

    The color-signal update uses the *same* baseline cached for the
    opp-delta history, so the two features share a single consistent
    notion of "what I would have bid without any history". Reusing
    the baseline also means the color signal is exactly zero for the
    first treasure round of the game (no history → delta = 0).

    The signal is global (not per-opponent). The rationale: the user's
    example is about "somebody" bidding high, and tracking per-opponent
    state would multiply the bookkeeping without giving a huge payoff
    when most games end in ~15-20 rounds. If the GA discovers the
    per-opponent breakdown matters, that's a future-Evo5 problem.
    """

    NUM_WEIGHTS = 35
    # Layout: 11 treasure + 8 invest + 8 loan + 1 color_bias + 7 internal.

    # Seed defaults = Evo3 class defaults with zeros for the two new
    # treasure-head weights (``w_opp_max``, ``w_opp_avg``) and
    # ``color_bias_influence``, plus the Evo2 class defaults for the
    # internal opponent-predictor head. Freshly constructed Evo4AI
    # with these defaults is bit-for-bit identical to the default
    # Evo3AI, which in turn is identical to default Evo2 — the new
    # features are all multiplied by a zero weight.
    DEFAULT_TREASURE = _Evo4TreasureModel(
        bias=0.9671062444221764,
        w_rounds=-0.0906995616980441,
        w_my=0.07804979550128198,
        w_avg=0.05375147152736104,
        w_top=-0.04247465810129918,
        w_ev=0.32783828473034604,
        w_std=-0.011838494331700117,
        w_mean_delta=0.0,
        w_std_delta=0.0,
        w_opp_max=0.0,
        w_opp_avg=0.0,
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
    DEFAULT_COLOR_BIAS_INFLUENCE = 0.0
    # Internal opponent-bid predictor: start from Evo2's own class
    # defaults (the GA-tuned ones baked into Evo2AI). These are the
    # "most plausible opponent" until the GA decides otherwise.
    DEFAULT_INTERNAL_EVO2_TREASURE = _Evo2TreasureModel(
        bias=0.9671062444221764,
        w_rounds=-0.0906995616980441,
        w_my=0.07804979550128198,
        w_avg=0.05375147152736104,
        w_top=-0.04247465810129918,
        w_ev=0.32783828473034604,
        w_std=-0.011838494331700117,
    )

    def __init__(
        self,
        name: str,
        *,
        treasure: _Evo4TreasureModel | None = None,
        invest: _Evo3InvestModel | None = None,
        loan: _Evo3LoanModel | None = None,
        color_bias_influence: float | None = None,
        internal_evo2_treasure: _Evo2TreasureModel | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(name)
        self._rng = random.Random(seed)
        self.treasure_model = (
            treasure if treasure is not None else self.DEFAULT_TREASURE
        )
        self.invest_model = (
            invest if invest is not None else self.DEFAULT_INVEST
        )
        self.loan_model = loan if loan is not None else self.DEFAULT_LOAN
        self.color_bias_influence = (
            color_bias_influence
            if color_bias_influence is not None
            else self.DEFAULT_COLOR_BIAS_INFLUENCE
        )
        self.internal_evo2_treasure = (
            internal_evo2_treasure
            if internal_evo2_treasure is not None
            else self.DEFAULT_INTERNAL_EVO2_TREASURE
        )
        # Evo3-style opp-delta history (category, max_opp_bid − baseline).
        self._opp_history: list[tuple[str, float]] = []
        # New in Evo4: per-color running sum of the same delta attributed
        # across the colors of the gems on offer. Updated in observe_round
        # for treasure rounds only. Persists for the life of this instance;
        # a fresh game starts with a fresh signal because a new Evo4AI is
        # constructed per game by the factories.
        self._color_signal: dict[Color, float] = _empty_color_signal()
        # Scratch cache — same role as Evo3._last_default_bid. Cleared
        # unconditionally in observe_round so a stale value can't leak
        # across rounds.
        self._last_default_bid: int | None = None

    @classmethod
    def from_weights(
        cls,
        name: str,
        weights: list[float],
        *,
        seed: int | None = None,
    ) -> "Evo4AI":
        """Build from a flat 35-element weights list.

        Layout::

            [treasure(11), invest(8), loan(8),
             color_bias(1), internal_evo2_treasure(7)]
        """
        if len(weights) != cls.NUM_WEIGHTS:
            raise ValueError(
                f"Expected {cls.NUM_WEIGHTS} weights, got {len(weights)}"
            )
        return cls(
            name,
            treasure=_Evo4TreasureModel(*weights[0:11]),
            invest=_Evo3InvestModel(*weights[11:19]),
            loan=_Evo3LoanModel(*weights[19:27]),
            color_bias_influence=weights[27],
            internal_evo2_treasure=_Evo2TreasureModel(*weights[28:35]),
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Post-round observation: grow opp-delta history and color signal.
    # ------------------------------------------------------------------
    def observe_round(
        self,
        public_state: "GameState",
        my_idx: int,
        result: dict,
    ) -> None:
        # Pop the baseline cache unconditionally — a round we skip
        # (unknown auction kind, no opponents) must not leak its stale
        # baseline into the next observation.
        baseline = self._last_default_bid
        self._last_default_bid = None

        auction = result.get("auction")
        cat = _category_of(auction) if auction is not None else None
        if cat is None:
            return
        if baseline is None:
            # choose_bid wasn't called this round — skip the whole
            # observation, consistent with Evo3's no-feedback rule.
            return
        bids = result.get("bids") or []
        opp_bids = [b for i, b in enumerate(bids) if i != my_idx]
        if not opp_bids:
            return
        max_opp = max(opp_bids)
        delta = float(max_opp - baseline)
        self._opp_history.append((cat, delta))

        # Color-signal update: treasures only. The user's example is
        # explicitly about treasure bidding ("two people bet 8 for
        # treasure… you can assume the two have a gem of that colour"),
        # and invest/loan bidding carries no color information anyway.
        if cat != _CAT_TREASURE:
            return
        taken_gems = result.get("taken_gems") or []
        if not taken_gems:
            return
        n = len(taken_gems)
        share = delta / n
        for gem in taken_gems:
            self._color_signal[gem.color] += share

    # ------------------------------------------------------------------
    # Bid selection.
    # ------------------------------------------------------------------
    def _color_index_shift(self) -> dict[Color, float]:
        """Per-color chart-index shift = influence × color_signal[c].

        Returns an all-zero dict when the influence or the signal is
        trivially zero, so the EV loop below can cheaply skip the
        bias recomputation for the default-bid path.
        """
        infl = self.color_bias_influence
        if infl == 0.0:
            return _empty_color_signal()
        return {c: infl * self._color_signal[c] for c in _COLORS}

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
            shift = self._color_index_shift()
            ev_biased, std_biased = _treasure_value_stats_biased(
                auction, public_state, my_state, shift
            )
            # Default/baseline: zero shift AND default delta inputs.
            # If the shift is already all-zero, ev_biased already is
            # the default — skip the extra EV computation.
            if any(v != 0.0 for v in shift.values()):
                ev_default, std_default = _treasure_value_stats_biased(
                    auction, public_state, my_state, {}
                )
            else:
                ev_default, std_default = ev_biased, std_biased

            # Opponent-bid prediction: per-seat internal Evo2 head.
            # Uses the zero-shift (unbiased) EV/std so our private
            # color-signal beliefs don't leak into what we think
            # opponents think. Computed once and used for both the
            # actual bid and the baseline — it's a current-state
            # feature, not a history feature, so stripping it from
            # the baseline would misrepresent "what I'd bid without
            # delta history".
            opp_max, opp_avg = _predict_opponent_treasure_bids(
                auction,
                public_state,
                my_state,
                ev_default,
                std_default,
                f.rounds_remaining,
                self.internal_evo2_treasure,
            )

            mean_delta, std_delta = _weighted_delta_stats(
                self._opp_history, _CAT_TREASURE
            )

            actual_raw = self.treasure_model.bid(
                f,
                ev_biased,
                std_biased,
                mean_delta,
                std_delta,
                opp_max,
                opp_avg,
            )
            default_raw = self.treasure_model.bid(
                f,
                ev_default,
                std_default,
                _DEFAULT_MEAN_DELTA,
                _DEFAULT_STD_DELTA,
                opp_max,
                opp_avg,
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
                f,
                auction.amount,
                _DEFAULT_MEAN_DELTA,
                _DEFAULT_STD_DELTA,
            )
            actual_bid = max(0, min(int(actual_raw), cap))
            default_bid = max(0, min(int(default_raw), cap))
            # Free money — the token-bid-if-zero rule applies to both
            # the actual bid and the cached baseline, so the recorded
            # baseline matches what choose_bid actually returns.
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
                f,
                auction.amount,
                _DEFAULT_MEAN_DELTA,
                _DEFAULT_STD_DELTA,
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
        shift = self._color_index_shift()
        treasure_ev = 0.0
        treasure_std = 0.0
        opp_max = 0.0
        opp_avg = 0.0
        if isinstance(auction, TreasureCard):
            treasure_ev, treasure_std = _treasure_value_stats_biased(
                auction, public_state, my_state, shift
            )
            # Zero-shift EV/std used for opponent prediction, matching
            # choose_bid exactly.
            if any(v != 0.0 for v in shift.values()):
                ev_default, std_default = _treasure_value_stats_biased(
                    auction, public_state, my_state, {}
                )
            else:
                ev_default, std_default = treasure_ev, treasure_std
            opp_max, opp_avg = _predict_opponent_treasure_bids(
                auction,
                public_state,
                my_state,
                ev_default,
                std_default,
                f.rounds_remaining,
                self.internal_evo2_treasure,
            )

        t_md, t_sd = _weighted_delta_stats(self._opp_history, _CAT_TREASURE)
        i_md, i_sd = _weighted_delta_stats(self._opp_history, _CAT_INVEST)
        l_md, l_sd = _weighted_delta_stats(self._opp_history, _CAT_LOAN)

        tb = self.treasure_model.bid(
            f, treasure_ev, treasure_std, t_md, t_sd, opp_max, opp_avg
        )
        ib = self.invest_model.bid(f, invest_amount, i_md, i_sd)
        lb = self.loan_model.bid(f, loan_amount, l_md, l_sd)

        kind = (
            "treasure" if isinstance(auction, TreasureCard)
            else "invest" if isinstance(auction, InvestCard)
            else "loan" if isinstance(auction, LoanCard)
            else None
        )
        marker = {
            "treasure": ("\u25c0", "  ", "  "),
            "invest":   ("  ", "\u25c0", "  "),
            "loan":     ("  ", "  ", "\u25c0"),
        }.get(kind, ("  ", "  ", "  "))

        signal_str = "  ".join(
            f"{c.value[0]}={self._color_signal[c]:+.2f}" for c in _COLORS
        )
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
            f"color-signal  (influence={self.color_bias_influence:+.3f}):  "
            f"{signal_str}",
        ]
        if isinstance(auction, TreasureCard):
            lines.append(
                f"treasure:  ev={treasure_ev:.1f}  std={treasure_std:.2f}  "
                f"opp_max={opp_max:.1f}  opp_avg={opp_avg:.1f}"
            )
        return lines

    # ------------------------------------------------------------------
    # Reveal policy — direct copy of Evo3/Evo2/HeuristicAI. Inlined so
    # this file has no behavioural dependency on the older AIs.
    # ------------------------------------------------------------------
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
