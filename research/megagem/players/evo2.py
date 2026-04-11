"""Evo2AI — clean-slate evolved player.

A sibling of ``HyperAdaptiveSplitAI`` that throws out the pre-GA scaffolding
the user found unprincipled and replaces it with features the GA can
actually use:

* No ``_reserve_for_future`` / ``spendable`` floor — bids cap at the legal
  cap directly.
* ``progress`` (auctions consumed / 25) is replaced with an *exact*
  ``E[rounds remaining]`` computed by closed-form multivariate
  hypergeometric over the known auction-deck composition and remaining
  gem supply.
* ``my/avg/top_cash_ratio`` (coins / EV-remaining) become raw integer
  ``my_coins``, ``avg_opp_coins``, ``top_opp_coins``. The GA discovers
  the right scale.
* ``variance`` (a hand-rolled chart-swing × hidden-fraction proxy) is
  replaced per head:
    - Treasure: two new features, ``ev`` and ``std`` of the random
      variable "coin value of winning this treasure", both derived from
      the existing hypergeometric distribution over hidden gems.
    - Invest: ``auction.amount`` (5 or 10).
    - Loan:    ``auction.amount`` (10 or 20).
* The three heads output the **bid in coins directly**, not a discount
  fraction multiplied by EV / amount. The linear model is now
  ``bid = bias + Σ wᵢ · featureᵢ``, clamped to ``[0, cap]`` once at
  ``choose_bid``. This frees the GA from the implicit "scale by EV"
  coupling that the discount formulation baked in — the right scaling
  for each feature is itself a weight to learn.

The treasure EV also gains a *mission probability delta* — for every
active mission, the difference between "P(I win this mission | I take
the gems on offer)" and "P(I win this mission | the highest-coin
opponent takes them)", scaled by the mission's coin value. The two
existing deterministic mission bonuses (hard completion + soft progress)
are kept and the delta is added on top.
"""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

from ..cards import (
    AuctionCard,
    Color,
    InvestCard,
    LoanCard,
    TreasureCard,
)
from ..engine import max_legal_bid
from ..missions import MissionCard
from ..value_charts import value_for
from .base_evo import BaseEvoAI
from .helpers import (
    _GEMS_PER_COLOR,
    _hyper_hidden_distribution,
    _mission_completion_bonus,
    _mission_progress_bonus,
)

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


# Sentinel for "no multiset of size ≤ max_k satisfies this mission".
_IMPOSSIBLE_DISTANCE = 99

# Cached tuple of all colors — list(Color) is surprisingly hot in the
# distance probing loop because it triggers enum metaclass machinery on
# every iteration. Caching once at import time is free.
_COLORS: tuple[Color, ...] = tuple(Color)
# Hoist individual color references for the hand-unrolled holdings key
# below. dict.get is much faster than a generator over an enum tuple.
_COL_BLUE = Color.BLUE
_COL_GREEN = Color.GREEN
_COL_PINK = Color.PINK
_COL_PURPLE = Color.PURPLE
_COL_YELLOW = Color.YELLOW


def _holdings_key(holdings) -> tuple[int, int, int, int, int]:
    """Hashable representation of a holdings Counter / dict.

    Returns a fixed-length 5-tuple in ``_COLORS`` order, so two holdings
    with the same per-color counts produce the same key regardless of
    insertion order. Hand-unrolled instead of using a generator over
    ``_COLORS`` because this is on the cache-key hot path; the
    generator-based form was a measurable fraction of total runtime.
    """
    g = holdings.get
    return (
        g(_COL_BLUE, 0),
        g(_COL_GREEN, 0),
        g(_COL_PINK, 0),
        g(_COL_PURPLE, 0),
        g(_COL_YELLOW, 0),
    )


# Module-level memo for _min_extra_gems_to_satisfy. The key
# ``(holdings_tuple, mission.name)`` is stable: mission names are unique
# across the deck (see megagem/missions.py make_mission_deck), so two
# missions with the same name necessarily have the same predicate.
#
# Why a manual dict instead of @lru_cache: MissionCard isn't hashable
# (it has a Callable field) and we want to key on its name string, not
# the object. Hard cap at _DISTANCE_CACHE_LIMIT entries; on overflow
# we drop the whole table — distance computations are pure so a stale
# clear is harmless, and the LRU bookkeeping isn't worth the cost at
# the call rates we see (~700 distance calls per game).
_DISTANCE_CACHE: dict[tuple[tuple[int, ...], str], int] = {}
_DISTANCE_CACHE_LIMIT = 16384


# ---------------------------------------------------------------------------
# Treasure value distribution: mean and variance per color, then aggregated
# over the gems on offer.
# ---------------------------------------------------------------------------


def _per_color_value_stats(
    state: "GameState",
    my_state: "PlayerState",
    chart: str,
) -> dict[Color, tuple[float, float]]:
    """For each color, ``(E[chart_value(final_display[c])], Var[...])``.

    Walks the hypergeometric distribution from
    :func:`megagem.players._hyper_hidden_distribution` once per color and
    accumulates the first and second moments. Variance is clamped at 0
    for floating-point safety.
    """
    distributions = _hyper_hidden_distribution(state, my_state)
    stats: dict[Color, tuple[float, float]] = {}
    for color, dist in distributions.items():
        ev = 0.0
        ev2 = 0.0
        for count, p in dist.items():
            v = value_for(chart, min(count, 5))
            ev += p * v
            ev2 += p * v * v
        var = max(0.0, ev2 - ev * ev)
        stats[color] = (ev, var)
    return stats


def _treasure_value_stats(
    auction: TreasureCard,
    state: "GameState",
    my_state: "PlayerState",
) -> tuple[float, float]:
    """``(EV, std)`` of the coin value of winning this treasure auction.

    EV = Σ over gems on offer of ``mean[color]``
       + hard mission completion bonus (kept from HeuristicAI helpers)
       + soft mission progress bonus  (kept from HeuristicAI helpers)
       + mission probability delta    (NEW — see ``_mission_probability_delta``)

    Variance is computed only over the gem-value uncertainty: same-color
    contributions are perfectly correlated (exact), across-color
    contributions are treated as independent (approximation — they share
    the same hypergeometric urn). The mission terms are deterministic
    given the public state and feed only the EV.
    """
    gems_for_sale = state.revealed_gems[: min(auction.gems, len(state.revealed_gems))]
    if not gems_for_sale:
        return 0.0, 0.0

    stats = _per_color_value_stats(state, my_state, state.value_chart)
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
# Exact expected rounds remaining.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _expected_rounds_remaining_impl(
    A: int, T1: int, T2: int, NT: int, G: int
) -> float:
    """Closed-form ``E[rounds]`` from the deck composition tuple.

    Pulled out from :func:`_expected_rounds_remaining` so the result can
    be cached on the integer signature alone — within a round, all four
    players hit this with the same arguments, and across the GA the
    same compositions recur constantly. ``lru_cache`` is bounded so the
    cache can't blow up across long runs.

    Two micro-optimizations vs. the obvious form:

    * **Precomputed binomial tables.** ``math.comb`` was called ~1M
      times per profile run; precomputing ``C(T1, *)``, ``C(T2, *)``,
      ``C(NT, *)`` and ``C(A, *)`` once per call collapses each inner
      iteration to three list indexings.
    * **Tightened ``j2`` bound.** The constraint ``j₁ + 2·j₂ < G`` means
      ``j₂ ≤ (G − j₁ − 1) // 2``, so we skip the (j₁, j₂) tuples that
      we'd just discard inside the loop body.
    """
    if A == 0 or G == 0:
        return 0.0
    # No treasures left → game can only end when the auction deck runs out.
    if T1 == 0 and T2 == 0:
        return float(A)

    comb_t1 = [math.comb(T1, k) for k in range(T1 + 1)]
    comb_t2 = [math.comb(T2, k) for k in range(T2 + 1)]
    comb_nt = [math.comb(NT, k) for k in range(NT + 1)]
    comb_a = [math.comb(A, k) for k in range(A + 1)]

    total = 1.0  # the j=0 term contributes P(0 gems < G) = 1
    for j in range(1, A):  # j = k - 1, k = 2..A (j=0 already added)
        denom = comb_a[j]
        p_under = 0.0
        j1_max = min(T1, j)
        for j1 in range(j1_max + 1):
            # Tightened upper bound: j2 ≤ (G − j1 − 1) // 2 from the
            # gems_consumed < G constraint, plus the original bounds.
            j2_cap = min(T2, j - j1, (G - j1 - 1) // 2)
            if j2_cap < 0:
                continue
            ct1 = comb_t1[j1]
            for j2 in range(j2_cap + 1):
                jnt = j - j1 - j2
                if jnt < 0 or jnt > NT:
                    continue
                p_under += ct1 * comb_t2[j2] * comb_nt[jnt] / denom
        total += p_under
    return total


def _expected_rounds_remaining(state: "GameState") -> float:
    """``E[number of auction rounds played until game end]``, in closed form.

    Setup:
        A  = remaining auction-deck size
        T1 = number of 1-gem treasures still in the deck
        T2 = number of 2-gem treasures still in the deck
        NT = A − T1 − T2  (loans + invests)
        G  = remaining gem supply (gem_deck + revealed_gems)

    Game ends after round k when (a) cumulative gems consumed ≥ G, or
    (b) k = A. Round k is played iff fewer than G gems were consumed in
    rounds 1..k−1, so by linearity of expectation::

        E[rounds] = Σ_{k=1..A} P(first k−1 cards consumed < G gems)

    The composition of the first j cards drawn from the auction multiset
    follows a multivariate hypergeometric, so::

        P(j₁ ones, j₂ twos in first j) =
            C(T1, j₁) · C(T2, j₂) · C(NT, j − j₁ − j₂) / C(A, j)

    Sum these for ``j₁ + 2·j₂ < G``. The actual numerical work lives in
    :func:`_expected_rounds_remaining_impl`, which is ``lru_cache``'d on
    the integer signature.
    """
    A = len(state.auction_deck)
    if A == 0:
        return 0.0
    G = len(state.gem_deck) + len(state.revealed_gems)
    if G == 0:
        return 0.0  # game is already over

    T1 = 0
    T2 = 0
    for c in state.auction_deck:
        if isinstance(c, TreasureCard):
            if c.gems == 1:
                T1 += 1
            elif c.gems == 2:
                T2 += 1
    NT = A - T1 - T2
    return _expected_rounds_remaining_impl(A, T1, T2, NT, G)


# ---------------------------------------------------------------------------
# Mission-win probability heuristic.
# ---------------------------------------------------------------------------


def _min_extra_gems_to_satisfy(
    holdings,
    mission: MissionCard,
    max_k: int = 6,
) -> int:
    """Smallest k such that adding some k-multiset of colors to ``holdings``
    satisfies ``mission``. Returns 0 if already satisfied,
    ``_IMPOSSIBLE_DISTANCE`` if no multiset of size ≤ ``max_k`` works.

    Probing-based: missions are opaque predicates, so we enumerate
    candidate multisets via ``combinations_with_replacement(_COLORS, k)``
    in ascending k. Two optimizations on the naive form:

    1. **Memoized at the module level** by ``(holdings_tuple, mission.name)``
       — within one round all four players hit the cache for unchanged
       collections, and across the GA the same (collection, mission)
       pair recurs constantly.

    2. **Mutate-restore on a single working dict** instead of the
       ``holdings + Counter(combo)`` allocation per candidate. The hot
       inner loop becomes ``candidate[c] += 1`` / ``candidate[c] -= 1``
       around one ``mission.is_satisfied_by`` call — Counter's
       constructor and ``__add__`` were each in the top 3 of cProfile
       on this function before the rewrite.

    ``holdings`` may be a Counter or plain dict — both expose the
    ``.get(color, 0)`` / ``.values()`` surface that mission predicates
    rely on.
    """
    key = (_holdings_key(holdings), mission.name)
    cached = _DISTANCE_CACHE.get(key)
    if cached is not None:
        return cached

    if mission.is_satisfied_by(holdings):
        result = 0
    else:
        # Single mutable working dict; we add a combo, test, and undo.
        candidate: dict = dict(holdings)
        result = _IMPOSSIBLE_DISTANCE
        for k in range(1, max_k + 1):
            found = False
            for combo in combinations_with_replacement(_COLORS, k):
                for c in combo:
                    candidate[c] = candidate.get(c, 0) + 1
                if mission.is_satisfied_by(candidate):
                    found = True
                # Undo regardless of outcome so the dict stays at the
                # original holdings between iterations.
                for c in combo:
                    candidate[c] -= 1
                if found:
                    result = k
                    break
            if found:
                break

    if len(_DISTANCE_CACHE) >= _DISTANCE_CACHE_LIMIT:
        _DISTANCE_CACHE.clear()
    _DISTANCE_CACHE[key] = result
    return result


def _p_player_wins_mission(
    player_idx: int,
    state: "GameState",
    mission: MissionCard,
    *,
    holding_overrides: dict[int, Counter] | None = None,
) -> float:
    """Heuristic probability that ``state.player_states[player_idx]`` ends
    up claiming ``mission`` before the game ends.

    Per-player score::

        if already satisfied:
            ∞ (deterministic; lowest-seat satisfied player wins per
              engine._check_missions tie-break)
        else:
            distance = _min_extra_gems_to_satisfy(player.collection)
            if distance == _IMPOSSIBLE_DISTANCE:        score = 0
            elif sum(in_play_per_color) < distance:     score = 0
            else:
                coin_ratio = (player.coins + 1) / (avg_coins + 1)
                score      = coin_ratio / (1 + distance)

    Returned probability is normalized across players. If at least one
    player is already satisfied, the lowest-seat satisfied player gets
    1.0 and everyone else 0.0 (matches the engine's greedy first-come
    behaviour). If all scores are 0, returns 0 — caller can interpret
    that as "no one will win this mission in time".

    ``holding_overrides`` lets the caller hypothetically add gems to a
    specific player's collection without mutating state. Used by the
    auction-win and auction-lose hypotheticals in
    :func:`_mission_probability_delta`.
    """
    overrides = holding_overrides or {}
    n = state.num_players

    # Build plain dicts (not Counters) for the per-player holdings.
    # The mission predicate only needs ``.get(color, 0)`` / ``.values()``,
    # both of which dict supports — and dict copy + in-place add is
    # markedly faster than Counter.__add__, which was a hot spot.
    holdings_per_player: list[dict] = []
    for idx, ps in enumerate(state.player_states):
        h = dict(ps.collection_gems)
        ovr = overrides.get(idx)
        if ovr:
            for c, count in ovr.items():
                h[c] = h.get(c, 0) + count
        holdings_per_player.append(h)

    # Engine tie-break: lowest seat with satisfaction wins.
    for idx, h in enumerate(holdings_per_player):
        if mission.is_satisfied_by(h):
            return 1.0 if idx == player_idx else 0.0

    # In-play pool per color (= cards not in any collection).
    in_play_total = 0
    for color in _COLORS:
        held_total = 0
        for h in holdings_per_player:
            held_total += h.get(color, 0)
        in_play_total += max(0, _GEMS_PER_COLOR - held_total)

    avg_coins = sum(ps.coins for ps in state.player_states) / max(1, n)

    scores: list[float] = []
    for idx, h in enumerate(holdings_per_player):
        distance = _min_extra_gems_to_satisfy(h, mission)
        if distance >= _IMPOSSIBLE_DISTANCE or in_play_total < distance:
            scores.append(0.0)
            continue
        coin_ratio = (state.player_states[idx].coins + 1) / (avg_coins + 1)
        score = coin_ratio / (1.0 + distance)
        scores.append(score)

    total = sum(scores)
    if total == 0.0:
        return 0.0
    return scores[player_idx] / total


def _mission_probability_delta(
    auction: TreasureCard,
    state: "GameState",
    my_state: "PlayerState",
) -> float:
    """Σ over active missions of ``(p_win − p_lose) · mission.coins``.

    ``p_win``  = P(I win mission | I take the gems on offer).
    ``p_lose`` = P(I win mission | the highest-coin opponent takes them).

    The "highest-coin opponent" is a cheap proxy for the most likely
    auction winner — modelling the bidding fixed point properly would
    cost far more than this whole helper. The simplification is signed
    correctly: if winning the gems makes it easier for me and harder
    for the most-threatening opponent, the delta is positive.

    Returns 0 when there are no gems on offer or no active missions.
    """
    gems_for_sale = state.revealed_gems[: min(auction.gems, len(state.revealed_gems))]
    if not gems_for_sale or not state.active_missions:
        return 0.0

    extra = Counter(g.color for g in gems_for_sale)
    my_idx = state.player_states.index(my_state)

    opp_idxs = [i for i in range(state.num_players) if i != my_idx]
    if not opp_idxs:
        return 0.0
    likely_opp = max(opp_idxs, key=lambda i: state.player_states[i].coins)

    delta = 0.0
    for mission in state.active_missions:
        p_win = _p_player_wins_mission(
            my_idx, state, mission, holding_overrides={my_idx: extra}
        )
        p_lose = _p_player_wins_mission(
            my_idx, state, mission, holding_overrides={likely_opp: extra}
        )
        delta += (p_win - p_lose) * mission.coins
    return delta


# ---------------------------------------------------------------------------
# Linear bid models. Three sibling classes — they don't share a base
# because their feature counts differ and a single base would invite
# padding bugs.
# ---------------------------------------------------------------------------


class _Evo2Features:
    """Four state-only features shared by every head.

    Per-head specific features (treasure ev/std, invest amount, loan amount)
    are passed to each model separately rather than stuffed in here, so
    the heads can have different feature counts without padding zeros.
    """

    __slots__ = ("rounds_remaining", "my_coins", "avg_opp_coins", "top_opp_coins")

    def __init__(
        self,
        rounds_remaining: float,
        my_coins: float,
        avg_opp_coins: float,
        top_opp_coins: float,
    ) -> None:
        self.rounds_remaining = rounds_remaining
        self.my_coins = my_coins
        self.avg_opp_coins = avg_opp_coins
        self.top_opp_coins = top_opp_coins


def _compute_evo2_features(
    state: "GameState", my_state: "PlayerState"
) -> _Evo2Features:
    rounds = _expected_rounds_remaining(state)
    opp = [ps.coins for ps in state.player_states if ps is not my_state]
    return _Evo2Features(
        rounds_remaining=rounds,
        my_coins=float(my_state.coins),
        avg_opp_coins=(sum(opp) / len(opp)) if opp else 0.0,
        top_opp_coins=float(max(opp)) if opp else 0.0,
    )


class _TreasureModel:
    """1 bias + 4 shared + 2 specific (ev, std) = 7 weights.

    Outputs the **bid in coins** directly:

        bid = bias + w_rounds·rounds + w_my·my_coins + w_avg·avg_opp
              + w_top·top_opp + w_ev·ev + w_std·std

    The result is a raw float — clamping to ``[0, cap]`` is done by
    ``Evo2AI.choose_bid``, so this method has no built-in saturation.
    """

    __slots__ = ("bias", "w_rounds", "w_my", "w_avg", "w_top", "w_ev", "w_std")

    def __init__(
        self,
        bias: float,
        w_rounds: float,
        w_my: float,
        w_avg: float,
        w_top: float,
        w_ev: float,
        w_std: float,
    ) -> None:
        self.bias = bias
        self.w_rounds = w_rounds
        self.w_my = w_my
        self.w_avg = w_avg
        self.w_top = w_top
        self.w_ev = w_ev
        self.w_std = w_std

    def bid(self, f: _Evo2Features, ev: float, std: float) -> float:
        return (
            self.bias
            + self.w_rounds * f.rounds_remaining
            + self.w_my * f.my_coins
            + self.w_avg * f.avg_opp_coins
            + self.w_top * f.top_opp_coins
            + self.w_ev * ev
            + self.w_std * std
        )


class _InvestModel:
    """1 bias + 4 shared + 1 specific (amount) = 6 weights.

    Outputs the bid in coins directly. See :class:`_TreasureModel`.
    """

    __slots__ = ("bias", "w_rounds", "w_my", "w_avg", "w_top", "w_amount")

    def __init__(
        self,
        bias: float,
        w_rounds: float,
        w_my: float,
        w_avg: float,
        w_top: float,
        w_amount: float,
    ) -> None:
        self.bias = bias
        self.w_rounds = w_rounds
        self.w_my = w_my
        self.w_avg = w_avg
        self.w_top = w_top
        self.w_amount = w_amount

    def bid(self, f: _Evo2Features, amount: int) -> float:
        return (
            self.bias
            + self.w_rounds * f.rounds_remaining
            + self.w_my * f.my_coins
            + self.w_avg * f.avg_opp_coins
            + self.w_top * f.top_opp_coins
            + self.w_amount * amount
        )


class _LoanModel:
    """1 bias + 4 shared + 1 specific (amount) = 6 weights.

    Structurally identical to ``_InvestModel`` but kept as its own class
    so the GA's flat-vector slicing is type-clear.
    """

    __slots__ = ("bias", "w_rounds", "w_my", "w_avg", "w_top", "w_amount")

    def __init__(
        self,
        bias: float,
        w_rounds: float,
        w_my: float,
        w_avg: float,
        w_top: float,
        w_amount: float,
    ) -> None:
        self.bias = bias
        self.w_rounds = w_rounds
        self.w_my = w_my
        self.w_avg = w_avg
        self.w_top = w_top
        self.w_amount = w_amount

    def bid(self, f: _Evo2Features, amount: int) -> float:
        return (
            self.bias
            + self.w_rounds * f.rounds_remaining
            + self.w_my * f.my_coins
            + self.w_avg * f.avg_opp_coins
            + self.w_top * f.top_opp_coins
            + self.w_amount * amount
        )


# ---------------------------------------------------------------------------
# Evo2AI itself.
# ---------------------------------------------------------------------------


class Evo2AI(BaseEvoAI):
    """Clean-slate evolved AI. See module docstring for design notes.

    Subclasses :class:`BaseEvoAI` (the shared scaffolding for the
    evolved chain) rather than ``HyperAdaptiveSplitAI`` so the
    inherited reserve / discount-feature plumbing doesn't bleed in.
    The reveal logic comes from ``BaseEvoAI.choose_gem_to_reveal``,
    which is a direct lift of ``HeuristicAI.choose_gem_to_reveal`` —
    the user's redesign is purely about bidding.
    """

    NUM_WEIGHTS = 19  # 7 (treasure) + 6 (invest) + 6 (loan)

    # Defaults pulled from a self-play GA run
    # (population=24, generation 0, fitness 0.405 vs 3× self-play opponents
    # on charts A–E, 4 players). Used when no per-seat-count weights file
    # exists in artifacts/ — ``_evo2_factory`` in ``megagem/__main__.py``
    # falls back to these so ``--ai evo2`` works out of the box.
    #
    # All numbers are in *coin units of bid output*, since the heads now
    # produce the bid directly (no discount-fraction × EV scaling).
    DEFAULT_TREASURE = _TreasureModel(
        bias=0.9671062444221764,
        w_rounds=-0.0906995616980441,
        w_my=0.07804979550128198,
        w_avg=0.05375147152736104,
        w_top=-0.04247465810129918,
        w_ev=0.32783828473034604,
        w_std=-0.011838494331700117,
    )
    DEFAULT_INVEST = _InvestModel(
        bias=1.908464547879478,
        w_rounds=0.4300303741599258,
        w_my=-0.1201852409204779,
        w_avg=-0.28421403664160627,
        w_top=0.3149361220138405,
        w_amount=0.07219353469220569,
    )
    DEFAULT_LOAN = _LoanModel(
        bias=-0.4139242208454687,
        w_rounds=-0.31190499765072527,
        w_my=0.13966251262722051,
        w_avg=0.12135141558388368,
        w_top=-0.0669196243751372,
        w_amount=0.36349000133503273,
    )

    def __init__(
        self,
        name: str,
        *,
        treasure: _TreasureModel | None = None,
        invest: _InvestModel | None = None,
        loan: _LoanModel | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(name, seed=seed)
        self.treasure_model = treasure if treasure is not None else self.DEFAULT_TREASURE
        self.invest_model = invest if invest is not None else self.DEFAULT_INVEST
        self.loan_model = loan if loan is not None else self.DEFAULT_LOAN

    @classmethod
    def from_weights(
        cls,
        name: str,
        weights: list[float],
        *,
        seed: int | None = None,
    ) -> "Evo2AI":
        """Build from a flat 19-element weights list.

        Layout: ``[treasure(7), invest(6), loan(6)]``.
        """
        if len(weights) != cls.NUM_WEIGHTS:
            raise ValueError(
                f"Expected {cls.NUM_WEIGHTS} weights, got {len(weights)}"
            )
        return cls(
            name,
            treasure=_TreasureModel(*weights[0:7]),
            invest=_InvestModel(*weights[7:13]),
            loan=_LoanModel(*weights[13:19]),
            seed=seed,
        )

    @classmethod
    def flatten_defaults(cls) -> list[float]:
        """Return the class-level ``DEFAULT_*`` constants as a flat 19-vector.

        The inverse of :meth:`from_weights`: feeding this result back
        through ``from_weights`` reconstructs the class-default AI. The
        unified GA tuner uses this as its fallback for individual #0
        when no saved weights file exists yet.
        """
        t = cls.DEFAULT_TREASURE
        i = cls.DEFAULT_INVEST
        l = cls.DEFAULT_LOAN
        return [
            t.bias, t.w_rounds, t.w_my, t.w_avg, t.w_top, t.w_ev, t.w_std,
            i.bias, i.w_rounds, i.w_my, i.w_avg, i.w_top, i.w_amount,
            l.bias, l.w_rounds, l.w_my, l.w_avg, l.w_top, l.w_amount,
        ]

    def choose_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> int:
        cap = max_legal_bid(my_state, auction)
        if cap == 0:
            return 0

        f = _compute_evo2_features(public_state, my_state)

        if isinstance(auction, TreasureCard):
            ev, std = _treasure_value_stats(auction, public_state, my_state)
            raw = self.treasure_model.bid(f, ev, std)
            return max(0, min(int(raw), cap))

        if isinstance(auction, InvestCard):
            raw = self.invest_model.bid(f, auction.amount)
            bid = max(0, min(int(raw), cap))
            # Free money — always grab a token bid if we can.
            if bid == 0 and cap > 0:
                bid = 1
            return bid

        if isinstance(auction, LoanCard):
            raw = self.loan_model.bid(f, auction.amount)
            return max(0, min(int(raw), cap))

        return 0

    # --- Debug rationale --------------------------------------------------

    def explain_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> list[str]:
        """Detail lines for ``--debug`` rationale output.

        The ``heads:`` line shows the **raw float bid** each head would
        produce for the current state — these are the linear-model
        outputs in coin units, before clamping to ``[0, cap]``. The
        active head (matching the auction kind) is marked with ``◀``.
        """
        f = _compute_evo2_features(public_state, my_state)
        # The amount feature for invest/loan is auction-specific. For the
        # heads we don't fire, fall back to the most common amounts so the
        # printout still gives a meaningful number rather than a literal 0.
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
        tb = self.treasure_model.bid(f, treasure_ev, treasure_std)
        ib = self.invest_model.bid(f, invest_amount)
        lb = self.loan_model.bid(f, loan_amount)

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
        ]
        if isinstance(auction, TreasureCard):
            delta = _mission_probability_delta(auction, public_state, my_state)
            lines.append(
                f"treasure:  ev={treasure_ev:.1f}  std={treasure_std:.2f}  "
                f"mission_delta={delta:+.2f}"
            )
        return lines
