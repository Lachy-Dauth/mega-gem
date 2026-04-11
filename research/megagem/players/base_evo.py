"""Shared base class + constants for the evolved AIs.

:class:`BaseEvoAI` hoists the scaffolding that ``Evo2AI``, ``Evo3AI``,
and ``Evo4AI`` previously re-implemented independently:

* ``__init__`` that stores the player name, seeds a private
  ``random.Random`` (currently unused but kept for subclass flexibility),
  and initializes the Evo3/Evo4 opponent-history fields
  (``_opp_history`` / ``_last_default_bid``). The history fields are
  created for every evolved AI — Evo2 never reads them, but the cost
  is one empty list and one ``None`` per instance, which is the
  cheapest way to keep Evo3/Evo4's ``observe_round`` free of init
  branching.
* ``choose_gem_to_reveal`` — the chart-delta-× relative-holdings reveal
  policy originally copied from ``HeuristicAI.choose_gem_to_reveal``
  and duplicated verbatim in all three Evo modules.
* ``_record_opp_delta`` — the pop-baseline / extract-bids /
  append-to-history logic Evo3 and Evo4 both need inside
  ``observe_round``. Evo4 additionally extends it with a color-signal
  update.
* ``_bid_amount_auction_with_delta`` — the amount-bid branch (invest /
  loan) used by Evo3 and Evo4, which runs the model twice (once with
  the current history-weighted deltas, once with the ``(0.0, 1.0)``
  default deltas) and returns both clamped integer bids.

The opponent-pricing category tags (``_CAT_*``), the matching-category
weight constants, the ``(0.0, 1.0)`` default-delta fallbacks, and the
pure helpers ``_category_of`` / ``_weighted_delta_stats`` also live
here so they have a single canonical home. They are re-exported from
``evo3.py`` for backward compatibility — the existing tests and
``evo4.py`` import them from there by name.

Per-generation feature math (``_treasure_value_stats``, the per-head
linear models, ``choose_bid``, ``explain_bid``, ``from_weights``,
``flatten_defaults``) still lives in each ``evo{N}.py`` because the
feature vectors and weight layouts differ per generation. This base
class only owns what is truly identical.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import TYPE_CHECKING

from ..cards import AuctionCard, GemCard, InvestCard, LoanCard, TreasureCard
from ..value_charts import value_for
from .base import Player

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


# ---------------------------------------------------------------------------
# Opponent-pricing category tags and weighted-history constants.
# ---------------------------------------------------------------------------

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


class BaseEvoAI(Player):
    """Common base for the evolved AI chain (Evo2 → Evo3 → Evo4).

    Subclasses must still implement :meth:`choose_bid`; the
    :class:`Player` ABC declares it abstract and this base class does
    not try to guess a generic dispatch. Reveal policy
    (:meth:`choose_gem_to_reveal`), the opponent-history scratch
    state, and the shared observe-round / amount-bid helpers live
    here.
    """

    def __init__(self, name: str, *, seed: int | None = None) -> None:
        super().__init__(name)
        # Private RNG — reserved for subclass use; none of the current
        # Evo AIs consume it, but keeping it here means a future evolved
        # AI can reach for ``self._rng`` without re-adding the import.
        self._rng = random.Random(seed)
        # Opp-delta history used by ``Evo3AI.observe_round`` and
        # ``Evo4AI.observe_round``. Evo2 never touches either field,
        # which costs one empty list + one ``None`` per instance — a
        # trivial price for keeping the subclass ``__init__`` bodies
        # free of repeated bookkeeping.
        self._opp_history: list[tuple[str, float]] = []
        # Scratch cache populated by ``choose_bid`` in Evo3/Evo4 with
        # the "baseline" bid (what the AI would bid with default
        # opponent-delta inputs). ``observe_round`` reads and clears it
        # so a stale value can't leak across rounds.
        self._last_default_bid: int | None = None

    # ------------------------------------------------------------------
    # Reveal policy — direct lift of the formerly-duplicated copies in
    # Evo2AI / Evo3AI / Evo4AI, which in turn copied HeuristicAI.
    # Canonical home for the logic is now here.
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

    # ------------------------------------------------------------------
    # Shared observe-round helper (Evo3 + Evo4).
    # ------------------------------------------------------------------
    def _record_opp_delta(
        self,
        my_idx: int,
        result: dict,
    ) -> tuple[str, float] | None:
        """Pop the cached baseline, append the opp-delta to history.

        Unconditionally clears ``self._last_default_bid`` so a round
        that gets skipped (unknown auction kind, no opponents, missing
        baseline) can't leak a stale value into the next observation.
        When the observation is valid, appends ``(category, delta)``
        to ``self._opp_history`` and returns the same tuple. Returns
        ``None`` when the round should be skipped.

        Evo3's :meth:`observe_round` is a one-line wrapper around
        this; Evo4 uses the returned tuple to drive its additional
        per-color signal update.
        """
        baseline = self._last_default_bid
        self._last_default_bid = None

        auction = result.get("auction")
        cat = _category_of(auction) if auction is not None else None
        if cat is None:
            return None
        if baseline is None:
            # choose_bid wasn't called (or didn't cache a baseline)
            # this round — skip the observation rather than fall back
            # to the actual bid, so the delta definition stays
            # consistent.
            return None
        bids = result.get("bids") or []
        opp_bids = [b for i, b in enumerate(bids) if i != my_idx]
        if not opp_bids:
            return None
        delta = float(max(opp_bids) - baseline)
        self._opp_history.append((cat, delta))
        return cat, delta

    # ------------------------------------------------------------------
    # Shared amount-auction bid helper (Evo3 + Evo4 invest/loan).
    # ------------------------------------------------------------------
    def _bid_amount_auction_with_delta(
        self,
        model,
        features,
        amount: int,
        cap: int,
        category: str,
        *,
        apply_token_rule: bool,
    ) -> tuple[int, int]:
        """Run ``model.bid`` for an amount-based auction (invest / loan).

        Computes both the *actual* bid (using the history-weighted
        ``(mean_delta, std_delta)`` for ``category``) and the
        *baseline* bid (using the ``(0.0, 1.0)`` defaults), each
        clamped to ``[0, cap]``. When ``apply_token_rule`` is true
        (invest auctions — free money), a zero clamped bid is
        promoted to ``1`` on both paths so the recorded baseline
        matches what ``choose_bid`` would actually return for an
        empty history.

        Returns ``(actual_bid, default_bid)``. The caller is expected
        to stash the default on ``self._last_default_bid`` and
        return ``actual_bid`` from ``choose_bid``.
        """
        mean_delta, std_delta = _weighted_delta_stats(
            self._opp_history, category
        )
        actual_raw = model.bid(features, amount, mean_delta, std_delta)
        default_raw = model.bid(
            features, amount, _DEFAULT_MEAN_DELTA, _DEFAULT_STD_DELTA
        )
        actual_bid = max(0, min(int(actual_raw), cap))
        default_bid = max(0, min(int(default_raw), cap))
        if apply_token_rule and cap > 0:
            if actual_bid == 0:
                actual_bid = 1
            if default_bid == 0:
                default_bid = 1
        return actual_bid, default_bid
