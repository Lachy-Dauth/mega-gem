"""Shared helpers for the MegaGem player zoo.

All value-estimation, discount-feature, and hypergeometric math used by
more than one AI lives here. The individual AI files import from this
module so the per-file implementations stay focused on their bidding
policy and reveal logic.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

from ..cards import Color, TreasureCard
from ..value_charts import VALUE_CHARTS, value_for

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


_GEMS_PER_COLOR = 6

# matches the breakdown in cards.make_auction_deck()
_TOTAL_AUCTIONS = 25


# ----------------------------------------------------------------------------
# Point-estimate value model (HeuristicAI family).
# ----------------------------------------------------------------------------


def _expected_final_display(
    state: "GameState", my_state: "PlayerState"
) -> dict[Color, float]:
    """Estimate the count of each color in the Value Display at game end.

    Final display = (current display) + (every gem currently in any hand).
    My hand is known. Opponents' hands are estimated by spreading the
    "hidden" gems of each color uniformly across hidden card slots
    (opponents' hands + the face-down gem deck).
    """
    display = state.value_display
    my_collection = my_state.collection_gems
    revealed_pool = Counter(g.color for g in state.revealed_gems)
    my_hand = Counter(g.color for g in my_state.hand)

    other_collection = Counter()
    opp_hand_size = 0
    for ps in state.player_states:
        if ps is my_state:
            continue
        other_collection.update(ps.collection_gems)
        opp_hand_size += len(ps.hand)

    deck_size = len(state.gem_deck)
    hidden_total_slots = opp_hand_size + deck_size
    opp_share = (opp_hand_size / hidden_total_slots) if hidden_total_slots else 0.0

    expected: dict[Color, float] = {}
    for color in Color:
        seen = (
            display.get(color, 0)
            + my_collection.get(color, 0)
            + my_hand.get(color, 0)
            + revealed_pool.get(color, 0)
            + other_collection.get(color, 0)
        )
        hidden_of_color = max(0, _GEMS_PER_COLOR - seen)
        expected_in_opp_hand = hidden_of_color * opp_share
        expected[color] = (
            display.get(color, 0)
            + my_hand.get(color, 0)
            + expected_in_opp_hand
        )
    return expected


def _chart_value(chart: str, count_float: float) -> int:
    """Round a float count to nearest integer index, clamp 0..5, lookup chart."""
    idx = int(round(count_float))
    if idx < 0:
        idx = 0
    if idx > 5:
        idx = 5
    return value_for(chart, idx)


def _marginal_gem_value(
    color: Color,
    expected_display: dict[Color, float],
    chart: str,
    current_owned: int,
) -> int:
    """How many coins one extra gem of this color is worth to me right now."""
    per_gem = _chart_value(chart, expected_display.get(color, 0.0))
    return per_gem  # owning N+1 vs N is one extra unit at the per-gem rate.


def _mission_completion_bonus(
    my_state: "PlayerState",
    active_missions,
    extra_gems: Counter,
) -> int:
    """How many mission coins would I claim if I added `extra_gems` to my collection?"""
    if not active_missions:
        return 0
    hypothetical = my_state.collection_gems + extra_gems
    bonus = 0
    for mission in active_missions:
        if mission.is_satisfied_by(my_state.collection_gems):
            continue  # already complete (shouldn't happen, but defensive)
        if mission.is_satisfied_by(hypothetical):
            bonus += mission.coins
    return bonus


def _mission_progress_bonus(
    my_state: "PlayerState",
    active_missions,
    extra_gems: Counter,
) -> int:
    """Soft credit for missions you're closer to but don't yet complete.

    Worth ~1/3 of the mission's value if the new gems strictly increase your
    coverage of any required color in any mission you have at least one piece of.
    Cheap to compute, prevents the AI from ignoring missions until the last moment.
    """
    if not active_missions:
        return 0
    hypothetical = my_state.collection_gems + extra_gems
    soft = 0
    for mission in active_missions:
        if mission.is_satisfied_by(my_state.collection_gems):
            continue
        if mission.is_satisfied_by(hypothetical):
            continue  # captured by the hard bonus
        # Generic heuristic: if any color in extra_gems is one this mission
        # cares about (i.e. having more of it makes us "satisfied" in a
        # hypothetical fully-stocked future), credit a third of the value.
        for color in extra_gems:
            stretched = hypothetical.copy()
            stretched[color] += 2  # imagine getting two more of this color later
            if mission.is_satisfied_by(stretched) and not mission.is_satisfied_by(hypothetical):
                soft += mission.coins // 3
                break
    return soft


def _treasure_value(
    auction: TreasureCard,
    state: "GameState",
    my_state: "PlayerState",
) -> int:
    """Estimated coin value of winning this treasure auction."""
    expected_display = _expected_final_display(state, my_state)
    gems_for_sale = state.revealed_gems[: min(auction.gems, len(state.revealed_gems))]
    if not gems_for_sale:
        return 0

    # Value each gem independently using the same expected final-display
    # count for its color. This heuristic intentionally does not apply
    # additional within-auction marginal adjustments based on other gems
    # won in the same treasure; bundle effects are captured separately
    # via the mission bonuses below.
    extra = Counter()
    gem_value = 0
    for gem in gems_for_sale:
        # Bought gems go to our collection, not the display, so the
        # per-gem chart lookup uses the unmodified expected display estimate.
        gem_value += _chart_value(state.value_chart, expected_display[gem.color])
        extra[gem.color] += 1

    mission_hard = _mission_completion_bonus(my_state, state.active_missions, extra)
    mission_soft = _mission_progress_bonus(my_state, state.active_missions, extra)
    return gem_value + mission_hard + mission_soft


def _remaining_supply(state: "GameState") -> int:
    return len(state.gem_deck) + len(state.revealed_gems)


def _expected_avg_treasure_value(
    state: "GameState", my_state: "PlayerState"
) -> float:
    """Rough average value of one of the next gems we might win."""
    expected_display = _expected_final_display(state, my_state)
    if not expected_display:
        return 0.0
    avg = sum(
        _chart_value(state.value_chart, expected_display[c]) for c in Color
    ) / len(Color)
    return avg


# ----------------------------------------------------------------------------
# Discount-feature model shared by every Adaptive* AI.
# ----------------------------------------------------------------------------


class _DiscountFeatures:
    """Plain holder for the five inputs to the discount-rate model.

    Not a dataclass — keeps this file dependency-free at the top level and
    makes the field order obvious to the dot-product below.
    """

    __slots__ = ("progress", "my_cash_ratio", "avg_cash_ratio", "top_cash_ratio", "variance")

    def __init__(
        self,
        progress: float,
        my_cash_ratio: float,
        avg_cash_ratio: float,
        top_cash_ratio: float,
        variance: float,
    ) -> None:
        self.progress = progress
        self.my_cash_ratio = my_cash_ratio
        self.avg_cash_ratio = avg_cash_ratio
        self.top_cash_ratio = top_cash_ratio
        self.variance = variance


def _format_discount_features(f: _DiscountFeatures) -> str:
    """One-line summary used by the ``explain_bid`` overrides for every
    AI in the AdaptiveHeuristic family. Centralised so the format string
    only lives in one place."""
    return (
        f"features:  progress={f.progress:.2f}  "
        f"my_cash={f.my_cash_ratio:.2f}  avg_cash={f.avg_cash_ratio:.2f}  "
        f"top_cash={f.top_cash_ratio:.2f}  var={f.variance:.2f}"
    )


def _ev_remaining_auctions(
    state: "GameState", my_state: "PlayerState"
) -> float:
    """Rough expected total coin payout still locked in the auction deck.

    Uses the existing per-gem average × remaining sellable gems. Loans and
    investments are intentionally excluded — they're cash-flow neutral and
    would distort the cash-vs-gems ratios used by the discount model.
    """
    avg_per_gem = _expected_avg_treasure_value(state, my_state)
    sellable_gems = len(state.revealed_gems) + len(state.gem_deck)
    return avg_per_gem * sellable_gems


def _compute_discount_features(
    state: "GameState", my_state: "PlayerState"
) -> _DiscountFeatures:
    # Game progress: how much of the auction deck has been consumed.
    auctions_left = len(state.auction_deck)
    progress = max(0.0, min(1.0, 1.0 - auctions_left / _TOTAL_AUCTIONS))

    ev_remaining = max(1.0, _ev_remaining_auctions(state, my_state))

    opp_coins = [ps.coins for ps in state.player_states if ps is not my_state]
    avg_opp = (sum(opp_coins) / len(opp_coins)) if opp_coins else 0.0
    top_opp = max(opp_coins) if opp_coins else 0.0

    my_cash_ratio = my_state.coins / ev_remaining
    avg_cash_ratio = avg_opp / ev_remaining
    top_cash_ratio = top_opp / ev_remaining

    # Variance proxy: hidden-card fraction × chart sensitivity. Will be
    # replaced with a true Var[chart_value] once the probabilistic estimator
    # lands; the units stay the same so the weights below carry over.
    hidden = sum(len(ps.hand) for ps in state.player_states if ps is not my_state)
    hidden += len(state.gem_deck)
    chart_table = VALUE_CHARTS[state.value_chart]
    chart_swing = (max(chart_table) - min(chart_table)) / 20.0
    variance = (hidden / 30.0) * chart_swing

    return _DiscountFeatures(
        progress=progress,
        my_cash_ratio=my_cash_ratio,
        avg_cash_ratio=avg_cash_ratio,
        top_cash_ratio=top_cash_ratio,
        variance=variance,
    )


# ----------------------------------------------------------------------------
# Hypergeometric value estimator.
#
# HeuristicAI's `_expected_final_display` collapses the unknown opponent-held
# gems into a single point estimate per color and then evaluates the value
# chart at that point. That throws away the full distribution and is wrong
# for any non-linear chart — most obviously chart E, which peaks at 3 gems
# and then crashes. The fix is to compute the full distribution of
# `final_display[c]` for each color (it's hypergeometric over the hidden
# pool) and take E[chart_value(X)] instead of chart_value(E[X]).
# ----------------------------------------------------------------------------


def _hyper_hidden_distribution(
    state: "GameState", my_state: "PlayerState"
) -> dict[Color, dict[int, float]]:
    """For each color c, return ``{final_display_count: probability}``.

    Model: every gem currently in any hand will eventually be revealed into
    the value display (by auction-win reveals plus end-of-game reveals), so

        final_display[c] = display[c] + my_hand[c] + (color-c cards in opp hands)

    The first two terms are known exactly. The third is the unknown. Treat
    the H = (opp_hand_total + deck_size) hidden card slots as an urn with
    `hidden[c]` cards of color c and (H - hidden[c]) of any other color;
    opponents collectively hold `opp_hand_total` of those H cards drawn
    without replacement. The number of color-c cards in their hands is then
    exactly hypergeometric.
    """
    display = state.value_display
    my_collection = my_state.collection_gems
    revealed_pool = Counter(g.color for g in state.revealed_gems)
    my_hand = Counter(g.color for g in my_state.hand)

    other_collection: Counter = Counter()
    opp_hand_total = 0
    for ps in state.player_states:
        if ps is my_state:
            continue
        other_collection.update(ps.collection_gems)
        opp_hand_total += len(ps.hand)

    deck_size = len(state.gem_deck)
    hidden_total = opp_hand_total + deck_size  # total hidden card slots ("H")

    # The hypergeometric denominator only depends on (hidden_total,
    # opp_hand_total), so it's the same for every color. Hoist it out of
    # the per-color loop — math.comb on the big-int sizes here is the
    # hottest single op in this function.
    has_randomness = hidden_total > 0 and opp_hand_total > 0
    denom = math.comb(hidden_total, opp_hand_total) if has_randomness else 0

    distributions: dict[Color, dict[int, float]] = {}
    for color in Color:
        seen = (
            display.get(color, 0)
            + my_collection.get(color, 0)
            + my_hand.get(color, 0)
            + revealed_pool.get(color, 0)
            + other_collection.get(color, 0)
        )
        hidden_of_color = max(0, _GEMS_PER_COLOR - seen)
        known_offset = display.get(color, 0) + my_hand.get(color, 0)

        dist: dict[int, float] = {}
        if not has_randomness or hidden_of_color == 0:
            # No randomness left for this color: either no opponent slots
            # exist, opponents hold no hidden cards, or none of those cards
            # could possibly be color c. Final count is exactly known_offset.
            dist[known_offset] = 1.0
        else:
            k_min = max(0, opp_hand_total - (hidden_total - hidden_of_color))
            k_max = min(hidden_of_color, opp_hand_total)
            for k in range(k_min, k_max + 1):
                num = math.comb(hidden_of_color, k) * math.comb(
                    hidden_total - hidden_of_color, opp_hand_total - k
                )
                dist[known_offset + k] = num / denom
        distributions[color] = dist
    return distributions


def _hyper_expected_per_gem_value(
    state: "GameState", my_state: "PlayerState", chart: str
) -> dict[Color, float]:
    """E[chart_value(final_display[c])] for each color.

    For each color, sums chart values weighted by the hypergeometric
    distribution returned by ``_hyper_hidden_distribution``. For monotonic
    charts (A, B, C, D) this is a small accuracy bump over a point estimate;
    for chart E (peaked at 3 gems) it can swing significantly because the
    chart is non-linear and the distribution is wide.
    """
    distributions = _hyper_hidden_distribution(state, my_state)
    per_gem: dict[Color, float] = {}
    for color, dist in distributions.items():
        ev = 0.0
        for count, p in dist.items():
            ev += p * value_for(chart, min(count, 5))
        per_gem[color] = ev
    return per_gem


def _hyper_treasure_gem_value(
    auction: TreasureCard,
    state: "GameState",
    my_state: "PlayerState",
    per_gem: dict[Color, float] | None = None,
) -> float:
    """Gem-only expected coin value of winning this treasure auction.

    Adds ``per_gem[c]`` for each visible gem on offer. Bought gems land in
    our collection, not the display, so the per-gem chart expectation is
    unchanged across the two gems of a 2-gem treasure even when both gems
    share a color — the doubling falls out for free.
    """
    if per_gem is None:
        per_gem = _hyper_expected_per_gem_value(state, my_state, state.value_chart)
    gems_for_sale = state.revealed_gems[: min(auction.gems, len(state.revealed_gems))]
    return sum(per_gem[g.color] for g in gems_for_sale)


def _hyper_treasure_value(
    auction: TreasureCard,
    state: "GameState",
    my_state: "PlayerState",
) -> float:
    """Estimated total coin value of winning this treasure auction.

    Sum of expected gem value plus mission completion / progress bonuses.
    Missions are evaluated against the *bundled* extra gems so that joint
    effects are captured (e.g. a 2-gem (Blue, Green) treasure completing a
    "1 Blue + 1 Green" pendant counts the +5 even though neither gem alone
    would).
    """
    gems_for_sale = state.revealed_gems[: min(auction.gems, len(state.revealed_gems))]
    if not gems_for_sale:
        return 0.0
    per_gem = _hyper_expected_per_gem_value(state, my_state, state.value_chart)
    gem_v = _hyper_treasure_gem_value(auction, state, my_state, per_gem=per_gem)
    extra: Counter = Counter(g.color for g in gems_for_sale)
    mission_hard = _mission_completion_bonus(my_state, state.active_missions, extra)
    mission_soft = _mission_progress_bonus(my_state, state.active_missions, extra)
    return gem_v + mission_hard + mission_soft


def _hyper_avg_treasure_value(
    state: "GameState", my_state: "PlayerState"
) -> float:
    """Average of the per-color expected per-gem values. Used for reserves."""
    per_gem = _hyper_expected_per_gem_value(state, my_state, state.value_chart)
    if not per_gem:
        return 0.0
    return sum(per_gem.values()) / len(per_gem)


def _hyper_ev_remaining_auctions(
    state: "GameState", my_state: "PlayerState"
) -> float:
    """Hyper-aware version of ``_ev_remaining_auctions``.

    Same shape as the original — average per-gem value × remaining sellable
    gems — but the per-gem value comes from the hypergeometric estimator so
    the cash-vs-gems ratios fed into the discount model are computed against
    the same value scale the bidder is actually using.
    """
    avg_per_gem = _hyper_avg_treasure_value(state, my_state)
    sellable_gems = len(state.revealed_gems) + len(state.gem_deck)
    return avg_per_gem * sellable_gems


def _hyper_compute_discount_features(
    state: "GameState", my_state: "PlayerState"
) -> _DiscountFeatures:
    """Hyper-aware version of ``_compute_discount_features``.

    Identical structure to the original; the only difference is the EV
    denominator, which uses ``_hyper_ev_remaining_auctions``. Variance proxy
    is unchanged for now (a true Var[chart_value] from the distribution
    would be a separate refinement and is out of scope for this step).
    """
    auctions_left = len(state.auction_deck)
    progress = max(0.0, min(1.0, 1.0 - auctions_left / _TOTAL_AUCTIONS))

    ev_remaining = max(1.0, _hyper_ev_remaining_auctions(state, my_state))

    opp_coins = [ps.coins for ps in state.player_states if ps is not my_state]
    avg_opp = (sum(opp_coins) / len(opp_coins)) if opp_coins else 0.0
    top_opp = max(opp_coins) if opp_coins else 0.0

    my_cash_ratio = my_state.coins / ev_remaining
    avg_cash_ratio = avg_opp / ev_remaining
    top_cash_ratio = top_opp / ev_remaining

    hidden = sum(len(ps.hand) for ps in state.player_states if ps is not my_state)
    hidden += len(state.gem_deck)
    chart_table = VALUE_CHARTS[state.value_chart]
    chart_swing = (max(chart_table) - min(chart_table)) / 20.0
    variance = (hidden / 30.0) * chart_swing

    return _DiscountFeatures(
        progress=progress,
        my_cash_ratio=my_cash_ratio,
        avg_cash_ratio=avg_cash_ratio,
        top_cash_ratio=top_cash_ratio,
        variance=variance,
    )
