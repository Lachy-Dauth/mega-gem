"""Player implementations for MegaGem.

The `Player` ABC is the modular AI seam: subclass it to plug in a smarter
strategy. The engine clamps any returned bid to the legal range, so AIs
cannot accidentally produce illegal moves — they will simply be capped.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING

from .cards import AuctionCard, Color, GemCard, InvestCard, LoanCard, TreasureCard
from .engine import max_legal_bid
from .value_charts import VALUE_CHARTS, value_for

if TYPE_CHECKING:
    from .state import GameState, PlayerState


class Player(ABC):
    """Base class for any MegaGem player (human or AI)."""

    name: str
    is_human: bool = False

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def choose_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> int:
        ...

    @abstractmethod
    def choose_gem_to_reveal(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
    ) -> GemCard:
        ...


class RandomAI(Player):
    """Picks bids and gem reveals uniformly at random over legal options."""

    def __init__(self, name: str, seed: int | None = None) -> None:
        super().__init__(name)
        self._rng = random.Random(seed)

    def choose_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> int:
        cap = max_legal_bid(my_state, auction)
        return self._rng.randint(0, cap)

    def choose_gem_to_reveal(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
    ) -> GemCard:
        return self._rng.choice(my_state.hand)


class HumanPlayer(Player):
    """CLI-driven human player. Lazily imports `render` to avoid cycles."""

    is_human = True

    def __init__(self, name: str = "You", debug: bool = False) -> None:
        super().__init__(name)
        self.debug = debug

    def choose_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> int:
        from . import render

        print(render.render_board(public_state, debug=self.debug))
        print()
        print(f"Auction card: {auction}")
        if isinstance(auction, TreasureCard):
            gems_for_sale = public_state.revealed_gems[: auction.gems]
            label = ", ".join(str(g) for g in gems_for_sale) if gems_for_sale else "(none)"
            print(f"  For sale: {label}")
        elif isinstance(auction, LoanCard):
            print(f"  Win to receive {auction.amount} coins; pay {auction.amount} back at game end.")
        elif isinstance(auction, InvestCard):
            print(f"  Win to lock your bid + receive an extra {auction.amount} at game end.")
        print()
        print(render.render_hand(my_state))

        cap = max_legal_bid(my_state, auction)
        prompt_extra = ""
        if isinstance(auction, LoanCard) and cap > my_state.coins:
            prompt_extra = f" (you may bid up to {cap} since this is a loan)"
        while True:
            raw = input(f"{self.name}, enter your bid 0-{cap}{prompt_extra}: ").strip()
            try:
                bid = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue
            if bid < 0 or bid > cap:
                print(f"Bid must be between 0 and {cap}.")
                continue
            return bid

    def choose_gem_to_reveal(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
    ) -> GemCard:
        from . import render

        print()
        print("You won — reveal one gem from your hand to the Value Display.")
        print(render.render_hand(my_state))
        while True:
            raw = input(f"Pick a gem 1-{len(my_state.hand)}: ").strip()
            try:
                idx = int(raw) - 1
            except ValueError:
                print("Please enter an integer.")
                continue
            if 0 <= idx < len(my_state.hand):
                return my_state.hand[idx]
            print("Out of range.")


# ----------------------------------------------------------------------------
# HeuristicAI: greedy estimator that beats RandomAI most of the time.
# ----------------------------------------------------------------------------


_GEMS_PER_COLOR = 6


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

    # Sum marginal values, accounting for diminishing returns within the same
    # auction (two of the same color increases own count by 2 -> chart bumps).
    extra = Counter()
    gem_value = 0
    for gem in gems_for_sale:
        # Bump the expected display by what we've already taken in this auction
        # to model the effect of putting both into the display indirectly via
        # future hand reveals — actually no, the gems we BUY go to our collection,
        # not the display. So per-gem value uses the unmodified display estimate.
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


class HeuristicAI(Player):
    """Greedy heuristic player. Beats RandomAI in head-to-head simulations."""

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
            value = _treasure_value(auction, public_state, my_state)
            target = int(value * self.DISCOUNT)
            # Floor: keep at least a tiny reserve so we're not broke.
            reserve = self._reserve_for_future(public_state)
            spendable = max(0, my_state.coins - reserve)
            bid = min(target, spendable, cap)
            return max(0, bid)

        if isinstance(auction, InvestCard):
            # Investments return their face value at end-of-game on top of the
            # locked bid — strictly positive cash flow. Bid surplus cash so we
            # don't starve future treasure bids.
            reserve = self._reserve_for_future(public_state)
            surplus = max(0, my_state.coins - reserve)
            # Always sneak in a token bid even if reserve is tight: free coins.
            bid = min(surplus, cap)
            if bid == 0 and cap > 0:
                bid = 1
            return bid

        if isinstance(auction, LoanCard):
            # Loans are net-negative cash flow (you pay back the full face).
            # Only useful as leverage when you have no coins AND there are
            # still meaningful gems to win.
            if my_state.coins >= 5:
                return 0
            if _remaining_supply(public_state) < 3:
                return 0
            return min(auction.amount, cap)

        return 0

    def _reserve_for_future(self, public_state: "GameState") -> int:
        """Coins to keep aside for upcoming treasure auctions."""
        gems_left = _remaining_supply(public_state)
        # Roughly: half the remaining gems are likely worth bidding on.
        future_treasures = max(0, gems_left // 2)
        avg_value = _expected_avg_treasure_value(public_state, public_state.player_states[0])
        # Use a fraction of avg so we don't over-reserve.
        return int(future_treasures * avg_value * 0.2)

    # --- Reveal -----------------------------------------------------------

    def choose_gem_to_reveal(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
    ) -> GemCard:
        chart = public_state.value_chart
        display = public_state.value_display

        # Tally collection holdings per color across all players.
        my_holding = my_state.collection_gems
        opp_holding: Counter = Counter()
        for ps in public_state.player_states:
            if ps is my_state:
                continue
            opp_holding.update(ps.collection_gems)

        # Score each unique color present in our hand.
        best_score: float | None = None
        best_card: GemCard | None = None
        for card in my_state.hand:
            color = card.color
            current = display.get(color, 0)
            delta = value_for(chart, current + 1) - value_for(chart, current)
            relative = my_holding.get(color, 0) - opp_holding.get(color, 0)
            net_benefit = delta * relative
            # Tie-breaker: prefer revealing a color we hold least of.
            tiebreaker = -my_holding.get(color, 0)
            score = (net_benefit, tiebreaker)
            if best_score is None or score > best_score:
                best_score = score
                best_card = card

        return best_card if best_card is not None else my_state.hand[0]


# ----------------------------------------------------------------------------
# AdaptiveHeuristicAI: same value model as HeuristicAI, but the bid-sizing
# discount is a small linear model over five game-state features instead of a
# fixed constant.
# ----------------------------------------------------------------------------


# matches the breakdown in cards.make_auction_deck()
_TOTAL_AUCTIONS = 25


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
        reserve = self._reserve_for_future(public_state)
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


# ----------------------------------------------------------------------------
# HypergeometricAI: probabilistic value-display estimator.
#
# HeuristicAI's `_expected_final_display` collapses the unknown opponent-held
# gems into a single point estimate per color and then evaluates the value
# chart at that point. That throws away the full distribution and is wrong
# for any non-linear chart — most obviously chart E, which peaks at 3 gems
# and then crashes. The fix is to compute the full distribution of
# `final_display[c]` for each color (it's hypergeometric over the hidden
# pool) and take E[chart_value(X)] instead of chart_value(E[X]).
#
# This file deliberately leaves HeuristicAI and AdaptiveHeuristicAI alone:
# the new logic lives in `_hyper_*` helpers and a new `HypergeometricAI`
# class so the upgrade can be evaluated in isolation.
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
        if (
            hidden_total == 0
            or opp_hand_total == 0
            or hidden_of_color == 0
        ):
            # No randomness left for this color: either no opponent slots
            # exist, opponents hold no hidden cards, or none of those cards
            # could possibly be color c. Final count is exactly known_offset.
            dist[known_offset] = 1.0
        else:
            denom = math.comb(hidden_total, opp_hand_total)
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
            reserve = self._reserve_for_future(public_state)
            spendable = max(0, my_state.coins - reserve)
            bid = min(target, spendable, cap)
            return max(0, bid)

        if isinstance(auction, InvestCard):
            # Investments return their face value at end-of-game on top of
            # the locked bid — strictly positive cash flow. Bid surplus cash
            # so we don't starve future treasure bids.
            reserve = self._reserve_for_future(public_state)
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

    def _reserve_for_future(self, public_state: "GameState") -> int:
        """Coins to keep aside for upcoming treasure auctions."""
        gems_left = _remaining_supply(public_state)
        future_treasures = max(0, gems_left // 2)
        avg_value = _hyper_avg_treasure_value(
            public_state, public_state.player_states[0]
        )
        return int(future_treasures * avg_value * 0.2)

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


# ----------------------------------------------------------------------------
# HyperAdaptiveAI: AdaptiveHeuristicAI's linear-model bid sizing fed by the
# HypergeometricAI value estimator. Synthesises both upgrades — the better
# per-gem value estimate AND the state-dependent discount — into a single
# player. Everything else (discount weights, loan thresholds, reveal logic)
# is inherited from AdaptiveHeuristicAI.
# ----------------------------------------------------------------------------


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
