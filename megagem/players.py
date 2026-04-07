"""Player implementations for MegaGem.

The `Player` ABC is the modular AI seam: subclass it to plug in a smarter
strategy. The engine clamps any returned bid to the legal range, so AIs
cannot accidentally produce illegal moves — they will simply be capped.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING

from .cards import AuctionCard, Color, GemCard, InvestCard, LoanCard, TreasureCard
from .engine import max_legal_bid
from .value_charts import value_for

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

