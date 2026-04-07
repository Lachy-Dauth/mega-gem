"""Mission cards for MegaGem.

Each mission has a coin value and a `requirement` predicate that takes a
`Counter[Color]` (a player's collection of gems) and returns True iff the
mission is satisfied.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

from .cards import Color

Requirement = Callable[[Counter], bool]


@dataclass
class MissionCard:
    name: str
    coins: int
    requirement: Requirement
    category: str  # "shield", "pendant", "crown"

    def is_satisfied_by(self, gems: Counter) -> bool:
        return self.requirement(gems)


# --- Requirement builders ---------------------------------------------------


def at_least_n_distinct_colors(n: int) -> Requirement:
    def check(gems: Counter) -> bool:
        return sum(1 for c in gems.values() if c > 0) >= n
    return check


def n_pairs_any_color(n: int) -> Requirement:
    """At least `n` distinct colors each with count >= 2."""
    def check(gems: Counter) -> bool:
        return sum(1 for c in gems.values() if c >= 2) >= n
    return check


def n_of_same_any_color(n: int) -> Requirement:
    def check(gems: Counter) -> bool:
        return any(c >= n for c in gems.values())
    return check


def color_counts_at_least(required: dict[Color, int]) -> Requirement:
    def check(gems: Counter) -> bool:
        return all(gems.get(color, 0) >= count for color, count in required.items())
    return check


# --- Deck factory ----------------------------------------------------------


def make_mission_deck() -> list[MissionCard]:
    """30 mission cards split 2 / 16 / 12 (shields / pendants / crowns)."""
    deck: list[MissionCard] = []

    # Shields (2)
    deck.append(MissionCard(
        name="Shield: 4 different colors",
        coins=10,
        requirement=at_least_n_distinct_colors(4),
        category="shield",
    ))
    deck.append(MissionCard(
        name="Shield: 2 pairs",
        coins=15,
        requirement=n_pairs_any_color(2),
        category="shield",
    ))

    # Pendants (16, all 5 coins)
    deck.append(MissionCard(
        name="Pendant: 2 of the same color",
        coins=5,
        requirement=n_of_same_any_color(2),
        category="pendant",
    ))
    for color in Color:
        deck.append(MissionCard(
            name=f"Pendant: 2 {color}",
            coins=5,
            requirement=color_counts_at_least({color: 2}),
            category="pendant",
        ))
    for c1, c2 in combinations(Color, 2):
        deck.append(MissionCard(
            name=f"Pendant: 1 {c1} + 1 {c2}",
            coins=5,
            requirement=color_counts_at_least({c1: 1, c2: 1}),
            category="pendant",
        ))

    # Crowns (12, all 10 coins)
    deck.append(MissionCard(
        name="Crown: 3 of the same color",
        coins=10,
        requirement=n_of_same_any_color(3),
        category="crown",
    ))
    deck.append(MissionCard(
        name="Crown: 3 different colors",
        coins=10,
        requirement=at_least_n_distinct_colors(3),
        category="crown",
    ))
    for c1, c2, c3 in combinations(Color, 3):
        deck.append(MissionCard(
            name=f"Crown: 1 {c1} + 1 {c2} + 1 {c3}",
            coins=10,
            requirement=color_counts_at_least({c1: 1, c2: 1, c3: 1}),
            category="crown",
        ))

    return deck
