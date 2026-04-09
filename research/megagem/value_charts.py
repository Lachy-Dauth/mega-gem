"""Value charts for MegaGem.

Each chart maps the number of gems of a given color in the Value Display
to the coin value of one such gem in a player's Collection. Index 5 is "5+".
"""

from __future__ import annotations

VALUE_CHARTS: dict[str, list[int]] = {
    "A": [0, 4, 8, 12, 16, 20],
    "B": [20, 16, 12, 8, 4, 0],
    "C": [0, 2, 5, 9, 14, 20],
    "D": [20, 18, 15, 11, 6, 0],
    "E": [0, 4, 10, 18, 6, 0],
}


def value_for(chart: str, count: int) -> int:
    """Coin value of one gem when `count` of that color sit in the Value Display."""
    if chart not in VALUE_CHARTS:
        raise ValueError(f"Unknown value chart: {chart!r}")
    table = VALUE_CHARTS[chart]
    idx = min(count, 5)
    if idx < 0:
        idx = 0
    return table[idx]
