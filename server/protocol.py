"""JSON serialization for engine objects.

The canonical engine (``research/megagem``) deals in dataclasses and
``Counter`` objects that aren't directly JSON-friendly. This module
centralises every conversion from engine-land to
wire-format-dicts so that ``server.session`` and the WebSocket
handlers never poke at engine internals directly.

Every function is pure and side-effect-free: pass in engine objects,
get back plain dicts / lists / primitives.
"""

from __future__ import annotations

from typing import Any

from megagem.cards import (
    AuctionCard,
    Color,
    GemCard,
    InvestCard,
    LoanCard,
    TreasureCard,
)
from megagem.missions import MissionCard
from megagem.state import GameState, PlayerState


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def serialize_gem(gem: GemCard) -> dict[str, str]:
    return {"color": gem.color.value}


def serialize_auction(card: AuctionCard) -> dict[str, Any]:
    if isinstance(card, TreasureCard):
        return {"kind": "treasure", "gems": card.gems}
    if isinstance(card, LoanCard):
        return {"kind": "loan", "amount": card.amount}
    if isinstance(card, InvestCard):
        return {"kind": "invest", "amount": card.amount}
    raise TypeError(f"Unknown auction card: {card!r}")


def serialize_mission(mission: MissionCard) -> dict[str, Any]:
    return {
        "name": mission.name,
        "coins": mission.coins,
        "category": mission.category,
    }


def _counter_to_dict(counter) -> dict[str, int]:
    """Normalise a ``Counter[Color]`` to ``{color_string: count}``.

    Engine code keys on ``Color`` enum values; the wire format uses the
    string name so the browser doesn't need to know the enum exists.
    """
    out: dict[str, int] = {}
    for color, count in counter.items():
        if count <= 0:
            continue
        key = color.value if isinstance(color, Color) else str(color)
        out[key] = count
    return out


# ---------------------------------------------------------------------------
# Player + game state
# ---------------------------------------------------------------------------


def serialize_player_state(
    ps: PlayerState,
    *,
    reveal_hand: bool,
) -> dict[str, Any]:
    """Turn one ``PlayerState`` into a dict.

    ``reveal_hand=True`` is used for the player's own seat — their full
    hand is sent. For opponents we only send the hand size, since hands
    are hidden information in MegaGem.
    """
    data: dict[str, Any] = {
        "name": ps.name,
        "is_human": ps.is_human,
        "coins": ps.coins,
        "collection": _counter_to_dict(ps.collection_gems),
        "hand_size": len(ps.hand),
        "completed_missions": [serialize_mission(m) for m in ps.completed_missions],
        "loans": [{"amount": l.amount} for l in ps.loans],
        "investments": [
            {"amount": card.amount, "locked": locked}
            for card, locked in ps.investments
        ],
    }
    if reveal_hand:
        data["hand"] = [serialize_gem(g) for g in ps.hand]
    return data


def serialize_state(
    state: GameState,
    *,
    viewer_idx: int | None,
) -> dict[str, Any]:
    """Serialize the whole game state from one viewer's perspective.

    ``viewer_idx`` is the seat index of the recipient so we can
    selectively reveal their own hand. Pass ``None`` to hide every hand
    (spectator / pre-game view).
    """
    return {
        "round_number": state.round_number,
        "value_chart": state.value_chart,
        "auction_deck_count": len(state.auction_deck),
        "gem_deck_count": len(state.gem_deck),
        "revealed_gems": [serialize_gem(g) for g in state.revealed_gems],
        "value_display": _counter_to_dict(state.value_display),
        "active_missions": [serialize_mission(m) for m in state.active_missions],
        "last_winner_idx": state.last_winner_idx,
        "viewer_idx": viewer_idx,
        "players": [
            serialize_player_state(ps, reveal_hand=(i == viewer_idx))
            for i, ps in enumerate(state.player_states)
        ],
    }


# ---------------------------------------------------------------------------
# Round results
# ---------------------------------------------------------------------------


def serialize_round_result(result: dict) -> dict[str, Any]:
    """Convert the dict ``play_round`` returns into wire format.

    The engine's dict mixes primitive fields (ints, lists of ints) with
    live dataclass references (``auction``, ``taken_gems``, …); here we
    strip every reference to engine objects.
    """
    return {
        "round": result["round"],
        "auction": serialize_auction(result["auction"]),
        "bids": list(result["bids"]),
        "winner_idx": result["winner_idx"],
        "winning_bid": result["winning_bid"],
        "taken_gems": [serialize_gem(g) for g in result["taken_gems"]],
        "revealed_gem": (
            serialize_gem(result["revealed_gem"])
            if result["revealed_gem"] is not None
            else None
        ),
        "completed_missions": [
            {"player_idx": idx, "mission": serialize_mission(m)}
            for idx, m in result["completed_missions"]
        ],
    }


def serialize_scores(scores: list[dict]) -> list[dict[str, Any]]:
    """Pass-through: ``engine.score_game`` already returns plain dicts."""
    return [dict(row) for row in scores]
