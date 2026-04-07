"""CLI rendering helpers. All functions return strings, so they're easy to test
and front-end-agnostic."""

from __future__ import annotations

from collections import Counter

from .cards import Color
from .state import GameState, PlayerState
from .value_charts import VALUE_CHARTS, value_for


_COLOR_ORDER = list(Color)


def _gem_counter_str(counter: Counter) -> str:
    parts = []
    for color in _COLOR_ORDER:
        n = counter.get(color, 0)
        if n:
            parts.append(f"{color}×{n}")
    return ", ".join(parts) if parts else "(none)"


def render_value_chart(chart: str) -> str:
    table = VALUE_CHARTS[chart]
    header = f"Value Chart {chart}:"
    body_rows = []
    labels = ["0", "1", "2", "3", "4", "5+"]
    body_rows.append("  " + "  ".join(f"{lbl:>3}" for lbl in labels))
    body_rows.append("  " + "  ".join(f"{v:>3}" for v in table))
    return header + "\n" + "\n".join(body_rows)


def render_value_display(state: GameState) -> str:
    lines = ["Value Display:"]
    chart = state.value_chart
    for color in _COLOR_ORDER:
        n = state.value_display.get(color, 0)
        worth = value_for(chart, n)
        lines.append(f"  {str(color):<7} ×{n}   (each worth {worth})")
    return "\n".join(lines)


def render_revealed_gems(state: GameState) -> str:
    total_remaining = len(state.revealed_gems) + len(state.gem_deck)
    if not state.revealed_gems:
        face_up = "(none)"
    else:
        face_up = " | ".join(str(g) for g in state.revealed_gems)
    return (
        f"Gems for sale (face-up): {face_up}\n"
        f"Gems remaining in supply: {total_remaining} "
        f"({len(state.revealed_gems)} face-up + {len(state.gem_deck)} face-down)"
    )


def render_active_missions(state: GameState) -> str:
    if not state.active_missions:
        return "Active missions: (none)"
    lines = ["Active missions:"]
    for m in state.active_missions:
        lines.append(f"  - {m.name} ({m.coins} coins)")
    return "\n".join(lines)


def render_player_summary(ps: PlayerState, debug: bool = False, you: bool = False) -> str:
    tag = " (you)" if you else ""
    lines = [f"{ps.name}{tag}: {ps.coins} coins"]
    lines.append(f"  Collection gems: {_gem_counter_str(ps.collection_gems)}")
    if ps.completed_missions:
        names = ", ".join(m.name for m in ps.completed_missions)
        lines.append(f"  Completed missions: {names}")
    if ps.loans:
        amounts = ", ".join(str(l.amount) for l in ps.loans)
        lines.append(f"  Loans owed: {amounts}")
    if ps.investments:
        invs = ", ".join(f"{c.amount}+{locked}" for c, locked in ps.investments)
        lines.append(f"  Investments (face+locked): {invs}")
    if you or debug:
        if ps.hand:
            hand = ", ".join(str(g) for g in ps.hand)
        else:
            hand = "(empty)"
        lines.append(f"  Hand: {hand}")
    else:
        lines.append(f"  Hand: {len(ps.hand)} card(s) hidden")
    return "\n".join(lines)


def render_board(state: GameState, debug: bool = False) -> str:
    sections = []
    sections.append(f"=== Round {state.round_number} ===")
    sections.append(render_value_chart(state.value_chart))
    sections.append(render_value_display(state))
    sections.append(render_revealed_gems(state))
    sections.append(f"Auction cards remaining (after this one): {len(state.auction_deck)}")
    sections.append(render_active_missions(state))
    sections.append("Players:")
    for ps in state.player_states:
        sections.append(render_player_summary(ps, debug=debug, you=ps.is_human))
    return "\n\n".join(sections)


def render_hand(ps: PlayerState) -> str:
    if not ps.hand:
        return "Your hand: (empty)"
    lines = ["Your hand:"]
    for i, gem in enumerate(ps.hand, start=1):
        lines.append(f"  {i}. {gem}")
    return "\n".join(lines)


def render_round_result(result: dict, state: GameState) -> str:
    """Short summary of what happened in a round."""
    lines = []
    auction = result["auction"]
    bids = result["bids"]
    winner_idx = result["winner_idx"]
    winner = state.player_states[winner_idx]
    bid_strs = ", ".join(
        f"{ps.name}={b}" for ps, b in zip(state.player_states, bids)
    )
    lines.append(f"Auction: {auction}")
    lines.append(f"Bids: {bid_strs}")
    lines.append(f"Winner: {winner.name} for {result['winning_bid']} coins")
    if result["taken_gems"]:
        gems = ", ".join(str(g) for g in result["taken_gems"])
        lines.append(f"  → received {gems}")
    if result["revealed_gem"] is not None:
        lines.append(f"  → revealed {result['revealed_gem']} to Value Display")
    for idx, mission in result["completed_missions"]:
        ps = state.player_states[idx]
        lines.append(f"  → {ps.name} completed mission: {mission.name} (+{mission.coins})")
    return "\n".join(lines)


def render_scores(scores: list[dict]) -> str:
    sorted_scores = sorted(scores, key=lambda s: s["total"], reverse=True)
    lines = ["=== FINAL SCORES ==="]
    for rank, s in enumerate(sorted_scores, start=1):
        lines.append(
            f"{rank}. {s['name']:<12} total={s['total']:>4}  "
            f"(coins {s['coins']} + gems {s['gem_value']} + missions {s['mission_value']} "
            f"- loans {s['loans_total']} + invest {s['invest_returns']})"
        )
    winner = sorted_scores[0]
    lines.append("")
    lines.append(f"Winner: {winner['name']} with {winner['total']} coins!")
    return "\n".join(lines)
