"""Game engine for MegaGem: setup, round resolution, and scoring."""

from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING

from .cards import (
    AuctionCard,
    GemCard,
    InvestCard,
    LoanCard,
    TreasureCard,
    make_auction_deck,
    make_gem_deck,
)
from .missions import MissionCard, make_mission_deck
from .state import GameState, PlayerState
from .value_charts import VALUE_CHARTS, value_for

if TYPE_CHECKING:
    from .players import Player


# Per the rules table on RULES.md.
STARTING_COINS = {3: 35, 4: 25, 5: 20}
STARTING_HAND_SIZE = {3: 5, 4: 4, 5: 3}


# --- Setup ------------------------------------------------------------------


def setup_game(
    players: list["Player"],
    chart: str = "A",
    seed: int | None = None,
) -> GameState:
    n = len(players)
    if n not in STARTING_COINS:
        raise ValueError(f"MegaGem supports 3–5 players, got {n}.")
    if chart not in VALUE_CHARTS:
        raise ValueError(f"Unknown value chart: {chart!r}")

    rng = random.Random(seed)

    gem_deck = make_gem_deck()
    auction_deck = make_auction_deck()
    mission_deck = make_mission_deck()

    rng.shuffle(gem_deck)
    rng.shuffle(auction_deck)
    rng.shuffle(mission_deck)

    coins = STARTING_COINS[n]
    hand_size = STARTING_HAND_SIZE[n]

    player_states: list[PlayerState] = []
    for player in players:
        hand = [gem_deck.pop() for _ in range(hand_size)]
        player_states.append(
            PlayerState(
                name=player.name,
                is_human=getattr(player, "is_human", False),
                hand=hand,
                coins=coins,
            )
        )

    revealed_gems = [gem_deck.pop() for _ in range(2)]
    active_missions = [mission_deck.pop() for _ in range(4)]

    return GameState(
        player_states=player_states,
        players=players,
        value_chart=chart,
        gem_deck=gem_deck,
        revealed_gems=revealed_gems,
        auction_deck=auction_deck,
        active_missions=active_missions,
    )


# --- Bid validation ---------------------------------------------------------


def max_legal_bid(player_state: PlayerState, auction: AuctionCard) -> int:
    """Highest bid this player may legally make on `auction`."""
    if isinstance(auction, LoanCard):
        return max(0, player_state.coins + auction.amount)
    return max(0, player_state.coins)


def clamp_bid(bid: object, player_state: PlayerState, auction: AuctionCard) -> int:
    """Force any value into the legal range. Belt-and-braces against buggy AIs."""
    try:
        b = int(bid)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        b = 0
    if b < 0:
        b = 0
    cap = max_legal_bid(player_state, auction)
    if b > cap:
        b = cap
    return b


# --- Round resolution -------------------------------------------------------


def is_game_over(state: GameState) -> bool:
    return not state.auction_deck or (not state.gem_deck and not state.revealed_gems)


def _resolve_winner(
    bids: list[int],
    last_winner_idx: int | None,
    num_players: int,
    rng: random.Random,
) -> int:
    top = max(bids)
    candidates = [i for i, b in enumerate(bids) if b == top]
    if len(candidates) == 1:
        return candidates[0]
    if last_winner_idx is None:
        return rng.choice(candidates)
    # Closest player to the left of the previous winner.
    # "Left" = next in seating order; we walk forward starting at last_winner+1.
    for offset in range(1, num_players + 1):
        idx = (last_winner_idx + offset) % num_players
        if idx in candidates:
            return idx
    # Unreachable, but keeps the type checker happy.
    return candidates[0]


def _apply_treasure(
    state: GameState, winner_state: PlayerState, card: TreasureCard
) -> list[GemCard]:
    """Move up to `card.gems` revealed gems into the winner's collection."""
    gems_to_take = min(card.gems, len(state.revealed_gems))
    taken: list[GemCard] = []
    for _ in range(gems_to_take):
        gem = state.revealed_gems.pop(0)
        winner_state.collection_gems[gem.color] += 1
        taken.append(gem)
    return taken


def _apply_loan(winner_state: PlayerState, bid: int, card: LoanCard) -> None:
    # Bid is paid to the supply (out of the loan if necessary).
    winner_state.coins -= bid
    winner_state.coins += card.amount
    winner_state.loans.append(card)


def _apply_invest(winner_state: PlayerState, bid: int, card: InvestCard) -> None:
    winner_state.coins -= bid
    winner_state.investments.append((card, bid))


def _replenish_revealed(state: GameState) -> None:
    while len(state.revealed_gems) < 2 and state.gem_deck:
        state.revealed_gems.append(state.gem_deck.pop())


def _check_missions(state: GameState) -> list[tuple[int, MissionCard]]:
    """Auto-complete any missions players qualify for. Returns list of (player_idx, mission)."""
    completed: list[tuple[int, MissionCard]] = []
    # Iterate over a snapshot since we mutate active_missions.
    for mission in list(state.active_missions):
        for idx, ps in enumerate(state.player_states):
            if mission.is_satisfied_by(ps.collection_gems):
                ps.completed_missions.append(mission)
                state.active_missions.remove(mission)
                completed.append((idx, mission))
                break
    return completed


def play_round(state: GameState, rng: random.Random | None = None) -> dict:
    """Run one full round. Returns a small dict describing what happened
    (useful for the CLI to print between rounds)."""
    if rng is None:
        rng = random.Random()
    if not state.auction_deck:
        raise RuntimeError("No auction cards left.")

    state.round_number += 1
    auction = state.auction_deck.pop()

    # Collect bids from every player simultaneously.
    raw_bids: list[int] = []
    for player, ps in zip(state.players, state.player_states):
        raw = player.choose_bid(state, ps, auction)
        raw_bids.append(clamp_bid(raw, ps, auction))

    winner_idx = _resolve_winner(raw_bids, state.last_winner_idx, state.num_players, rng)
    winning_bid = raw_bids[winner_idx]
    winner_state = state.player_states[winner_idx]

    taken_gems: list[GemCard] = []
    if isinstance(auction, TreasureCard):
        # Pay first, then take gems.
        winner_state.coins -= winning_bid
        taken_gems = _apply_treasure(state, winner_state, auction)
    elif isinstance(auction, LoanCard):
        _apply_loan(winner_state, winning_bid, auction)
    elif isinstance(auction, InvestCard):
        _apply_invest(winner_state, winning_bid, auction)
    else:
        raise TypeError(f"Unknown auction card: {auction!r}")

    # Winner reveals one gem from their hand to the value display.
    revealed_gem: GemCard | None = None
    if winner_state.hand:
        winner = state.players[winner_idx]
        choice = winner.choose_gem_to_reveal(state, winner_state)
        if choice not in winner_state.hand:
            choice = rng.choice(winner_state.hand)
        winner_state.hand.remove(choice)
        state.value_display[choice.color] += 1
        revealed_gem = choice

    _replenish_revealed(state)
    completed_missions = _check_missions(state)

    state.last_winner_idx = winner_idx

    return {
        "round": state.round_number,
        "auction": auction,
        "bids": raw_bids,
        "winner_idx": winner_idx,
        "winning_bid": winning_bid,
        "taken_gems": taken_gems,
        "revealed_gem": revealed_gem,
        "completed_missions": completed_missions,
    }


# --- Scoring ----------------------------------------------------------------


def score_game(state: GameState) -> list[dict]:
    """Compute final scoring. Reveals leftover hands first.

    Returns a list of per-player score dicts in seating order, sorted by score
    is the caller's job.
    """
    # Reveal any remaining hand cards.
    for ps in state.player_states:
        for card in ps.hand:
            state.value_display[card.color] += 1
        ps.hand.clear()

    chart = state.value_chart

    results: list[dict] = []
    for ps in state.player_states:
        gem_value = 0
        for color, count in ps.collection_gems.items():
            display_count = state.value_display.get(color, 0)
            gem_value += count * value_for(chart, display_count)

        mission_value = sum(m.coins for m in ps.completed_missions)
        loans_total = sum(loan.amount for loan in ps.loans)
        invest_returns = sum(card.amount + locked for card, locked in ps.investments)

        total = ps.coins + gem_value + mission_value - loans_total + invest_returns

        results.append({
            "name": ps.name,
            "coins": ps.coins,
            "gem_value": gem_value,
            "mission_value": mission_value,
            "loans_total": loans_total,
            "invest_returns": invest_returns,
            "total": total,
        })
    return results
