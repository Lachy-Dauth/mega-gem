"""Browser session driver for the static (GitHub Pages) web client.

The web client in ``web/`` runs this exact package inside Pyodide, so
the browser plays the canonical engine rather than a JS port. This
module is the only glue between the two: it owns one game, exposes a
small step-wise API (``start`` / ``bid`` / ``reveal`` / ``step``) and
hands back plain JSON-friendly dicts. Like the CLI, it is a frontend
adapter — the engine itself knows nothing about it.

The engine is synchronous: ``play_round`` calls ``choose_bid`` and
``choose_gem_to_reveal`` inline, and a browser can't block waiting for
a click. Bids are simultaneous, so the human's bid is simply collected
*before* ``play_round`` runs. The reveal is harder — it only happens if
the human wins, mid-round. We handle it by snapshot-and-replay: deep-copy
``(state, rng)`` before the round, and if the human's reveal is needed,
``_BrowserHuman.choose_gem_to_reveal`` raises :class:`_NeedReveal`. Once
the player picks a gem, the snapshot is restored and the round is
replayed with the answer preloaded. Every AI and the tie-break RNG live
inside the snapshot, so the replay is bit-for-bit the same round.

Stdlib only; tested under CPython in ``tests/test_browser.py``.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Callable

from .cards import (
    AuctionCard,
    Color,
    GemCard,
    InvestCard,
    LoanCard,
    TreasureCard,
)
from .engine import is_game_over, max_legal_bid, play_round, score_game, setup_game
from .missions import MissionCard
from .players import (
    Evo2AI,
    Evo3AI,
    Evo4AI,
    HeuristicAI,
    HyperAdaptiveSplitAI,
    Player,
    RandomAI,
    SearchAI,
)
from .state import GameState, PlayerState
from .value_charts import VALUE_CHARTS


DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "saved_best_weights"


# ---------------------------------------------------------------------------
# AI roster
# ---------------------------------------------------------------------------


def _candidate_weight_paths(
    weights_dir: Path, profile_key: str, num_players: int
) -> list[Path]:
    """Uniform lookup chain, mirroring ``scripts.evolve.opponents.candidate_filenames``.

    Unlike the CLI, if nothing exists for this seat count we fall back to
    the 4-player file — every checked-in weights file is 4p, and tuned
    4p weights play 3p/5p far better than the class defaults do.
    """
    tags = (
        "vs_all", "vs_random", "vs_heuristic",
        "vs_evo1", "vs_evo2", "vs_evo3", "vs_evo4",
        "self",
    )
    paths: list[Path] = []
    for n in dict.fromkeys((num_players, 4)):
        paths += [
            weights_dir / f"best_weights_{profile_key}_{tag}_{n}p.json"
            for tag in tags
        ]
        paths.append(weights_dir / f"best_weights_{profile_key}_{n}p.json")
    return paths


def _load_evo_weights(
    weights_dir: Path, profile_key: str, num_players: int
) -> list[float] | None:
    for path in _candidate_weight_paths(weights_dir, profile_key, num_players):
        if path.exists():
            return json.loads(path.read_text())["weights"]
    return None


def _evo_factory(profile_key: str, cls: type) -> Callable[..., Player]:
    def build(name: str, *, seed: int, num_players: int, weights_dir: Path) -> Player:
        weights = _load_evo_weights(weights_dir, profile_key, num_players)
        if weights is None:
            if cls is HyperAdaptiveSplitAI:
                return HyperAdaptiveSplitAI(name, seed=seed)
            return cls(name, seed=seed)
        return cls.from_weights(name, weights, seed=seed)
    return build


def _search_factory(name: str, *, seed: int, num_players: int, weights_dir: Path) -> Player:
    # The search bot's internal models use the same weights as the bots
    # actually seated at the table (same lookup chain, same 4p fallback).
    weights = {
        key: _load_evo_weights(weights_dir, key, num_players)
        for key in ("evo1", "evo2", "evo3", "evo4")
    }
    return SearchAI.standard(name, seed=seed, weights=weights)


# Keys match the CLI's ``AI_FACTORIES`` so docs and the browser share
# vocabulary. Missing weights fall back to class defaults, never an error.
AI_FACTORIES: dict[str, Callable[..., Player]] = {
    "random": lambda name, *, seed, num_players, weights_dir: RandomAI(name, seed=seed),
    "heuristic": lambda name, *, seed, num_players, weights_dir: HeuristicAI(name, seed=seed),
    "evolved": _evo_factory("evo1", HyperAdaptiveSplitAI),
    "evo2": _evo_factory("evo2", Evo2AI),
    "evo3": _evo_factory("evo3", Evo3AI),
    "evo4": _evo_factory("evo4", Evo4AI),
    "search": _search_factory,
}

AI_KINDS = tuple(AI_FACTORIES)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize_gem(gem: GemCard) -> str:
    return gem.color.value


def serialize_auction(card: AuctionCard) -> dict[str, Any]:
    if isinstance(card, TreasureCard):
        return {"kind": "treasure", "gems": card.gems}
    if isinstance(card, LoanCard):
        return {"kind": "loan", "amount": card.amount}
    if isinstance(card, InvestCard):
        return {"kind": "invest", "amount": card.amount}
    raise TypeError(f"Unknown auction card: {card!r}")


def serialize_mission(mission: MissionCard) -> dict[str, Any]:
    return {"name": mission.name, "coins": mission.coins, "category": mission.category}


def _counter_to_dict(counter) -> dict[str, int]:
    return {
        (c.value if isinstance(c, Color) else str(c)): n
        for c, n in counter.items()
        if n > 0
    }


def serialize_player_state(ps: PlayerState, *, reveal_hand: bool) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": ps.name,
        "is_human": ps.is_human,
        "coins": ps.coins,
        "collection": _counter_to_dict(ps.collection_gems),
        "hand_size": len(ps.hand),
        "completed_missions": [serialize_mission(m) for m in ps.completed_missions],
        "loans": [loan.amount for loan in ps.loans],
        "investments": [
            {"amount": card.amount, "locked": locked} for card, locked in ps.investments
        ],
    }
    if reveal_hand:
        data["hand"] = [serialize_gem(g) for g in ps.hand]
    return data


def serialize_state(state: GameState, *, viewer_idx: int | None) -> dict[str, Any]:
    """Whole-table snapshot. ``viewer_idx=None`` is a spectator: every hand shown."""
    return {
        "round_number": state.round_number,
        "value_chart": state.value_chart,
        "chart_values": list(VALUE_CHARTS[state.value_chart]),
        "auction_deck_count": len(state.auction_deck),
        "gem_deck_count": len(state.gem_deck),
        "revealed_gems": [serialize_gem(g) for g in state.revealed_gems],
        "value_display": _counter_to_dict(state.value_display),
        "active_missions": [serialize_mission(m) for m in state.active_missions],
        "last_winner_idx": state.last_winner_idx,
        "players": [
            serialize_player_state(
                ps, reveal_hand=(viewer_idx is None or i == viewer_idx)
            )
            for i, ps in enumerate(state.player_states)
        ],
    }


def serialize_round_result(result: dict) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Human seat
# ---------------------------------------------------------------------------


class _NeedReveal(Exception):
    """Raised mid-``play_round`` when the human won and must pick a gem."""


class _BrowserHuman(Player):
    """Human seat whose decisions are preloaded before ``play_round`` runs."""

    is_human = True

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.pending_bid = 0
        self.pending_reveal: str | None = None

    def choose_bid(self, public_state, my_state, auction) -> int:
        return self.pending_bid

    def choose_gem_to_reveal(self, public_state, my_state) -> GemCard:
        if self.pending_reveal is None:
            raise _NeedReveal()
        for gem in my_state.hand:
            if gem.color.value == self.pending_reveal:
                return gem
        return my_state.hand[0]


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class BrowserSession:
    """One game driven step-by-step from the browser.

    ``seats`` is a list of ``{"kind": "human" | <AI kind>, "name": str}``
    in seating order, with at most one human. With no human the game is
    a spectator match advanced by :meth:`step`.

    Every public method returns a *view* dict::

        {"phase": "bid" | "reveal" | "ready" | "game_over",
         "state": {...}, "auction": {...} | None, "max_bid": int | None,
         "last_round": {...} | None, "rationale": [...] | None,
         "scores": [...] | None, ...}
    """

    def __init__(
        self,
        seats: list[dict],
        *,
        chart: str = "A",
        seed: int | None = None,
        explain: bool = False,
        weights_dir: Path | str | None = None,
    ) -> None:
        if seed is None:
            seed = random.randrange(1_000_000)
        self.seed = int(seed)
        self.chart = chart
        self.explain = explain
        weights_dir = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR

        humans = [i for i, s in enumerate(seats) if s.get("kind") == "human"]
        if len(humans) > 1:
            raise ValueError("At most one human seat is supported.")
        self.human_idx: int | None = humans[0] if humans else None
        self.seat_kinds = [s.get("kind", "heuristic") for s in seats]

        seat_rng = random.Random(self.seed)
        players: list[Player] = []
        for seat in seats:
            kind = seat.get("kind", "heuristic")
            name = str(seat.get("name") or kind)[:24]
            if kind == "human":
                players.append(_BrowserHuman(name))
                continue
            if kind not in AI_FACTORIES:
                raise ValueError(f"Unknown AI kind: {kind!r}")
            players.append(
                AI_FACTORIES[kind](
                    name,
                    seed=seat_rng.randrange(2**31),
                    num_players=len(seats),
                    weights_dir=weights_dir,
                )
            )

        self.state = setup_game(players, chart=chart, seed=self.seed)
        self.rng = random.Random(self.seed)
        self.phase = "new"
        self.history: list[dict] = []
        self.scores: list[dict] | None = None
        self.final_hands: list[list[str]] | None = None
        self._snapshot: tuple[GameState, random.Random] | None = None
        self._rationale: list | None = None

    # -- public API ------------------------------------------------------

    def start(self) -> dict:
        if self.phase != "new":
            raise RuntimeError("Session already started.")
        return self._begin_round(last_round=None)

    def bid(self, amount: int) -> dict:
        if self.phase != "bid":
            raise RuntimeError(f"Not expecting a bid (phase={self.phase}).")
        self._human.pending_bid = int(amount)
        return self._play()

    def reveal(self, color: str) -> dict:
        if self.phase != "reveal" or self._snapshot is None:
            raise RuntimeError(f"Not expecting a reveal (phase={self.phase}).")
        # Restore the pre-round snapshot and replay with the answer preloaded.
        self.state, self.rng = self._snapshot
        self._snapshot = None
        self._human.pending_reveal = str(color)
        result = play_round(self.state, rng=self.rng)
        self._human.pending_reveal = None
        return self._finish_round(result)

    def step(self) -> dict:
        if self.phase != "ready":
            raise RuntimeError(f"Not a spectator turn (phase={self.phase}).")
        return self._play()

    def view(self) -> dict:
        """Re-render the current phase without advancing (no ``last_round``)."""
        return self._view(last_round=None)

    # -- internals -------------------------------------------------------

    @property
    def _human(self) -> _BrowserHuman:
        assert self.human_idx is not None
        human = self.state.players[self.human_idx]
        assert isinstance(human, _BrowserHuman)
        return human

    def _current_auction(self) -> AuctionCard | None:
        if self.phase == "reveal" and self._snapshot is not None:
            # The live state has already popped this round's card.
            return self._snapshot[0].auction_deck[-1]
        if self.phase in ("bid", "ready") and self.state.auction_deck:
            return self.state.auction_deck[-1]
        return None

    def _play(self) -> dict:
        self._rationale = self._explain_round() if self.explain else None
        snapshot = copy.deepcopy((self.state, self.rng))
        try:
            result = play_round(self.state, rng=self.rng)
        except _NeedReveal:
            self._snapshot = snapshot
            self.phase = "reveal"
            return self._view(last_round=None)
        return self._finish_round(result)

    def _explain_round(self) -> list:
        """Each AI's ``explain_bid`` lines, computed on a throwaway copy.

        Running the explainers on a deep copy guarantees they can't
        perturb the real game (RNGs, caches, observation history).
        """
        scratch = copy.deepcopy(self.state)
        auction = scratch.auction_deck[-1]
        out: list = []
        for i, (player, ps) in enumerate(zip(scratch.players, scratch.player_states)):
            if player.is_human:
                out.append(None)
                continue
            try:
                lines = player.explain_bid(scratch, ps, auction)
            except Exception as exc:  # noqa: BLE001 — observability only
                lines = [f"rationale-error: {exc!r}"]
            out.append({"ai_class": type(player).__name__, "lines": list(lines)})
        return out

    def _finish_round(self, result: dict) -> dict:
        serialized = serialize_round_result(result)
        if self._rationale is not None:
            serialized["rationale"] = self._rationale
        self._rationale = None
        self.history.append(serialized)
        return self._begin_round(last_round=serialized)

    def _begin_round(self, last_round: dict | None) -> dict:
        if is_game_over(self.state):
            self.final_hands = [
                [serialize_gem(g) for g in ps.hand] for ps in self.state.player_states
            ]
            self.scores = [dict(row) for row in score_game(self.state)]
            self.phase = "game_over"
        elif self.human_idx is None:
            self.phase = "ready"
        else:
            self.phase = "bid"
        return self._view(last_round=last_round)

    def _view(self, last_round: dict | None) -> dict:
        auction = self._current_auction()
        max_bid = None
        if auction is not None and self.human_idx is not None:
            max_bid = max_legal_bid(self.state.player_states[self.human_idx], auction)
        viewer = None if self.phase == "game_over" else self.human_idx
        return {
            "phase": self.phase,
            "seed": self.seed,
            "human_idx": self.human_idx,
            "seat_kinds": list(self.seat_kinds),
            "state": serialize_state(self.state, viewer_idx=viewer),
            "auction": serialize_auction(auction) if auction is not None else None,
            "max_bid": max_bid,
            "last_round": last_round,
            "scores": self.scores,
            "final_hands": self.final_hands,
        }
