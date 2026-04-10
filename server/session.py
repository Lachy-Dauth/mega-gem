"""``GameSession`` — drives the engine in a background thread.

The canonical engine is synchronous: ``play_round`` iterates over
players and calls ``player.choose_bid``/``player.choose_gem_to_reveal``
inline. For multiplayer that means we must:

1. Run the engine off the asyncio event loop so blocking on a human
   bid doesn't freeze FastAPI's whole process.
2. Bridge thread → event loop whenever the game thread wants to emit
   a broadcast to WebSocket clients.

We use a dedicated ``threading.Thread`` per room. Bids arrive via
thread-safe ``queue.Queue`` on ``RemotePlayer``. Broadcasts go the
other way: the game thread calls ``asyncio.run_coroutine_threadsafe``
to schedule ``Room.broadcast`` / ``Room.send_to`` coroutines on the
main event loop, where they actually talk to the WebSocket.

The session also peeks at the top of the auction deck before calling
``play_round`` so it can emit a ``round_start`` broadcast *before*
the engine starts blocking on the first human's bid queue — otherwise
the UI wouldn't know what card to render.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
from typing import TYPE_CHECKING

from megagem.engine import (
    is_game_over,
    max_legal_bid,
    play_round,
    score_game,
    setup_game,
)
from megagem.players.base import Player

from .ai_factory import build_ai
from .db import record_game
from .protocol import (
    serialize_auction,
    serialize_gem,
    serialize_round_result,
    serialize_scores,
    serialize_state,
)
from .remote_player import RemotePlayer

if TYPE_CHECKING:
    from megagem.state import GameState

    from .rooms import Room


logger = logging.getLogger("megagem.session")


def _log_future_error(future: "asyncio.Future") -> None:
    """Callback for fire-and-forget coroutines scheduled from the game thread."""
    try:
        future.result()
    except Exception:
        logger.exception("emit coroutine failed")


class GameSession:
    """One game, one room, one background thread."""

    def __init__(self, room: "Room", loop: asyncio.AbstractEventLoop) -> None:
        self.room = room
        self.loop = loop
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state: "GameState | None" = None
        # RemotePlayer instances by seat index so the WS handler can
        # push bid/reveal values from async code.
        self._remote_players: dict[int, RemotePlayer] = {}
        # Last request message the engine emitted to each human that
        # hasn't been answered yet. On WS reconnect we re-send the
        # pending entry (if any) so a refresh mid-game doesn't leave
        # the player stuck with no prompt.
        self._pending_requests: dict[int, dict] = {}
        # Most recent round_start payload so reconnecting clients can
        # re-render the current auction card even if they missed the
        # original broadcast.
        self._last_round_start: dict | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Session already started.")
        self._thread = threading.Thread(
            target=self._run, name=f"megagem-{self.room.code}", daemon=True
        )
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        for remote in self._remote_players.values():
            remote.shutdown()

    # ------------------------------------------------------------------
    # Called from the async WebSocket handlers
    # ------------------------------------------------------------------

    def notify_disconnect(self, player_idx: int) -> None:
        """Called when a human's WebSocket drops during a game."""
        remote = self._remote_players.get(player_idx)
        if remote is not None:
            remote.forfeit()

    def notify_reconnect(self, player_idx: int) -> None:
        """Called when a human reconnects during a game."""
        remote = self._remote_players.get(player_idx)
        if remote is not None:
            remote.reactivate()

    def submit_bid(self, player_idx: int, amount: int) -> None:
        remote = self._remote_players.get(player_idx)
        if remote is None:
            return
        self._pending_requests.pop(player_idx, None)
        remote.bid_queue.put(int(amount))

    def submit_reveal(self, player_idx: int, color: str) -> None:
        remote = self._remote_players.get(player_idx)
        if remote is None:
            return
        self._pending_requests.pop(player_idx, None)
        remote.reveal_queue.put(color)

    def pending_request(self, player_idx: int) -> dict | None:
        """Return the last unanswered request for this seat, if any."""
        return self._pending_requests.get(player_idx)

    @property
    def last_round_start(self) -> dict | None:
        return self._last_round_start

    @property
    def state(self) -> "GameState | None":
        return self._state

    # ------------------------------------------------------------------
    # Called from the game thread (by RemotePlayer)
    # ------------------------------------------------------------------

    def request_bid(self, player_idx: int) -> None:
        """Notify one human that the engine is now blocking on their bid.

        Includes their legal cap so the client can clamp the input
        field — the engine will also clamp the server-side, but a
        graceful UI is nicer than silent capping.
        """
        assert self._state is not None
        ps = self._state.player_states[player_idx]
        # ``state.auction_deck`` has already had the round's card
        # popped off by the time ``choose_bid`` runs, so we stash the
        # live auction card on the session before ``play_round`` is
        # called (see ``_run``).
        auction = self._current_auction
        cap = max_legal_bid(ps, auction) if auction is not None else ps.coins
        message = {
            "type": "request_bid",
            "max_bid": cap,
            "my_coins": ps.coins,
            "auction": serialize_auction(auction) if auction is not None else None,
        }
        self._pending_requests[player_idx] = message
        self._emit_to(player_idx, message)

    def request_reveal(self, player_idx: int, hand: list) -> None:
        message = {
            "type": "request_reveal",
            "hand": [serialize_gem(g) for g in hand],
        }
        self._pending_requests[player_idx] = message
        self._emit_to(player_idx, message)

    # ------------------------------------------------------------------
    # Thread → event loop bridge
    # ------------------------------------------------------------------

    def _emit(self, message: dict) -> None:
        """Broadcast to every connected human in the room."""
        future = asyncio.run_coroutine_threadsafe(
            self.room.broadcast(message), self.loop
        )
        future.add_done_callback(_log_future_error)

    def _emit_to(self, player_idx: int, message: dict) -> None:
        """Send to one specific seat."""
        future = asyncio.run_coroutine_threadsafe(
            self.room.send_to(player_idx, message), self.loop
        )
        future.add_done_callback(_log_future_error)

    def _emit_state_snapshots(self) -> None:
        """Send each human a personalised state snapshot.

        Each seat needs its own rendering because ``serialize_state``
        only reveals one hand at a time — the viewer's own.
        """
        assert self._state is not None
        for slot in self.room.slots:
            if slot.kind != "human":
                continue
            snapshot = serialize_state(self._state, viewer_idx=slot.index)
            self._emit_to(slot.index, {"type": "state", "state": snapshot})

    # ------------------------------------------------------------------
    # Main game thread
    # ------------------------------------------------------------------

    def _build_players(self) -> list[Player]:
        rng = random.Random(self.room.seed)
        players: list[Player] = []
        num_players = len(self.room.slots)
        for slot in self.room.slots:
            if slot.kind == "human":
                remote = RemotePlayer(slot.name, slot.index)
                remote.session = self
                self._remote_players[slot.index] = remote
                players.append(remote)
            else:
                ai = build_ai(
                    slot.ai_kind or "heuristic",
                    slot.name,
                    seed=rng.randrange(2**31),
                    num_players=num_players,
                )
                players.append(ai)
        return players

    def _run(self) -> None:
        try:
            players = self._build_players()
            self._state = setup_game(
                players, chart=self.room.chart, seed=self.room.seed
            )
            self._current_auction = None

            self._emit({
                "type": "game_start",
                "chart": self.room.chart,
                "num_players": len(players),
            })
            self._emit_state_snapshots()

            rng = random.Random(self.room.seed)

            while not is_game_over(self._state) and not self._stop.is_set():
                # Peek at the next auction card so clients can render it
                # *before* the engine starts blocking on human bids.
                next_auction = self._state.auction_deck[-1]
                self._current_auction = next_auction
                round_start_msg = {
                    "type": "round_start",
                    "round": self._state.round_number + 1,
                    "auction": serialize_auction(next_auction),
                    "revealed_gems": [serialize_gem(g) for g in self._state.revealed_gems],
                }
                self._last_round_start = round_start_msg
                self._emit(round_start_msg)

                try:
                    result = play_round(self._state, rng=rng)
                except Exception:
                    logger.exception("engine crashed mid-round")
                    self._emit({
                        "type": "error",
                        "message": "Engine crashed — see server logs.",
                    })
                    return

                self._current_auction = None
                self._emit({
                    "type": "round_end",
                    "result": serialize_round_result(result),
                })
                self._emit_state_snapshots()

            if self._stop.is_set():
                self._emit({"type": "session_cancelled"})
                return

            scores = score_game(self._state)
            self.room.status = "done"
            self._emit({
                "type": "game_end",
                "scores": serialize_scores(scores),
            })
            self._emit_state_snapshots()

            # Persist for the leaderboards. We do this in the game
            # thread so a slow disk write can't block the event loop;
            # any failure is logged but never crashes the session.
            try:
                slot_records = [
                    {
                        "name": s.name,
                        "kind": s.kind,
                        "ai_kind": s.ai_kind,
                    }
                    for s in self.room.slots
                ]
                record_game(
                    chart=self.room.chart,
                    seed=self.room.seed,
                    slots=slot_records,
                    scores=scores,
                )
            except Exception:
                logger.exception("failed to record game in db")
        except Exception:
            logger.exception("session %s crashed", self.room.code)
            self._emit({
                "type": "error",
                "message": "Session crashed — see server logs.",
            })
