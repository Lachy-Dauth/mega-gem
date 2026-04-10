"""Engine-side ``Player`` adapter for a human connected over WebSocket.

The canonical engine (``research/megagem``) calls ``player.choose_bid``
and ``player.choose_gem_to_reveal`` synchronously from inside
``play_round``. For human seats we want those calls to *block* the
game thread until the real player submits their choice over the
WebSocket. ``RemotePlayer`` wires that up with two thread-safe queues:

- ``bid_queue``  — the WS handler pushes ints onto this when the
  player sends ``{"type": "bid", "amount": ...}``.
- ``reveal_queue`` — same idea for the gem the player picks to
  reveal after winning.

When the engine calls one of the blocking methods, ``RemotePlayer``
asks the owning ``GameSession`` to notify the client that it's their
turn (so the UI unlocks the bid input), then blocks the game thread
on the appropriate queue.

Disconnection handling is deliberately simple for the MVP: if the
session shuts down while we're blocked, it puts a sentinel on every
queue so the game thread wakes up and can exit cleanly.
"""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING

from megagem.cards import AuctionCard, Color, GemCard
from megagem.players.base import Player

if TYPE_CHECKING:
    from megagem.state import GameState, PlayerState

    from .session import GameSession


# Sentinel pushed onto queues when the session is shutting down.
_SHUTDOWN = object()


class RemotePlayer(Player):
    """A ``Player`` whose decisions come from a WebSocket client."""

    is_human = True

    def __init__(self, name: str, player_idx: int) -> None:
        super().__init__(name)
        self.player_idx = player_idx
        self.bid_queue: "queue.Queue[object]" = queue.Queue()
        self.reveal_queue: "queue.Queue[object]" = queue.Queue()
        # Set by GameSession right after construction.
        self.session: "GameSession | None" = None
        self._stopped = False

    # ------------------------------------------------------------------
    # Engine callbacks — run on the game thread.
    # ------------------------------------------------------------------

    def choose_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> int:
        if self.session is not None:
            self.session.request_bid(self.player_idx)
        while True:
            try:
                value = self.bid_queue.get(timeout=5.0)
                break
            except queue.Empty:
                if self._stopped:
                    return 0
                continue
        if value is _SHUTDOWN:
            # Session is dying — return a legal no-op and let the
            # engine unwind. ``clamp_bid`` will coerce this into range.
            return 0
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0

    def choose_gem_to_reveal(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
    ) -> GemCard:
        if not my_state.hand:
            # Shouldn't happen — engine only calls this when hand is
            # non-empty — but be defensive so we never index [0].
            raise RuntimeError("choose_gem_to_reveal called with empty hand")

        if self.session is not None:
            self.session.request_reveal(self.player_idx, my_state.hand)

        while True:
            try:
                value = self.reveal_queue.get(timeout=5.0)
                break
            except queue.Empty:
                if self._stopped:
                    return my_state.hand[0]
                continue
        if value is _SHUTDOWN:
            return my_state.hand[0]

        # Wire format is a color name string; map back to the first
        # gem in hand matching that color. If the client sends garbage
        # we fall through to hand[0] — the engine's own safety net
        # would otherwise swap in a random gem.
        if isinstance(value, str):
            for gem in my_state.hand:
                if gem.color.value == value:
                    return gem
            try:
                color_enum = Color(value)
            except ValueError:
                color_enum = None
            if color_enum is not None:
                for gem in my_state.hand:
                    if gem.color == color_enum:
                        return gem
        return my_state.hand[0]

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Unblock any waiting ``choose_*`` call so the game thread exits."""
        self._stopped = True
        self.bid_queue.put(_SHUTDOWN)
        self.reveal_queue.put(_SHUTDOWN)
