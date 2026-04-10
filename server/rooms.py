"""Room + slot model.

A ``Room`` is the unit a client joins. It has up to five seats, each
either empty, filled by a connected human, or filled by an AI the
host picked. Once the host starts the game, a ``GameSession`` is
attached to the room and drives the engine until it finishes.

All rooms live in one process-wide ``RoomManager`` keyed by a short
random code (``ABC123``) that we also use as the share URL fragment.
Rooms are kept entirely in memory — no database, no redis — which is
fine for the MVP: a Railway restart wipes in-flight games.
"""

from __future__ import annotations

import asyncio
import logging
import random
import secrets
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from fastapi import WebSocket

if TYPE_CHECKING:
    from .session import GameSession


logger = logging.getLogger("megagem.rooms")


ROOM_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I
ROOM_CODE_LEN = 5
MAX_PLAYERS = 5
MIN_PLAYERS = 3
VALID_CHARTS = ("A", "B", "C", "D", "E")


def _gen_room_code() -> str:
    return "".join(secrets.choice(ROOM_CODE_ALPHABET) for _ in range(ROOM_CODE_LEN))


def _gen_player_id() -> str:
    return secrets.token_urlsafe(12)


@dataclass
class Slot:
    """One seat in a room."""

    index: int
    name: str
    kind: str  # "human" or "ai"
    player_id: str  # stable identity (for humans this is the session cookie)
    ai_kind: Optional[str] = None  # only set when kind == "ai"
    websocket: Optional[WebSocket] = None
    connected: bool = False

    def public_view(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "kind": self.kind,
            "ai_kind": self.ai_kind,
            "connected": self.connected,
        }


@dataclass
class Room:
    code: str
    host_player_id: str
    chart: str = "A"
    seed: Optional[int] = None
    status: str = "lobby"  # "lobby" | "playing" | "done"
    slots: list[Slot] = field(default_factory=list)
    session: "GameSession | None" = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # Reserved so RoomManager can garbage-collect idle rooms.
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # -- Slot management --------------------------------------------------

    def is_host(self, player_id: str) -> bool:
        return player_id == self.host_player_id

    def find_slot_by_player(self, player_id: str) -> Optional[Slot]:
        for slot in self.slots:
            if slot.player_id == player_id:
                return slot
        return None

    def add_human(self, name: str) -> Slot:
        if self.status != "lobby":
            raise RuntimeError("Room is not accepting new players.")
        if len(self.slots) >= MAX_PLAYERS:
            raise RuntimeError("Room is full.")
        slot = Slot(
            index=len(self.slots),
            name=name.strip() or f"Player{len(self.slots) + 1}",
            kind="human",
            player_id=_gen_player_id(),
        )
        self.slots.append(slot)
        self.updated_at = time.time()
        return slot

    def add_ai(self, ai_kind: str, name: str) -> Slot:
        if self.status != "lobby":
            raise RuntimeError("Room is not accepting new players.")
        if len(self.slots) >= MAX_PLAYERS:
            raise RuntimeError("Room is full.")
        slot = Slot(
            index=len(self.slots),
            name=name,
            kind="ai",
            ai_kind=ai_kind,
            player_id=f"ai-{_gen_player_id()}",
        )
        self.slots.append(slot)
        self.updated_at = time.time()
        return slot

    def remove_slot(self, player_id: str) -> bool:
        for i, slot in enumerate(self.slots):
            if slot.player_id == player_id:
                del self.slots[i]
                # Re-index remaining slots so indices stay contiguous.
                for j, s in enumerate(self.slots):
                    s.index = j
                self.updated_at = time.time()
                return True
        return False

    def remove_slot_by_index(self, index: int) -> bool:
        if index < 0 or index >= len(self.slots):
            return False
        del self.slots[index]
        for j, s in enumerate(self.slots):
            s.index = j
        self.updated_at = time.time()
        return True

    # -- Lobby view -------------------------------------------------------

    @property
    def host_slot_index(self) -> int:
        for slot in self.slots:
            if slot.player_id == self.host_player_id:
                return slot.index
        return 0

    def public_view(self) -> dict:
        return {
            "code": self.code,
            "status": self.status,
            "chart": self.chart,
            "seed": self.seed,
            "host_slot_index": self.host_slot_index,
            "min_players": MIN_PLAYERS,
            "max_players": MAX_PLAYERS,
            "slots": [s.public_view() for s in self.slots],
        }

    # -- Broadcasting -----------------------------------------------------

    async def broadcast(self, message: dict) -> None:
        """Send ``message`` to every connected human slot."""
        dead: list[Slot] = []
        for slot in self.slots:
            if slot.kind != "human" or slot.websocket is None:
                continue
            try:
                await slot.websocket.send_json(message)
            except Exception as exc:  # noqa: BLE001
                logger.info("broadcast to %s failed: %s", slot.name, exc)
                dead.append(slot)
        for slot in dead:
            slot.connected = False
            slot.websocket = None

    async def send_to(self, player_idx: int, message: dict) -> None:
        if player_idx < 0 or player_idx >= len(self.slots):
            return
        slot = self.slots[player_idx]
        if slot.kind != "human" or slot.websocket is None:
            return
        try:
            await slot.websocket.send_json(message)
        except Exception as exc:  # noqa: BLE001
            logger.info("send_to %s failed: %s", slot.name, exc)
            slot.connected = False
            slot.websocket = None


class RoomManager:
    """Process-wide registry of active rooms."""

    def __init__(self) -> None:
        self._rooms: dict[str, Room] = {}
        self._lock = asyncio.Lock()

    async def create_room(self, host_name: str, chart: str = "A", seed: Optional[int] = None) -> tuple[Room, Slot]:
        if chart not in VALID_CHARTS:
            raise ValueError(f"Invalid chart {chart!r}")
        async with self._lock:
            for _ in range(10):
                code = _gen_room_code()
                if code not in self._rooms:
                    break
            else:
                raise RuntimeError("Could not allocate a unique room code.")

            # Build room first so we can generate a player_id for the
            # host via the slot's ``add_human`` path and then point
            # ``host_player_id`` at it.
            room = Room(code=code, host_player_id="", chart=chart, seed=seed)
            host_slot = room.add_human(host_name)
            room.host_player_id = host_slot.player_id
            self._rooms[code] = room
            logger.info("room %s created by %s", code, host_name)
            return room, host_slot

    def get(self, code: str) -> Optional[Room]:
        return self._rooms.get(code.upper())

    async def delete(self, code: str) -> None:
        async with self._lock:
            room = self._rooms.pop(code.upper(), None)
        if room is not None and room.session is not None:
            room.session.shutdown()

    def list_public(self) -> list[dict]:
        """Debug endpoint: list every room by code + status."""
        return [
            {"code": r.code, "status": r.status, "players": len(r.slots)}
            for r in self._rooms.values()
        ]


# Single shared instance — FastAPI dependency hooks grab this.
manager = RoomManager()


def default_seed() -> int:
    """Used when the host leaves the seed field blank."""
    return random.randrange(2**31)
