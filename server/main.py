"""FastAPI entry point.

Run locally with:

    uvicorn server.main:app --reload

On Railway, the ``Procfile`` / ``nixpacks.toml`` passes
``$PORT`` via uvicorn. All routes live on this single app and one
process serves both the REST API, the WebSocket, and the static
``web/`` frontend.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .ai_factory import AI_KINDS
from .protocol import serialize_state
from .rooms import MAX_PLAYERS, MIN_PLAYERS, VALID_CHARTS, manager
from .session import GameSession


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("megagem.main")


REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"


app = FastAPI(title="MegaGem Multiplayer", version="0.1.0")

# Permissive CORS for local dev; Railway terminates TLS at the edge
# and the frontend is served from the same origin as the API so in
# production this is a no-op.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CreateRoomRequest(BaseModel):
    host_name: str = Field(..., min_length=1, max_length=24)
    chart: str = Field("A", pattern="^[A-E]$")
    seed: Optional[int] = None


class JoinRoomRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=24)


class AddAIRequest(BaseModel):
    player_id: str
    ai_kind: str
    name: Optional[str] = None


class ConfigureRequest(BaseModel):
    player_id: str
    chart: Optional[str] = Field(None, pattern="^[A-E]$")
    seed: Optional[int] = None


class StartRequest(BaseModel):
    player_id: str


class RemoveSlotRequest(BaseModel):
    player_id: str
    target_player_id: str


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "ai_kinds": list(AI_KINDS)}


@app.get("/api/config")
async def config() -> dict:
    """Static metadata the browser needs on the lobby screen."""
    return {
        "min_players": MIN_PLAYERS,
        "max_players": MAX_PLAYERS,
        "charts": list(VALID_CHARTS),
        "ai_kinds": list(AI_KINDS),
    }


@app.post("/api/rooms")
async def create_room(req: CreateRoomRequest) -> dict:
    try:
        room, host_slot = await manager.create_room(
            host_name=req.host_name, chart=req.chart, seed=req.seed
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "room": room.public_view(),
        "you": {
            "player_id": host_slot.player_id,
            "slot_index": host_slot.index,
            "is_host": True,
        },
    }


@app.get("/api/rooms/{code}")
async def get_room(code: str) -> dict:
    room = manager.get(code)
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    return {"room": room.public_view()}


@app.post("/api/rooms/{code}/join")
async def join_room(code: str, req: JoinRoomRequest) -> dict:
    room = manager.get(code)
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    if room.status != "lobby":
        raise HTTPException(status_code=409, detail="Game already started")
    try:
        slot = room.add_human(req.name)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "room": room.public_view(),
        "you": {
            "player_id": slot.player_id,
            "slot_index": slot.index,
            "is_host": room.is_host(slot.player_id),
        },
    }


@app.post("/api/rooms/{code}/add_ai")
async def add_ai(code: str, req: AddAIRequest) -> dict:
    room = manager.get(code)
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    if not room.is_host(req.player_id):
        raise HTTPException(status_code=403, detail="Only the host can add AI seats")
    if req.ai_kind not in AI_KINDS:
        raise HTTPException(status_code=400, detail=f"Unknown AI kind: {req.ai_kind}")
    ai_display_names = ["Avery", "Blair", "Casey", "Dylan", "Elliot"]
    name = req.name or ai_display_names[len(room.slots) % len(ai_display_names)]
    try:
        slot = room.add_ai(req.ai_kind, name)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    await _broadcast_lobby(room)
    return {"room": room.public_view(), "added": slot.public_view()}


@app.post("/api/rooms/{code}/configure")
async def configure(code: str, req: ConfigureRequest) -> dict:
    room = manager.get(code)
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    if not room.is_host(req.player_id):
        raise HTTPException(status_code=403, detail="Only the host can configure the room")
    if room.status != "lobby":
        raise HTTPException(status_code=409, detail="Cannot reconfigure once the game has started")
    if req.chart is not None:
        room.chart = req.chart
    if req.seed is not None:
        room.seed = req.seed
    await _broadcast_lobby(room)
    return {"room": room.public_view()}


@app.post("/api/rooms/{code}/remove_slot")
async def remove_slot(code: str, req: RemoveSlotRequest) -> dict:
    room = manager.get(code)
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    if not room.is_host(req.player_id):
        raise HTTPException(status_code=403, detail="Only the host can remove seats")
    if room.status != "lobby":
        raise HTTPException(status_code=409, detail="Cannot remove seats mid-game")
    if req.target_player_id == room.host_player_id:
        raise HTTPException(status_code=400, detail="Host cannot remove themselves")
    if not room.remove_slot(req.target_player_id):
        raise HTTPException(status_code=404, detail="Slot not found")
    await _broadcast_lobby(room)
    return {"room": room.public_view()}


@app.post("/api/rooms/{code}/start")
async def start_game(code: str, req: StartRequest) -> dict:
    room = manager.get(code)
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    if not room.is_host(req.player_id):
        raise HTTPException(status_code=403, detail="Only the host can start the game")
    if room.status != "lobby":
        raise HTTPException(status_code=409, detail="Game already started")
    if len(room.slots) < MIN_PLAYERS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_PLAYERS} players ({len(room.slots)} joined)",
        )
    if len(room.slots) > MAX_PLAYERS:
        raise HTTPException(status_code=400, detail="Too many players")

    loop = asyncio.get_running_loop()
    room.session = GameSession(room, loop)
    room.status = "playing"
    room.session.start()

    return {"room": room.public_view()}


async def _broadcast_lobby(room) -> None:
    """Tell every connected human in the room the lobby state has changed."""
    await room.broadcast({"type": "lobby_update", "room": room.public_view()})


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/api/ws/{code}")
async def game_ws(
    websocket: WebSocket,
    code: str,
    player_id: str = Query(...),
) -> None:
    await websocket.accept()
    room = manager.get(code)
    if room is None:
        await websocket.send_json({"type": "error", "message": "Room not found"})
        await websocket.close()
        return
    slot = room.find_slot_by_player(player_id)
    if slot is None or slot.kind != "human":
        await websocket.send_json({"type": "error", "message": "Not a valid player in this room"})
        await websocket.close()
        return

    # Replace any previous socket the player might still have open.
    old_ws = slot.websocket
    slot.websocket = websocket
    slot.connected = True
    if old_ws is not None:
        try:
            await old_ws.close()
        except Exception:  # noqa: BLE001
            pass

    # Send an initial lobby-or-state snapshot so the client doesn't
    # have to poll the REST API after reconnecting mid-game.
    await websocket.send_json({"type": "welcome", "room": room.public_view(), "slot_index": slot.index})
    if room.status == "playing" and room.session is not None and room.session.state is not None:
        # Re-send the current state + the latest round_start so the
        # client can render the board. If the session is currently
        # waiting on *this* player for a bid or a reveal, re-send that
        # request too — otherwise the UI would sit idle after a reload.
        await websocket.send_json({
            "type": "state",
            "state": serialize_state(room.session.state, viewer_idx=slot.index),
        })
        last_round_start = room.session.last_round_start
        if last_round_start is not None:
            await websocket.send_json(last_round_start)
        pending = room.session.pending_request(slot.index)
        if pending is not None:
            await websocket.send_json(pending)

    try:
        while True:
            message = await websocket.receive_json()
            await _handle_ws_message(room, slot, message)
    except WebSocketDisconnect:
        logger.info("player %s disconnected from %s", slot.name, code)
    except Exception:  # noqa: BLE001
        logger.exception("ws handler crashed")
    finally:
        slot.connected = False
        if slot.websocket is websocket:
            slot.websocket = None
        # Fire-and-forget lobby update so other players see the drop.
        if room.status == "lobby":
            await _broadcast_lobby(room)


async def _handle_ws_message(room, slot, message: dict) -> None:
    msg_type = message.get("type")
    if msg_type == "ping":
        await slot.websocket.send_json({"type": "pong"})
        return
    if msg_type == "bid":
        if room.session is None:
            return
        amount = int(message.get("amount", 0))
        room.session.submit_bid(slot.index, amount)
        return
    if msg_type == "reveal":
        if room.session is None:
            return
        color = str(message.get("color", ""))
        room.session.submit_reveal(slot.index, color)
        return
    if msg_type == "chat":
        text = str(message.get("text", "")).strip()[:200]
        if text:
            await room.broadcast({
                "type": "chat",
                "from": slot.name,
                "from_idx": slot.index,
                "text": text,
            })
        return
    logger.info("unknown ws message: %r", msg_type)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------


if WEB_DIR.exists():
    # Mount at root *after* the API routes so ``/api/*`` still resolves.
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))

    @app.get("/room/{code}")
    async def room_page(code: str) -> FileResponse:  # noqa: ARG001
        # The same SPA handles the lobby + game views — the client
        # reads ``location.pathname`` to pick up the room code.
        return FileResponse(str(WEB_DIR / "index.html"))
