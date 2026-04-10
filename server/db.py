"""SQLite-backed game record store.

We persist every finished multiplayer game so we can compute bot
leaderboards. The schema is intentionally tiny: one row per game and
one row per seat. Aggregations (win rates per AI kind per player
count) are pure SQL — no in-process bookkeeping.

The DB lives at ``MEGAGEM_DB_PATH`` if set, otherwise at
``data/megagem.db`` next to the repo root. **On Railway you almost
certainly want to point this at a mounted volume**, otherwise the
file gets wiped on every redeploy.

All public functions are thread-safe: writes happen on the game
thread (via ``GameSession``) and reads happen on the asyncio loop
(REST handlers). We open a fresh connection per call rather than
sharing one across threads, and serialize writes with a module-level
lock so two finishing games can't race.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("megagem.db")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "megagem.db"
DB_PATH = Path(os.environ.get("MEGAGEM_DB_PATH", str(DEFAULT_DB_PATH)))

# Single global lock guards every connection. SQLite handles concurrent
# *readers* fine but a write lock blocks everyone else, so we just keep
# things simple and gate all access here. Game volume is tiny.
_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # WAL gives us decent concurrency for the rare case of a leaderboard
    # GET landing while a game is being recorded.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables + indexes if they don't already exist."""
    with _lock, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                finished_at TEXT NOT NULL,
                chart TEXT NOT NULL,
                seed INTEGER,
                num_players INTEGER NOT NULL,
                num_humans INTEGER NOT NULL,
                num_ais INTEGER NOT NULL,
                winner_idx INTEGER NOT NULL,
                winner_name TEXT NOT NULL,
                winner_kind TEXT NOT NULL,
                winner_ai_kind TEXT,
                winner_score INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS game_players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                slot_idx INTEGER NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                ai_kind TEXT,
                total_score INTEGER NOT NULL,
                coins INTEGER NOT NULL,
                gem_value INTEGER NOT NULL,
                mission_value INTEGER NOT NULL,
                invest_returns INTEGER NOT NULL,
                loans_total INTEGER NOT NULL,
                is_winner INTEGER NOT NULL,
                FOREIGN KEY (game_id) REFERENCES games(id)
            );

            CREATE INDEX IF NOT EXISTS idx_game_players_game
                ON game_players(game_id);
            CREATE INDEX IF NOT EXISTS idx_game_players_ai_kind
                ON game_players(ai_kind);
            CREATE INDEX IF NOT EXISTS idx_games_num_players
                ON games(num_players);
            CREATE INDEX IF NOT EXISTS idx_games_num_humans
                ON games(num_humans);
            """
        )
    logger.info("db ready at %s", DB_PATH)


def record_game(
    *,
    chart: str,
    seed: int | None,
    slots: list[dict[str, Any]],
    scores: list[dict[str, Any]],
) -> int:
    """Insert a finished-game record. Returns the new game id.

    ``slots`` is the room's seat list aligned with ``scores``. Each
    slot dict needs ``name``, ``kind`` ("human"/"ai"), and (for AIs)
    ``ai_kind``. ``scores`` is whatever ``engine.score_game`` returned.

    Winner is the seat with the highest ``total`` (ties broken by
    lowest slot index — same convention engine uses internally).
    """
    if not slots:
        raise ValueError("Cannot record a game with zero slots")
    if len(slots) != len(scores):
        raise ValueError(
            f"slots/scores length mismatch: {len(slots)} vs {len(scores)}"
        )

    winner_idx = max(range(len(scores)), key=lambda i: scores[i]["total"])
    winner_slot = slots[winner_idx]
    winner_score = int(scores[winner_idx]["total"])
    num_humans = sum(1 for s in slots if s["kind"] == "human")
    num_ais = sum(1 for s in slots if s["kind"] == "ai")

    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with _lock, _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO games (
                finished_at, chart, seed, num_players, num_humans, num_ais,
                winner_idx, winner_name, winner_kind, winner_ai_kind, winner_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                finished_at,
                chart,
                seed,
                len(slots),
                num_humans,
                num_ais,
                winner_idx,
                winner_slot["name"],
                winner_slot["kind"],
                winner_slot.get("ai_kind"),
                winner_score,
            ),
        )
        game_id = cur.lastrowid
        for i, (slot, score) in enumerate(zip(slots, scores)):
            conn.execute(
                """
                INSERT INTO game_players (
                    game_id, slot_idx, name, kind, ai_kind,
                    total_score, coins, gem_value, mission_value,
                    invest_returns, loans_total, is_winner
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    i,
                    slot["name"],
                    slot["kind"],
                    slot.get("ai_kind"),
                    int(score["total"]),
                    int(score["coins"]),
                    int(score["gem_value"]),
                    int(score["mission_value"]),
                    int(score["invest_returns"]),
                    int(score["loans_total"]),
                    1 if i == winner_idx else 0,
                ),
            )
        conn.commit()
    logger.info(
        "recorded game id=%s n=%s humans=%s winner=%s (%s)",
        game_id, len(slots), num_humans, winner_slot["name"], winner_slot["kind"],
    )
    return int(game_id)


def get_leaderboards(
    player_counts: tuple[int, ...] = (3, 4, 5),
) -> dict[int, list[dict[str, Any]]]:
    """Return win-rate leaderboards for each requested player count.

    Filters to games with **exactly one human seat** so we get clean
    "AI bot vs single human opponent" stats — that's what the
    leaderboard's "vs one opponent" framing actually means.

    Each row: ``{ai_kind, games_played, wins, win_rate, avg_score}``.
    Sorted by win_rate desc, then games_played desc.
    """
    out: dict[int, list[dict[str, Any]]] = {}
    with _lock, _connect() as conn:
        for n in player_counts:
            rows = conn.execute(
                """
                SELECT
                    gp.ai_kind AS ai_kind,
                    COUNT(*) AS games_played,
                    SUM(gp.is_winner) AS wins,
                    AVG(gp.total_score) AS avg_score
                FROM game_players gp
                JOIN games g ON g.id = gp.game_id
                WHERE g.num_players = ?
                  AND g.num_humans = 1
                  AND gp.kind = 'ai'
                  AND gp.ai_kind IS NOT NULL
                GROUP BY gp.ai_kind
                HAVING COUNT(*) > 0
                ORDER BY (CAST(SUM(gp.is_winner) AS REAL) / COUNT(*)) DESC,
                         games_played DESC
                """,
                (n,),
            ).fetchall()
            out[n] = [
                {
                    "ai_kind": r["ai_kind"],
                    "games_played": int(r["games_played"]),
                    "wins": int(r["wins"]),
                    "win_rate": (
                        float(r["wins"]) / float(r["games_played"])
                        if r["games_played"] else 0.0
                    ),
                    "avg_score": float(r["avg_score"]) if r["avg_score"] is not None else 0.0,
                }
                for r in rows
            ]
    return out


def stats() -> dict[str, Any]:
    """Quick summary used by the health endpoint + UI footer."""
    with _lock, _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        with_humans = conn.execute(
            "SELECT COUNT(*) FROM games WHERE num_humans >= 1"
        ).fetchone()[0]
    return {
        "total_games": int(total),
        "games_with_humans": int(with_humans),
        "db_path": str(DB_PATH),
    }


# Initialise on import so the very first /api/health hit doesn't race
# with table creation. Failure here is fatal — better to fail boot than
# crash silently when the first game tries to record.
init_db()
