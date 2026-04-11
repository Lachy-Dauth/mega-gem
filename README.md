# mega-gem

A pure-Python implementation of the **MegaGem** auction-and-collection card
game, plus a small zoo of bidder AIs, a genetic-algorithm tuner that
breeds the strongest one, and a FastAPI + WebSocket server that lets you
play against the AI zoo or other humans from a browser.

The full game rules — turn structure, bid ties, mission categories, value
charts — live in [`research/RULES.md`](research/RULES.md). This README
covers everything *around* the rules: how to run the game, what each AI
does, how to test them, how to evolve a stronger one, and how to read
the benchmark plots.

> **Repo layout note.** The canonical engine, AI zoo, tests, GA
> tuners, and checked-in weights live under **`research/`**. The
> **`server/`** + **`web/`** browser stack lives at the repo root.
> **Every Python command in this README assumes you are inside
> `research/` *unless* it explicitly runs the server** — the research
> scripts resolve paths like `saved_best_weights/` and `artifacts/`
> relative to the current working directory; the server resolves them
> via an absolute `Path(__file__)` lookup and can run from the repo
> root.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Repo layout](#repo-layout)
3. [Playing in the terminal](#playing-in-the-terminal)
4. [Playing in the browser (FastAPI server)](#playing-in-the-browser-fastapi-server)
5. [The game engine](#the-game-engine)
6. [The AI zoo](#the-ai-zoo)
7. [Testing](#testing)
8. [Evolving a better AI (the GA)](#evolving-a-better-ai-the-ga)
9. [Benchmarking: pairwise heatmap](#benchmarking-pairwise-heatmap)
10. [Adding your own AI](#adding-your-own-ai)
11. [Common gotchas](#common-gotchas)

---

## Quick start

```bash
# 1. Clone and enter the repo
git clone <this repo>
cd mega-gem

# 2. (Recommended) virtualenv
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows

# 3. The Python game + tests need no third-party deps.
#    matplotlib is only required for the GA + heatmap scripts in research/scripts/.
pip install matplotlib              # only if you want plots

# 4. All Python commands live under research/ — switch there.
cd research

# 5. Run the test suite (should print "Ran 112 tests ... OK")
python -m unittest discover

# 6. Play a game against three Heuristic AIs in the terminal
python -m megagem

# 7. Watch four AIs play each other end-to-end on chart E
python -m megagem --all-ai --ai adaptive --chart E --seed 42

# 8. Or play in the browser (from the repo root):
cd ..
pip install -r requirements.txt
uvicorn server.main:app --reload    # → http://127.0.0.1:8000/
```

If everything above worked you have a clean install.

---

## Repo layout

```
mega-gem/
├── server/                  # FastAPI server (imports research/megagem)
│   ├── main.py                 # App + REST routes + WebSocket endpoint
│   ├── rooms.py                # Room / Slot / RoomManager
│   ├── session.py              # GameSession — runs the engine in a thread
│   ├── remote_player.py        # Player adapter whose decisions come over WS
│   ├── ai_factory.py           # Mirrors research/megagem/__main__.py's AI_FACTORIES
│   └── protocol.py             # JSON serialization for engine objects
├── web/                     # Browser client (served by server/)
│   ├── index.html              # Menu → lobby / quick play → game → scores
│   ├── style.css               # Minimal dark theme
│   └── app.js                  # REST + WebSocket client, DOM rendering
├── requirements.txt         # Server deps (fastapi, uvicorn, pydantic)
├── nixpacks.toml            # Railway build config
├── Procfile                 # Railway start command
├── railway.json             # Railway service config (healthcheck, restart)
├── README.md
├── CLAUDE.md                # Cheat-sheet for Claude Code
└── research/                # Everything Python — run commands from here
    ├── megagem/                 # Game engine + AIs (the bit you import)
    │   ├── __main__.py          # Terminal CLI entry point: `python -m megagem`
    │   ├── cards.py             # Gem / Auction / Treasure / Loan / Invest cards
    │   ├── engine.py            # setup_game, play_round, score_game
    │   ├── explain.py           # --debug rationale-printing wrapper
    │   ├── missions.py          # Mission deck (shields / pendants / crowns)
    │   ├── players/             # Player ABC + every AI (one module per class)
    │   │   ├── base.py              # Player ABC
    │   │   ├── helpers.py           # Shared value / discount-feature math
    │   │   ├── random_ai.py         # RandomAI
    │   │   ├── human.py             # HumanPlayer
    │   │   ├── heuristic.py         # HeuristicAI
    │   │   ├── hyper_adaptive_split.py  # HyperAdaptiveSplitAI (GA target)
    │   │   ├── evo2.py              # Evo2AI (GA target)
    │   │   ├── evo3.py              # Evo3AI (GA target)
    │   │   └── evo4.py              # Evo4AI (GA target, current champion)
    │   ├── render.py            # Pretty-printing for the CLI
    │   ├── state.py             # GameState / PlayerState dataclasses
    │   └── value_charts.py      # The five value charts A–E
    ├── scripts/                 # Standalone runnables (NOT imported by megagem/)
    │   ├── evolve/                   # Unified GA tuner for evo1..evo4
    │   │   ├── __main__.py           # `python -m scripts.evolve --ai evoN`
    │   │   ├── profiles.py           # Per-AI registry (ai_class, num_weights, mutation σ)
    │   │   ├── opponents.py          # 8-mode opponent providers + shared lookup chain
    │   │   └── ga.py                 # GA loop, evaluation, plot/json output
    │   └── heatmap_pairwise.py        # All-vs-all win-rate matrix plot
    ├── tests/                   # unittest suite (112 tests, stdlib only)
    │   ├── test_cards.py
    │   ├── test_engine.py
    │   ├── test_heuristic.py    # Big file — covers every AI's helper math
    │   ├── test_evo2.py         # Evo2AI-specific helpers + head-to-head
    │   ├── test_evo3.py         # Evo3AI-specific helpers + head-to-head
    │   ├── test_evo4.py         # Evo4AI-specific helpers + head-to-head
    │   ├── test_missions.py
    │   └── test_scoring.py
    ├── saved_best_weights/      # Checked-in GA outputs — the CLI reads these
    │   ├── best_weights_evo1_vs_heuristic_4p.json  # HyperAdaptiveSplitAI
    │   ├── best_weights_evo2_vs_evo1_4p.json       # Evo2AI
    │   └── best_weights_evo3_vs_all_4p.json        # Evo3AI
    ├── artifacts/               # Transient GA output (gitignored)
    └── RULES.md                 # The actual game rules — read this first
```

The Python CLI, engine, and tests have **zero** third-party dependencies
— stdlib only. matplotlib is needed only for the GA + heatmap scripts in
`research/scripts/`. The server in `server/` depends on FastAPI + uvicorn
+ pydantic (see `requirements.txt`). The browser client in `web/` has no
dependencies (vanilla JS).

---

## Playing in the terminal

The CLI lives at `python -m megagem` (see `megagem/__main__.py`).

### Common invocations

```bash
# You vs three Heuristic AIs on chart A (default).
python -m megagem

# You vs four Adaptive AIs (5-player game) on chart C, reproducible seed.
python -m megagem --players 5 --ai adaptive --chart C --seed 7

# Smoke-test: four random AIs play each other silently.
python -m megagem --all-ai --ai random --quiet

# Watch four heuristic AIs play out a full game with full board logs.
python -m megagem --all-ai --ai heuristic --chart D --seed 1

# Play against the current champion (GA-evolved Evo4).
python -m megagem --ai evo4

# Play against any evolved AI with debug mode: opponent hands AND
# AI rationale (features, per-head discounts, value estimates) printed
# after every round.
python -m megagem --ai evo4 --debug
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--players {3,4,5}` | `4` | Total seats. One is human unless `--all-ai`. |
| `--chart {A,B,C,D,E}` | `A` | Which value chart to score against. |
| `--seed N` | random | Reproducibility — same seed → same deck order. |
| `--ai {random,heuristic,adaptive,hyper,hyper_adapt,evolved,evo2,evo3,evo4}` | `heuristic` | Opponent AI class. `evolved`, `evo2`, `evo3`, and `evo4` load GA-tuned weights from `saved_best_weights/`. |
| `--all-ai` | off | Replace the human seat with another AI. |
| `--debug` | off | Reveal every player's hand AND print each AI's rationale (features, discount per head, reserve, value estimate) after every round. |
| `--quiet` | off | Suppress per-round logs (fast headless runs). |

### How a turn is rendered

`megagem/render.py` formats each round's events. You'll see:

* The auction card on offer (treasure / loan / invest).
* The reserved bids and the winner.
* What gem cards moved (taken from the display, revealed from the
  winner's hand).
* Any missions that auto-completed.

At the end of the game, `render.render_scores` prints a per-player
breakdown: cash, gem value (chart-dependent!), mission bonuses, loan
deductions, and grand totals.

---

## Playing in the browser (FastAPI server)

The `server/` directory is a FastAPI + WebSocket app that hosts all
browser-based play. It reuses the canonical Python engine and AI zoo
directly (`server/__init__.py` prepends `research/` to `sys.path`), so
any AI you add under `research/megagem/players/` is immediately
available as an opponent.

### Run it locally

```bash
# From the repo root — NOT from inside research/.
pip install -r requirements.txt
uvicorn server.main:app --reload
# → open http://127.0.0.1:8000/
```

The menu offers three ways to play:

- **Quick play** — pick your name, player count, AI difficulty, chart,
  and optional seed. Hits `POST /api/rooms/quick_play` which creates a
  room pre-filled with bots and starts the game immediately. No lobby
  step.
- **Create room** — creates a multiplayer lobby with a 5-character share
  code. The host can add AI seats, invite friends, configure the chart /
  seed, and start when ready.
- **Join existing** — enter a friend's room code to claim a seat.

### Architecture

```
Browser ──HTTP──▶  /api/rooms (create/join/add_ai/start)
         ──WS────▶  /api/ws/{code}?player_id=...
                         │
                         ▼
                 ┌─────────────────┐
                 │  server.main    │  FastAPI app (async)
                 │  ├─ rooms.py    │  in-memory RoomManager
                 │  ├─ session.py  │  one background thread per room
                 │  │    ↓         │
                 │  │  engine.play_round() ← synchronous
                 │  │    ↑         │
                 │  └─ remote_player.py  ← queue.Queue blocks the
                 │                         game thread until a human
                 │                         WS message arrives
                 └─────────────────┘
```

The single most important design decision: the canonical engine is
**synchronous** (`play_round` iterates players and calls
`player.choose_bid` inline), so the multiplayer server drives it in a
background `threading.Thread` per room and bridges the two worlds with
thread-safe queues:

- **WS → game thread**: a `queue.Queue` on each `RemotePlayer`.
  `choose_bid` blocks on `queue.get()`; the async WS handler does
  `queue.put(amount)` when a `{"type": "bid"}` message arrives.
- **Game thread → WS**: `asyncio.run_coroutine_threadsafe(room.broadcast(...), loop)`
  schedules the outbound broadcast on the main event loop.

The session also tracks "pending requests" per seat so a mid-game page
refresh re-receives the last `request_bid` / `request_reveal` on WS
reconnect — otherwise a reload would leave the player staring at a
frozen board while the game thread is still blocked on their queue.

### Protocol

Every WS message is JSON with a `"type"` field:

| Direction | Type | Meaning |
|-----------|------|---------|
| S → C | `welcome` | Initial snapshot with your seat index. |
| S → C | `lobby_update` | Room composition changed. |
| S → C | `game_start` | Game kicked off. |
| S → C | `state` | Personalised state snapshot (your own hand is revealed; opponents are hidden-hand). |
| S → C | `round_start` | New auction card on offer. |
| S → C | `request_bid` | It's your turn to bid. Includes `max_bid` and the current auction. |
| S → C | `request_reveal` | You won the round — pick a gem from your hand to reveal. |
| S → C | `round_end` | Bids, winner, taken gems, completed missions. |
| S → C | `game_end` | Final scores. |
| S → C | `chat` | Player chat (broadcast). |
| S → C | `error` | Something went wrong. |
| C → S | `bid` | `{amount: int}` — bid in response to `request_bid`. |
| C → S | `reveal` | `{color: "Blue" \| …}` — gem to reveal from your hand. |
| C → S | `chat` | `{text: str}` |
| C → S | `ping` | Heartbeat (server replies `pong`). |

### REST routes

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/api/health` | Liveness probe — also used by Railway. |
| `GET`  | `/api/config` | Min/max players, valid charts, AI kinds. |
| `POST` | `/api/rooms` | Create a room; returns `{room, you}`. |
| `POST` | `/api/rooms/quick_play` | Create a room pre-filled with AI bots and start immediately. |
| `GET`  | `/api/rooms/{code}` | Fetch current lobby state. |
| `POST` | `/api/rooms/{code}/join` | Claim a human seat. |
| `POST` | `/api/rooms/{code}/add_ai` | (host) Add an AI seat with a given kind. |
| `POST` | `/api/rooms/{code}/configure` | (host) Change chart / seed in lobby. |
| `POST` | `/api/rooms/{code}/remove_slot` | (host) Kick a seat. |
| `POST` | `/api/rooms/{code}/start` | (host) Spin up the `GameSession` thread. |
| `GET`  | `/api/leaderboard` | Bot win-rate leaderboards (3p / 4p / 5p) over recorded games. |

### Game-record database

Every finished multiplayer game is persisted to a SQLite file via
`server/db.py`. The schema is two tables (`games` + `game_players`)
and the leaderboard query is a single GROUP BY filtered to games with
exactly one human seat — that's the "vs one opponent" framing for the
three leaderboards (3p / 4p / 5p).

The file path is configurable:

```bash
# Default — relative to repo root
data/megagem.db

# Override (use this on Railway with a mounted volume)
export MEGAGEM_DB_PATH=/data/megagem.db
```

> **Railway gotcha.** Railway's container filesystem is ephemeral, so
> the default `data/megagem.db` gets wiped on every redeploy. To keep
> stats across deploys, attach a Railway Volume and set
> `MEGAGEM_DB_PATH` to a path inside it.

### Deploying to Railway

The repo ships with both `nixpacks.toml` and a `Procfile`, so Railway's
Nixpacks builder will pick it up without any manual config:

1. Push this repo to GitHub.
2. Create a new Railway project → **Deploy from GitHub** → pick the repo.
3. Railway reads `requirements.txt`, runs `pip install`, and starts
   `uvicorn server.main:app --host 0.0.0.0 --port $PORT`.
4. The healthcheck at `/api/health` (configured in `railway.json`)
   gates the deploy.

Active room state is in-memory only — a redeploy wipes in-flight
rooms. That's fine for the MVP; swap in Redis or Postgres for
persistent rooms when it matters. The **finished-game record store**
(`server/db.py`) is a SQLite file at `MEGAGEM_DB_PATH` and *is*
persisted across deploys provided you point it at a Railway volume
mount (see [Game-record database](#game-record-database) above).

### Limitations / next steps

- **No auth.** Players enter a display name and get a random
  `player_id` back. Anyone with the `player_id` can act as that seat.
  Good enough for playing with friends over a shared link; not good
  enough for ranked matches.
- **Rooms are not persisted.** An in-memory `RoomManager`. A Railway
  restart drops every active game. Fine for now; swap in Redis when it
  matters. (The game record DB *is* persisted — see above.)
- **Single-process.** One uvicorn worker. Scaling horizontally requires
  moving room state out of process memory first.
- **No spectators.** A WS connection is tied to a seat. Adding
  `spectator` connections that just receive state snapshots is a small
  follow-up.
- **Frontend is intentionally minimal.** The UI in `web/` is functional
  but plain. Improvements to the board rendering are a welcome follow-up.

---

## The game engine

Three short, pure files do all the work:

* **`megagem/state.py`** — `PlayerState` and `GameState` dataclasses. No
  behaviour, just data. Anything that mutates state goes through the
  engine.
* **`megagem/cards.py`** — gem / auction card factories
  (`make_gem_deck`, `make_auction_deck`).
* **`megagem/engine.py`** — the engine itself.

### Key entry points

| Function | Purpose |
|----------|---------|
| `setup_game(players, chart, seed)` | Build a fresh `GameState` for a given list of `Player`s. Deals starting hands, lays out two revealed gems, draws four mission cards. |
| `play_round(state, rng=None)` | Run one auction round: collect bids, resolve the winner (with tie-break), apply the card, let the winner reveal a gem from hand, replenish, auto-complete missions. Returns a small dict that the CLI uses to print. |
| `is_game_over(state)` | True once the auction deck is empty *or* both the gem deck and the revealed gem display are exhausted. |
| `score_game(state)` | Reveals leftover hands, computes per-player totals using the active value chart, returns a list of dicts. |
| `max_legal_bid(player_state, auction)` | The biggest bid this player can legally place — accounts for loans being a "bid out of borrowed coins" loophole. |
| `clamp_bid(bid, player_state, auction)` | Belt-and-braces: forces any AI's return value into legal range so a buggy bot can't crash the game. |

### Value charts

`megagem/value_charts.py` defines five charts (A–E). Each chart maps a
gem's *display count* (how many of that color got revealed during the
game) to its end-of-game coin value.

Charts A–D are roughly monotonic — more revealed = (usually) more
valuable. Chart **E** is non-monotonic and peaks at 3 gems revealed,
which is what makes it the interesting battleground for the
hypergeometric AIs (see below).

`value_for(chart, count)` is the function every scorer uses.

---

## The AI zoo

All AIs live under `megagem/players/` (one module per class) and
implement the abstract base class `Player`:

```python
class Player(ABC):
    name: str
    is_human: bool

    def choose_bid(self, public_state, my_state, auction) -> int: ...
    def choose_gem_to_reveal(self, public_state, my_state) -> GemCard: ...
    def observe_round(self, public_state, my_idx, result) -> None: ...
```

`choose_bid` returns the player's intended bid for the current auction
card; `choose_gem_to_reveal` is called only on the winner each round to
pick which gem from their hand goes into the public value display;
`observe_round` is an optional post-round hook that lets an AI update
internal state (Evo3 uses it to log opponent-bid deltas — see §9 —
and Evo4 uses it to update its per-color gem-inference signal — §10).

Shared value-estimation math (`_expected_final_display`,
`_treasure_value`, the `_hyper_*` family, `_compute_discount_features`,
`_expected_rounds_remaining`, …) lives in `megagem/players/helpers.py`
so every AI can reuse the same well-tested primitives.

The AIs, in roughly increasing order of strength:

### 1. `RandomAI` — `megagem/players/random_ai.py`

Picks bids uniformly between 0 and the legal cap. Reveals a uniformly
random gem from hand. The baseline floor.

### 2. `HumanPlayer` — `megagem/players/human.py`

Interactive console UI. Prints the board (yours or `--debug`-revealed
opponents'), prompts for a bid, prompts for the gem to reveal.

### 3. `HeuristicAI` — `megagem/players/heuristic.py`

The "vanilla strong" baseline. For each auction:

* **Treasures:** estimates the *expected final display count* for every
  color using a uniform-share model over hidden cards (`_expected_final_display`),
  computes per-gem chart value, adds a marginal-gem bonus and a
  mission-completion bonus, then bids `0.75 ×` that estimate.
* **Loans:** takes them only when cash-poor and only when the headline
  rate is small.
* **Invests:** always takes a token bid (free money).
* **Reveal phase:** picks the gem whose color is most likely to *help
  others* and dumps it into the display (giveaway logic).

This is the AI that the GA's fitness function competes against — beat
this and you've actually built something.

### 4. `HyperAdaptiveSplitAI(HeuristicAI)` — `megagem/players/hyper_adaptive_split.py`

The pre-Evo2 champion. The single shared discount was forced to be the
right answer for treasures, invests, *and* loans simultaneously, which
is a contradiction (loans are negative cash flow; invests are free
money). This class splits the bidder into **three independent linear
heads**:

```
treasure_model: _BidModel(bias, w_progress, w_my_cash, w_avg_cash, w_top_cash, w_variance)
invest_model:   _BidModel(...)
loan_model:     _BidModel(...)
```

Each head has 1 bias + 5 weights = 6 constants, for **18 tunable
constants total**. The class ships with sensible defaults
(`DEFAULT_TREASURE`, `DEFAULT_INVEST`, `DEFAULT_LOAN`) but the real
power comes from the GA-evolved weights — see the next section.

Construct one of these in three ways:

```python
from megagem.players import HyperAdaptiveSplitAI, _BidModel

# 1. Default weights (sane starting point).
ai = HyperAdaptiveSplitAI("Eve")

# 2. Custom heads (e.g. unit tests).
ai = HyperAdaptiveSplitAI(
    "Eve",
    treasure=_BidModel(0.7, 0.25, 0.35, -0.10, -0.15, -0.05),
    invest=_BidModel(0.8, 0.10, 0.20, -0.05, -0.05, 0.00),
    loan=_BidModel(0.1, 0.05, -0.40, 0.10, 0.10, -0.05),
)

# 3. Flat 18-element weights vector (what the GA uses).
ai = HyperAdaptiveSplitAI.from_weights("Eve", [0.7, 0.25, 0.35, ..., -0.05])
```

The flat-vector layout is `[treasure (6), invest (6), loan (6)]` in the
exact order `_BidModel.__init__` accepts.

### 8. `Evo2AI` — `megagem/players/evo2.py`

A clean-slate evolved AI — not a subclass of the hyper-adaptive family.
The design brief was "throw out the cruft accumulated across the
heuristic lineage and let the GA tune a lean feature set":

* **Drops `_reserve_for_future`.** The old lineage withheld coins for
  hypothetical later rounds; Evo2 lets the GA figure out pacing
  directly via a progress feature.
* **Exact round-count estimator.** `auctions_left / 25` is replaced by
  `_expected_rounds_remaining`, a closed-form multivariate hypergeometric
  over the remaining auction-deck multiset.
* **Raw cash features.** The `cash_ratio` family is replaced by raw
  integer `my_coins`, `avg_opp_coins`, `top_opp_coins`.
* **Distribution-aware treasure features.** The treasure head gets two
  new per-card features — `ev` and `std` of the prize value — derived
  from the same hypergeometric distribution used for estimation.
* **Mission-probability delta.** Treasure EV adds a per-mission
  `(P(I win | I take the gems) − P(I win | likely opponent does)) ×
  mission.coins` term, on top of the existing hard/soft mission bonuses.
* **Heads output the bid in coins directly.** No more `discount × EV`
  multiplication: `bid = bias + Σ wᵢ · featureᵢ`, clamped to
  `[0, cap]` once at `choose_bid`. This frees the GA from the implicit
  "scale by EV" coupling the old discount form baked in.

**19 weights total** (treasure 7 + invest 6 + loan 6). Tuned by
`python -m scripts.evolve --ai evo2`.

### 9. `Evo3AI` — `megagem/players/evo3.py`

Same feature base as Evo2, plus an **opponent-pricing signal**: every
head reads two new features — a recency-weighted `mean_delta` and
`std_delta` over the observed gap between the highest-opponent bid
and Evo3's own "baseline" bid, split by auction category (treasure /
invest / loan).

The signal is populated via the `observe_round` hook: after every
auction the engine hands Evo3 the public result, and Evo3 logs
`(category, max_opp_bid − baseline)` into a short ring buffer.

> **The important detail.** `baseline` is **not** Evo3's actual bid —
> it's the bid Evo3 *would* have produced with the default `(0, 1)`
> deltas. Using the actual bid would create a feedback loop where the
> signal depends on Evo3's own learned response. `choose_bid` caches
> the default-delta bid on `self._last_default_bid`, and
> `observe_round` reads it back to compute a stable delta.

Before any history exists, each head sees `(mean_delta, std_delta) =
(0, 1)` — so Evo3 degrades smoothly to "Evo2 + zero opponent signal"
on round one.

**25 weights total** (treasure 9 + invest 8 + loan 8). Tuned by
`python -m scripts.evolve --ai evo3`, which defaults to training
against the averaged fitness of every other bot in the zoo.

### 10. `Evo4AI` — `megagem/players/evo4.py`

The current champion. Identical to Evo3 on invest and loan; extends
the treasure head with two independent mechanisms, both tuneable by
the GA:

1. **Per-color signal (gem inference from prices).** Evo4 watches how
   much opponents bid on *treasures* versus its own baseline bid, and
   attributes any excess to the colors of the gems that were on offer.
   A persistent per-color `_color_signal` dictionary accumulates that
   evidence ("opponents consistently bid above expectation on
   treasures containing Blue → they probably already hold Blue gems
   in hand → the final value display will skew toward Blue"). When
   Evo4 evaluates a pending treasure, it pushes the hypergeometric
   chart-index expectation for each color by
   `color_bias_influence × color_signal[color]` before the
   treasure-value chart lookup, with linear interpolation between
   adjacent chart rows for fractional shifts. A single
   `color_bias_influence` scalar gates the whole mechanism — set it
   to zero and Evo4 is chart-blind to the signal.
2. **Opponent-bid prediction.** For every seat that isn't us, Evo4
   runs a small internal Evo2-style treasure head from that seat's
   POV — its own coins become `my_coins`, everyone else (including
   us) becomes the `avg_opp_coins` / `top_opp_coins` bucket — feeding
   in the same `(ev, std)` we computed for the treasure as a proxy
   for their estimate. Taking `max` and `mean` across the per-seat
   predicted bids gives two new features, `opp_max` and `opp_avg`,
   that the treasure head can weight via `w_opp_max` / `w_opp_avg`.
   **The internal-predictor weights are themselves evolvable** — they
   live in the flat weights vector and the GA tunes them alongside
   the outer-head weights, so the GA gets to decide how to model
   opponents rather than inheriting a fixed Evo2 snapshot. Each
   per-opponent prediction is clamped to that seat's `max_legal_bid`,
   and the opponent prediction uses the *unbiased* (zero-shift) EV/std
   so Evo4's private color-signal beliefs don't leak into what it
   thinks opponents see.

**35 weights total** — treasure 11 (Evo3's 9 + `w_opp_max`
/ `w_opp_avg`) + invest 8 + loan 8 + `color_bias_influence` 1 +
internal Evo2 treasure predictor 7. Tuned by
`python -m scripts.evolve --ai evo4`.

Defaults zero the new outer weights and seed the internal predictor
with the Evo2 champion, so a freshly constructed `Evo4AI()`
reproduces Evo3 behaviour exactly until the GA lights up any of the
new weights.

---

## Testing

```bash
# Whole suite (currently 112 tests, ~1s).
python -m unittest discover

# Just the AI / heuristic tests (the biggest file).
python -m unittest tests.test_heuristic -v

# Evo2-specific tests.
python -m unittest tests.test_evo2 -v

# Evo3-specific tests.
python -m unittest tests.test_evo3 -v

# Evo4-specific tests.
python -m unittest tests.test_evo4 -v

# A specific test class.
python -m unittest tests.test_heuristic.HyperAdaptiveSplitBidTest -v

# A single test.
python -m unittest tests.test_heuristic.HyperAdaptiveSplitBidTest.test_invest_uses_invest_model_not_treasure_model
```

`tests/test_heuristic.py` is intentionally large because the AIs share
many helper functions (`_expected_final_display`, `_treasure_value`,
`_hyper_*` family, `_compute_discount_features`, …) and each of those
helpers gets its own focused unit test before the AI itself is
exercised. `tests/test_evo2.py`, `tests/test_evo3.py`, and
`tests/test_evo4.py` cover the feature additions unique to those AIs
(exact rounds-remaining estimator, mission-probability delta,
opponent-delta ring buffer, baseline-bid caching, per-color signal
updates, biased chart-index shift, opponent-bid prediction). The
head-to-head tests at the end of each file are smoke checks ("X
should beat 3× RandomAI on chart Y at least 60% of the time"); they
run ~60 games each and the whole suite finishes in about a second.

---

## Evolving a better AI (the GA)

All four evolvable AIs are tuned by the **single unified GA** in
`research/scripts/evolve/`. Pick which AI you want to tune via `--ai`
and which opponent mode via `--opponent`:

```bash
python -m scripts.evolve --ai evo1                    # tune HyperAdaptiveSplitAI
python -m scripts.evolve --ai evo2 --opponent vs_all  # default opponent
python -m scripts.evolve --ai evo3 --opponent self_play
python -m scripts.evolve --ai evo4 --opponent vs_evo3
```

The four profile keys (`evo1`, `evo2`, `evo3`, `evo4`) cover every
GA-targeted AI. `evo1` is `HyperAdaptiveSplitAI` (the class name is
unchanged — only the GA uses the symmetric `evo1` label). Each profile
only declares the minimal metadata it needs in
`scripts/evolve/profiles.py` (its AI class, weight count, and
mutation sigma/clip); the lookup chain is shared across all profiles
(parameterized only by profile key), and the fallback genome comes
from the AI class's own `flatten_defaults()` classmethod, so the
profile registry stays tiny. The GA loop in `scripts/evolve/ga.py`
is fully generic over the profile so all four share one code path.

**Individual #0 of every GA run is seeded from
`saved_best_weights/`** via the shared lookup chain — same chain used
to pull frozen opponents for `vs_all` / `vs_evoK` modes. Re-running
`python -m scripts.evolve --ai evo3` therefore iteratively refines the
current champion instead of starting from a stale constant. If no
weights file exists yet, the GA falls back to
`profile.ai_class.flatten_defaults()` (the class's hardcoded
`DEFAULT_*` constants) and the run starts from there.

| Profile | Class | Weights | Mutation σ |
|---------|-------|---------|------------|
| `evo1` | `HyperAdaptiveSplitAI` | 18 | 0.15 |
| `evo2` | `Evo2AI` | 19 | 0.05 |
| `evo3` | `Evo3AI` | 25 | 0.05 |
| `evo4` | `Evo4AI` | 35 | 0.05 |

### Opponent modes

Every profile supports the same eight opponent modes:

| `--opponent` | Opponents | Notes |
|--------------|-----------|-------|
| `vs_all` *(default)* | Pools fitness across `Heuristic` + every other `evo*` profile (4 slates) | **4× longer per generation** vs single-opponent modes; avoids overfit to any single baseline. Random is intentionally excluded — every tuned bot trivially beats it, so its near-100% win rate would wash out the signal from the harder opponents. Evo opponents are loaded from `saved_best_weights/` if present, else fall back to class defaults. |
| `vs_random` | Fixed `RandomAI` opponents | Floor-of-the-zoo sanity check. |
| `vs_heuristic` | Fixed `HeuristicAI` opponents | Mid-tier baseline. |
| `vs_evo1` | Fixed `HyperAdaptiveSplitAI` from `saved_best_weights/` | Train against the pre-Evo2 champion. |
| `vs_evo2` | Fixed `Evo2AI` from `saved_best_weights/` | |
| `vs_evo3` | Fixed `Evo3AI` from `saved_best_weights/` | |
| `vs_evo4` | Fixed `Evo4AI` from `saved_best_weights/` | |
| `self_play` | Sampled from the current population each generation | Pure co-evolution. |

`vs_all` for any profile excludes the challenger's own class
automatically — `--ai evo3 --opponent vs_all` pools across
Heuristic, evo1, evo2, evo4.

Outputs land in `artifacts/` (gitignored):

* `artifacts/best_weights_evo{K}_{tag}_{N}p.json` — winning genome + GA config
* `artifacts/evolve_evo{K}_history_{tag}_{N}p.png` — best/mean fitness curve

where `{tag}` is the opponent mode (`vs_all`, `vs_random`, …, `self`)
and `{N}` is the player count.

### Promoting a fresh winner

The CLI and heatmap only read weights from `saved_best_weights/`, so
**promoting a fresh winner is an explicit copy**:

```bash
python -m scripts.evolve --ai evo4
cp artifacts/best_weights_evo4_vs_all_4p.json saved_best_weights/
```

### Fitness strategy

Every profile uses the same fitness strategy:

* **Rotating per-generation seeds.** Each generation uses a fresh
  seed offset `(seed + gen + 1) * 9973` instead of a fixed range.
  Consequence: best-fitness per generation is **not** monotone — a
  generation can land on a harder seed batch and the printed best
  dips. The plot reflects raw per-generation scores.
* **Held-out re-evaluation of the top-5 elites.** At the end of the
  GA the top survivors of the last generation are rescored on a
  held-out seed range (offset `(seed + generations + 100) * 9973`),
  against the *same* opponent providers training used. The pooled
  winner becomes the saved best.
* **Per-opponent breakdown.** When the mode is multi-provider
  (`vs_all`), the held-out winner's per-opponent rates print to
  stdout (`vs 3x Random = 88.0%`, …) so you can see which baselines
  the new champion still struggles against.

### Shared flags

```bash
python -m scripts.evolve --ai evoN \
    [--opponent {vs_all,vs_random,vs_heuristic,vs_evo1..vs_evo4,self_play}] \
    [--population 24] [--generations 30] [--games-per-chart 40] \
    [--seed 0] [--num-players {3,4,5}] [--output-dir artifacts]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--ai` | *(required)* | Profile to tune: `evo1`, `evo2`, `evo3`, `evo4`. |
| `--opponent` | `vs_all` | Opponent mode (one of the eight above). |
| `--population` | `24` | Individuals per generation. |
| `--generations` | `30` | Generation count. |
| `--games-per-chart` | `40` | Games per fitness eval per chart (5 charts × 40 = 200 games/eval). |
| `--seed` | `0` | Master RNG seed for the GA itself. |
| `--num-players` | `4` | Seats per fitness game. |
| `--output-dir` | `artifacts` | Where the plot + JSON go. |

> **Default-reduces-to-Evo3 guarantee for evo4.** When no evo4
> weights file exists in `saved_best_weights/`, the GA's individual
> #0 comes from `Evo4AI.flatten_defaults()`, which zeros the new
> outer weights (`color_bias_influence`, `w_opp_max`, `w_opp_avg`)
> and uses the Evo2 class defaults for the internal predictor, so
> generation 0's best individual plays exactly like Evo3. Any
> improvement from generation 1 onward is strictly due to the GA
> discovering useful settings for the new features. (Once you
> promote an evo4 file into `saved_best_weights/`, fresh runs
> start from that champion instead.)

### Expected results

A default `python -m scripts.evolve --ai evo1 --opponent vs_heuristic`
run reaches **best fitness ≥ 0.80** within the first ~20 generations
and finishes around 0.85–0.88 vs 3× `HeuristicAI`. On held-out seeds
the same weights give **~70–75% win rate** averaged across all charts.

`python -m scripts.evolve --ai evo{2,3,4} --opponent vs_all` are the
current strongest recipes — the evo2/evo3 runs reach a held-out pooled
fitness of roughly 0.70 across the earlier bots. The per-opponent
breakdown is uneven: Evo2/Evo3/Evo4 crush Random (>95%) and dominate
the heuristic family (~70–85%) but are close enough to each other that
small population / generation changes can flip the head-to-head. Evo4
in particular is initialized to produce an exact Evo3 replica on
generation 0, so any held-out improvement over Evo3 is a clean win
for the new color-signal and opponent-bid-prediction features.

That's the one number that matters: training fitness is just a signal
the GA optimises — the held-out heatmap (next section) is where you
verify the model actually generalised.

---

## Benchmarking: pairwise heatmap

`scripts/heatmap_pairwise.py` builds an `N × N` matrix where `M[row,
col]` is the win rate of one `row` AI seated against three copies of
`col`, averaged across all five charts on a held-out seed range. The
seed range is set well above every GA's training seeds (default 200..)
so the evolved-AI rows reflect generalisation, not memorisation.

The current matrix covers **6 AIs**: `Random`, `Heuristic`,
`EvolvedSplit` (GA-tuned `HyperAdaptiveSplitAI`), `Evo2`, `Evo3`, and
`Evo4`. Each cell is 1000 games (5 charts × 200 seeds).

### Run it

```bash
# Reads weights from saved_best_weights/. If a weights file for a
# particular AI is missing, that AI is either dropped (EvolvedSplit)
# or falls back to class defaults (Evo2, Evo3, Evo4).
python -m scripts.heatmap_pairwise
```

The script prints:

1. Every cell as it's computed (`Heuristic vs 3x Evo4 = 9.0%`).
2. A formatted ASCII table at the end.
3. A note where `artifacts/heatmap_pairwise.png` got saved.

### Configuring it

The constants near the top are deliberately easy to bump:

```python
CHARTS = "ABCDE"
SEED_START = 200       # held out: GA trained on 0..9 (or rotating seeds
                       # based on (seed + gen) * 9973 for evo2/evo3)
GAMES_PER_CHART = 200  # → 1000 games per cell when CHARTS == "ABCDE"
```

`GAMES_PER_CHART = 20` is enough to read trends in seconds; bump to
`200` if you want the published numbers to be tight (each cell averages
1000 games and the full 8×8 matrix takes a few minutes).

### Reading the heatmap

* **Rows** = challenger. Higher row average = stronger AI overall.
* **Columns** = opponent pool. Higher column average = an *easier*
  opponent (everyone beats it). *Lower* column = a *harder* opponent.
* **Diagonal** = self-play. With a 4-player game, perfect symmetry
  predicts ~25% per seat; deviations from 25% on the diagonal hint at
  seating-order bias.
* **Evo2, Evo3, and Evo4's columns should be the hardest** — the
  classical bots typically score in single digits against three
  Evo2/Evo3/Evo4 opponents. If they don't, whichever GA produced
  those weights either overfit or didn't run long enough.
* **Evo2, Evo3, and Evo4's rows should dominate** everything except
  each other; they're close enough that small population / generation
  changes to `--ai evo3` / `--ai evo4` runs can flip the head-to-head.

---

## Adding your own AI

1. Drop a new module in `megagem/players/` and subclass `Player` (or
   one of the existing AIs to inherit helpers):

   ```python
   # megagem/players/my_ai.py
   import random

   from megagem.engine import max_legal_bid
   from .base import Player

   class MyAI(Player):
       def __init__(self, name, *, seed=None):
           super().__init__(name)
           self.rng = random.Random(seed)

       def choose_bid(self, public_state, my_state, auction):
           cap = max_legal_bid(my_state, auction)
           # ... your logic here ...
           return min(some_target, cap)

       def choose_gem_to_reveal(self, public_state, my_state):
           return self.rng.choice(my_state.hand)
   ```

2. Re-export the class from `megagem/players/__init__.py` so
   `from megagem.players import MyAI` continues to work.

3. Add unit tests (either a new `tests/test_my_ai.py` file or extend
   an existing one). At minimum, prove your AI beats `RandomAI` ≥60%
   on chart A — see the existing `HeadToHeadTest` classes for the
   pattern.

4. Add it to `scripts/heatmap_pairwise.py`'s `make_factories()` if you
   want it to show up in the all-vs-all matrix.

5. (Optional) wire it into the CLI by adding an entry to the
   `AI_FACTORIES` dict in `megagem/__main__.py`.

The engine never imports anything from `players/` at runtime (look at
the `TYPE_CHECKING` import in `engine.py`), so adding a new AI is
purely additive — you can't break the engine by editing a new module
in `megagem/players/`.

---

## Common gotchas

* **`--ai evolved` / `--ai evo2` / `--ai evo3` / `--ai evo4` need
  weights in `saved_best_weights/`.** The CLI **never** reads
  `artifacts/` — promoting a fresh GA winner is always an explicit
  `cp`. `evolved` errors out if no weights file is found;
  `evo2`/`evo3`/`evo4` fall back to class defaults with a stderr
  warning. (Evo4's class defaults reproduce Evo3 exactly on round
  one, so missing Evo4 weights still give you a reasonable opponent.)
* **`ModuleNotFoundError: matplotlib`** — only the GA + heatmap scripts
  need it. `pip install matplotlib`. The CLI and tests don't.
* **Tests pass but the GA hangs** — you probably ran it inside an
  IDE/notebook that wired up matplotlib's interactive backend. The
  script forces `matplotlib.use("Agg")` before importing pyplot so this
  shouldn't happen, but if you've fiddled with backends, set
  `MPLBACKEND=Agg` in your environment.
* **`scripts.evolve` best-fitness dips between generations** — that's
  expected. The unified GA uses rotating seeds
  (`(seed + gen + 1) * 9973`) for *every* profile, so generation N can
  land on a harder batch than generation N-1. The final "winner" is
  chosen by a held-out re-eval of the top-5 elites *after* the main
  loop, not by picking the best per-generation score. Generalise-check
  the held-out fitness against the heatmap (`SEED_START = 200` by
  default, kept above the GA's training seed range).
* **`python megagem/__main__.py` doesn't work** — use
  `python -m megagem` so the package imports resolve correctly.
* **Reveal phase is mandatory** — the auction winner *must* reveal a
  gem from their hand if their hand is non-empty. AIs that return a gem
  not in hand will get a random one substituted by the engine; this is
  a safety net, not a feature you should rely on.
* **Bid clamping is per-card, not per-coin** — `LoanCard` lets you bid
  up to `coins + loan_amount` because the loan amount is conceptually
  paid first. `clamp_bid` handles this for you.
