# mega-gem

A research playground for the **MegaGem** auction-and-collection card
game: a pure-Python engine, a zoo of bidder AIs, genetic-algorithm
tuners that breed stronger ones, and a static web client where you can
play the current champions in your browser.

**▶ Play it: <https://lachy-dauth.github.io/mega-gem/>**. The page runs
the canonical Python engine client-side via
[Pyodide](https://pyodide.org) and is hosted on GitHub Pages, so there
is no game server.

The full game rules (turn structure, bid ties, mission categories, value
charts) live in [`research/RULES.md`](research/RULES.md). This README
covers everything *around* the rules: how to run the game, what each AI
does, how to test them, how to evolve a stronger one, and how to read
the benchmark plots.

> **Repo layout note.** The canonical engine, AI zoo, tests, GA
> tuners, and checked-in weights live under **`research/`**. The static
> browser client lives in **`web/`**. **Every Python command in this
> README assumes you are inside `research/` unless it says otherwise.**
> The research scripts resolve paths like `saved_best_weights/` and
> `artifacts/` relative to the current working directory. The only
> exception is the site build (`python web/build.py`), which runs from
> the repo root.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Repo layout](#repo-layout)
3. [Playing in the terminal](#playing-in-the-terminal)
4. [Playing in the browser (GitHub Pages)](#playing-in-the-browser-github-pages)
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

# 4. All Python commands live under research/, so switch there.
cd research

# 5. Run the test suite (should print "Ran 122 tests ... OK")
python -m unittest discover

# 6. Play a game against three Heuristic AIs in the terminal
python -m megagem

# 7. Watch four AIs play each other end-to-end on chart E
python -m megagem --all-ai --ai evo4 --chart E --seed 42

# 8. Or build and serve the browser client locally (from the repo root):
cd ..
python web/build.py
python -m http.server -d _site      # → http://127.0.0.1:8000/
```

If everything above worked you have a clean install.

---

## Repo layout

```
mega-gem/
├── web/                     # Static single-player client (GitHub Pages)
│   ├── index.html              # Setup → game → scores
│   ├── bots.html               # Bot descriptions, learned weights, heatmap
│   ├── app.js                  # UI: renders BrowserSession views, sends bids/reveals
│   ├── engine-worker.js        # Web Worker: boots Pyodide, unpacks megagem.zip
│   ├── style.css
│   ├── heatmap_data.json       # Written by research/scripts/generate_heatmap_json.py
│   └── build.py                # Assembles _site/ (web files + megagem.zip)
├── .github/workflows/pages.yml  # Tests on every push/PR; deploys Pages from main
├── README.md
├── CLAUDE.md                # Cheat-sheet for Claude Code
└── research/                # Everything Python. Run commands from here
    ├── megagem/                 # Game engine + AIs (the bit you import)
    │   ├── __main__.py          # Terminal CLI entry point: `python -m megagem`
    │   ├── browser.py           # Step-wise session driver used by the web client
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
    │   ├── cycle_evolve.py           # Round-robin retraining of every evo profile
    │   ├── generate_heatmap_json.py  # Heatmap data for web/bots.html
    │   └── heatmap_pairwise.py       # All-vs-all win-rate matrix plot
    ├── tests/                   # unittest suite (122 tests, stdlib only)
    │   ├── test_browser.py      # Web session driver: replay determinism, phases
    │   ├── test_cards.py
    │   ├── test_engine.py
    │   ├── test_heuristic.py    # Big file — covers every AI's helper math
    │   ├── test_evo2.py         # Evo2AI-specific helpers + head-to-head
    │   ├── test_evo3.py         # Evo3AI-specific helpers + head-to-head
    │   ├── test_evo4.py         # Evo4AI-specific helpers + head-to-head
    │   ├── test_missions.py
    │   └── test_scoring.py
    ├── saved_best_weights/      # Checked-in GA outputs — the CLI + site read these
    │   ├── best_weights_evo1_vs_all_4p.json   # HyperAdaptiveSplitAI
    │   ├── best_weights_evo2_vs_all_4p.json   # Evo2AI
    │   ├── best_weights_evo3_vs_all_4p.json   # Evo3AI
    │   └── best_weights_evo4_vs_all_4p.json   # Evo4AI
    ├── artifacts/               # Transient GA output (gitignored)
    └── RULES.md                 # The actual game rules — read this first
```

Everything in `research/` (engine, CLI, tests) has **zero** third-party
dependencies; it is stdlib only. matplotlib is needed only for the GA +
heatmap scripts in `research/scripts/`. The browser client in `web/` is
vanilla JS and loads Pyodide from a CDN at runtime; `web/build.py` is
stdlib only.

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

## Playing in the browser (GitHub Pages)

The live site is <https://lachy-dauth.github.io/mega-gem/>. It is a
single-player client: you against 2–4 bots of your choice (any of
Random, Heuristic, Evo1–Evo4 and the Search champion), or a **Watch bots** spectator mode where
every hand is face-up and you can step or auto-play through a bot-only
game. Turn on *Show bot reasoning* to see each AI's `explain_bid` output
(features, head outputs, opponent-delta history) in the round log, the
same lines the CLI's `--debug` prints. Every game shows its seed; enter
it on the setup screen (or hit *Rematch*) to replay the same deal.

### How it works

There is no game server and no JS port of the rules. The browser runs
the **canonical `research/megagem` package** inside
[Pyodide](https://pyodide.org) (CPython compiled to WebAssembly):

```
index.html + app.js ──postMessage──▶ engine-worker.js (Web Worker)
   (renders views,                      ├─ loadPyodide()  ← cdn.jsdelivr.net/npm/pyodide
    sends bids/reveals)                  ├─ unpack megagem.zip
                                         │    megagem/**.py + saved_best_weights/*.json
                                         └─ megagem.browser.BrowserSession
                                               ↓
                                            engine.play_round()  ← unchanged
```

`megagem/browser.py` is a frontend adapter, like the CLI in
`__main__.py`. The engine is synchronous and calls
`choose_bid` / `choose_gem_to_reveal` inline, but a page can't block
waiting for a click:

- **Bids** are simultaneous, so the human's bid is collected *before*
  `play_round` runs and preloaded on the human seat.
- **Reveals** happen mid-round, and only if the human wins. The session
  deep-copies `(state, rng)` before each round. If the human's reveal is
  needed, the seat raises `_NeedReveal`, the page asks which gem to
  reveal, and the round is **replayed from the snapshot** with the answer
  preloaded. Every AI and the tie-break RNG live inside the snapshot, so
  the replay is the same round. `tests/test_browser.py` checks that a
  browser game produces exactly the bids and scores of the same game
  played straight through the engine.
- **Bot reasoning** is computed on a throwaway deep copy, so turning it
  on cannot change a game.

Weights come from `saved_best_weights/` via the same lookup chain as the
CLI. If no file matches the seat count, the site falls back to the 4p
weights (every checked-in file is 4p), then to class defaults.

### Build and run locally

```bash
# From the repo root.
python web/build.py                 # → _site/ (web files + megagem.zip)
python -m http.server -d _site      # → http://127.0.0.1:8000/
```

`build.py` copies `web/*.{html,css,js,svg,json}`, zips
`research/megagem/**/*.py` plus `research/saved_best_weights/*.json` into
`_site/megagem.zip`, and stamps a content hash into the asset URLs so
browsers pick up new engines and weights. Rebuild after changing
anything under `research/megagem/` or promoting new weights.

The first page load downloads the Pyodide runtime (~10 MB, then cached).
To work offline, point the worker at a local copy of the
[`pyodide` npm package](https://www.npmjs.com/package/pyodide):
`npm pack pyodide@0.29.5`, unpack it into `_site/pyodide/`, and open
`http://127.0.0.1:8000/?pyodide=/pyodide/`.

### Deployment

`.github/workflows/pages.yml` runs the unit tests and a headless CLI
smoke test on every push and PR, builds `_site/`, and deploys it to
GitHub Pages on pushes to `main`. One-time setup: **Settings → Pages →
Build and deployment → Source: GitHub Actions**.

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
internal state: Evo3 logs opponent-bid deltas (§9), Evo4 updates its
per-color gem-inference signal (§10), and SearchAI identifies opponents
and infers their hands (§11). NashAI (§12) deliberately keeps no
per-opponent state.

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

### 11. `SearchAI` — `megagem/players/search.py`

**The current champion**, and a different kind of bot: there is no bid
formula. Before every bid it *plays the rest of the game forward* in a
few dozen sampled worlds and picks the bid that wins most often
(determinized Monte Carlo search). `--ai search` in the CLI; "Search" on
the website.

1. **Sample worlds.** The only hidden information is how the unseen
   gems split between opponents' hands and the gem deck, plus the order
   of the remaining auction cards (`unseen_pool` derives the pool from
   public information only). Each world deals that pool into hands and
   a deck of the right sizes and shuffles the auction deck.
2. **Infer hands from bids** (`infer`). On treasure rounds it predicts
   each opponent's bid for every hand composition they could hold (at
   most 70), and once bids are public keeps a likelihood table per
   (seat, round). Worlds are importance-sampled by how well their hands
   explain every past bid, with gems revealed since then added back. A
   floor on each likelihood keeps a model mismatch (or a human) from
   zeroing out the true world.
3. **Identify opponents** (`identify`). A bank of model bots (Random,
   Heuristic, Evo1–Evo4) is scored against each seat's bids; each seat
   is then modelled by its most likely bot. **Shadow** copies of every
   (seat, model) watch the real game, so Evo3/Evo4 history features
   (bid deltas, colour signal) match the real opponent's (`shadow`).
4. **Calibrate** (`calibrate`). For a seat no model fits (a
   lowballer, an erratic human), it tracks how far each bid fell
   outside the range the seat's model could produce for *any* hand, and
   adds a recent error to that seat's predicted bids. It's gated: only
   seats that miss at least twice in their last 8 bids are adjusted, so
   the near-exactly-predicted zoo bots are left alone.
5. **Price to win.** Bids are sealed and simultaneous, so in a world
   its own bid can't change anyone else's. Each world collapses to one
   threshold: the cheapest winning bid under the engine's tie-break.
6. **Two rollouts per world**, lose vs win-at-threshold, sharing the
   world and RNG seed. Opponents replay exactly the predicted bids for
   this round, then continue as their seat models. Its own future turns
   use a *sniper* rollout policy (`rollout_snipe`): Evo4's valuation,
   but paying only the threshold when Evo4 would win anyway, which is
   closer to how SearchAI itself plays. Outcomes are scored as a sigmoid
   of the final margin over the best opponent.
7. **Choose.** For each legal bid `b`, worlds with threshold > `b`
   score their lose branch and the rest their win branch, minus the
   overpayment `b − threshold`. Highest average wins.

Gem reveals don't change final values (every hand ends up in the Value
Display), only what opponents learn; it uses `HeuristicAI`'s reveal
rule. (Searching reveals with rollouts was tried and didn't help.)

**Results** (win share with ties split, all five charts, seat rotated;
an equal-strength bot would get 25% at a 4-player table):

| Table | SearchAI | Fair share |
|---|---|---|
| vs 3× Evo4 (600 games) | **79.6% ± 1.6** | 25% |
| vs 3× Evo3 | **87.0% ± 2.4** | 25% |
| vs Heuristic + Evo3 + Evo4 | **89.0% ± 2.2** | 25% |
| vs Evo2 + Evo1 + Heuristic | **84.5% ± 2.6** | 25% |
| 3 players, vs 2× Evo4 | **89.3% ± 2.5** | 33% |
| 5 players, vs 4× Evo4 | **84.7% ± 2.9** | 20% |

For comparison, Evo4 wins about a quarter of games against copies of
itself. Ablation (200 paired seeds each; the components stack):

| Configuration | vs 3× Evo4 | vs 3× Evo3 |
|---|---|---|
| plain search (Evo4 models, no history) | 74.3% | 61.8% |
| + shadow models | 76.7% | |
| + hand inference | 79.0% | |
| + opponent identification | 80.5% | 80.5%¹ |
| + sniper rollouts | 84.0% | 85.5% |
| + gated calibration (**standard**) | 84.3% | 87.0%² |

¹ identify + shadow, without inference. Identification matters most
when the table isn't all Evo4: it lifts the mixed Heuristic/Evo3/Evo4
table from 70.5% to 83.3% and 3× Heuristic from 76.7% to 95.9%. A paired
400-seed test put the sniper at +5.0 ± 2.2 points. Worlds trade
strength for time (12 → 72%, 24 → 80%, 48 → 82.5% at 1×/2×/4× cost);
the sigmoid temperature (3, 6, 12) and searched reveals made no
measurable difference. Runtime is ~0.2 s per bid in CPython.

² different run; calibration is neutral against the zoo bots (paired
difference within ±2 points on every bot table).

**Robustness to players it can't model.** The zoo bots are all in
SearchAI's model bank, but humans aren't. To check, three synthetic
"human-like" opponents distort Evo4's bids in ways no bank model
reproduces (200 paired seeds each):

| Opponents (3×) | without calibration | with calibration |
|---|---|---|
| Lowball (Evo4's bid × 0.5) | 31.2% | **67.2%** |
| Noisy (Evo4's bid ± up to 3) | 58.0% | **69.8%** |
| Overbid (Evo4's bid × 1.5) | 81.0% | **86.8%** |

Without calibration a lowballer *beats* SearchAI: it keeps predicting
a higher price-to-win than anyone pays and sits out. An ungated
version fixed that but cost ~3 points against the exactly-predicted
zoo bots; gating it on two misses in the last eight bids keeps both.

Construct it with `SearchAI.standard(name, seed=..., weights={...})`,
where `weights` maps `evo1`..`evo4` to weight vectors for its internal
models (`None` → class defaults). The CLI and web client pass their
loaded `saved_best_weights/`. Every component is a constructor flag
(`infer`, `shadow`, `identify`, `calibrate`, `rollout_snipe`,
`reveal_search`, `worlds`, `temperature`) so it's easy to ablate. It is reproducible
for a fixed `seed`, and never reads hidden state:
`tests/test_search.py::NoPeekingTest` reshuffles the real hidden cards
and checks the bid doesn't change.

### 12. `NashAI` — `megagem/players/nash.py`

SearchAI wins by **exploiting**: it predicts each opponent's exact bid
with a model of a zoo bot and outbids it by one. That is also what makes
it exploitable. NashAI asks a different question: *what would rational
players bid here, and what should I bid against them?*

1. **Worlds.** Sample the unseen gems into hands + deck, as SearchAI does.
2. **Everyone's stakes.** In each world, roll the game out once per
   possible winner of this auction, twice each: once with the item
   free and once at a probe price. That gives every player's payoff if
   they win, if each rival wins instead (the externality: losing a
   treasure to the leader hurts more than losing it to last place), and
   how much a coin of bid really costs them in final margin. That cost
   is measured, not assumed to be 1:1. Coins buy underpriced items
   later, so assuming 1:1 once made it pay 19 for a 20-coin loan.
3. **Solve the auction.** Treat the auction as a Bayesian game: each
   opponent's type is how much it values winning (worlds bucketed by
   that value); I'm uncertain which world is real. Regret matching+
   with linear averaging finds an approximate equilibrium, with the real
   tie-break rule, in which every player's bid distribution is a best
   response to everyone else's.
4. **Play the mixed strategy.** Sample the bid from my equilibrium
   distribution. Someone who knows exactly how NashAI plays still can't
   predict its bid, only the distribution.

It carries no per-opponent state and never assumes the opponents are
zoo bots. It's an equilibrium of each *auction* with rollout continuation
values, not of the whole game, but that's enough to make it much harder
to exploit.

**Measuring "Nash-ness": exploitability.** Seat one *exploiter* against
three copies of a bot. The exploiter is a SearchAI whose only opponent
model is the target itself (a fresh copy with a different seed: it knows
the strategy, not the coin flips; shadow copies track targets with
per-game history). Exploitability = its win share minus the fair 25%. A
Nash equilibrium would score 0.

| Target (3 copies) | Exploiter wins | Exploitability |
|---|---|---|
| HeuristicAI | 96.0% | +0.71 |
| Evo4AI | 64.5% ± 4.8 | +0.40 |
| SearchAI | 60.8% ± 6.3 | +0.36 |
| NashAI, pure (α = 0) | 39.9% ± 4.5 | +0.15 |
| **NashAI (default, α = 0.75)** | **41.6% ± 4.5** | **+0.17** |

(6-world exploiter, seats rotated, 60–120 games; Heuristic used an
8-world exploiter.) SearchAI beats the zoo by exploiting deterministic
bots, and that's exactly why it's nearly as exploitable as Evo4. NashAI's
bids come from a mixed strategy, so even a perfect model of it only
yields a bid distribution, not a bid to beat by one.

Head to head, in practice:

| Table | Win share |
|---|---|
| SearchAI vs 3× Evo4 | 79.6% |
| SearchAI vs 3× NashAI (pure) | 44.5% ± 5.0 |
| SearchAI vs 3× NashAI (α = 0.75) | **33.8% ± 4.7** |
| NashAI (pure) vs 3× SearchAI | 24.0% ± 4.3 |
| NashAI (α = 0.75) vs 3× SearchAI | **28.3% ± 4.5** |

For comparison, in the heatmap every GA bot wins 12% or less against 3×
SearchAI.

**Restricted Nash response.** `model_weight=α` forces each opponent,
with probability α, to play what a model (the rollout policy, Evo4)
would bid, and lets it best-respond otherwise (Johanson et al.'s
restricted Nash response). α = 0 is the pure equilibrium bot. Raising α
exploits Evo4-like opponents more and becomes more exploitable itself:

| α | vs 3× Evo4 | vs 3× "lowballers" | Exploiter wins vs 3× it |
|---|---|---|---|
| 0 (pure equilibrium) | 20.8% | | 39.9% (120 games) |
| 0.25 | 32.1% | | 48.3% (60) |
| 0.5 | 36.2% | | 39.7% (60) |
| **0.75 (default)** | **50.8%** | **41.2%** | **41.6% (120)** |
| 1.0 (pure model response) | 57.8% | 33.8% | 30.0% (60) |

(120 games per strength cell, all on the same seeds.) Strength against
Evo4 rises steadily with α, while the exploitability differences between
α values are all within noise. Every setting stays far below Evo4 and
SearchAI, because the bids remain mixed. The cost of α = 1 shows up
against opponents the model *doesn't* fit. Against "lowballers" (Evo4
bidding half its usual amount, see §11), α = 0.75 beats α = 1 by
7.5 ± 5.0 points, mirroring the 6.9 ± 4.4 that α = 1 gains against
real Evo4s. A human is never exactly Evo4, so 0.75 is the default.

---

## Testing

```bash
# Whole suite (currently 122 tests, ~1s).
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
