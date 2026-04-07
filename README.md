# mega-gem

A pure-Python implementation of the **MegaGem** auction-and-collection card
game, plus a small zoo of bidder AIs and a genetic-algorithm tuner that
breeds the strongest one. Also (eventually) an online version.

The full game rules — turn structure, bid ties, mission categories, value
charts — live in [`RULES.md`](RULES.md). This README covers everything
*around* the rules: how to run the game, what each AI does, how to test
them, how to evolve a stronger one, and how to read the benchmark plots.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Repo layout](#repo-layout)
3. [Playing in the terminal](#playing-in-the-terminal)
4. [Playing in the browser](#playing-in-the-browser)
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
#    matplotlib is only required for the GA + heatmap scripts in scripts/.
pip install matplotlib              # only if you want plots

# 4. Run the test suite (should print "Ran 73 tests ... OK")
python -m unittest discover

# 5. Play a game against three Heuristic AIs in the terminal
python -m megagem

# 6. Watch four AIs play each other end-to-end on chart E
python -m megagem --all-ai --ai adaptive --chart E --seed 42

# 7. Or play in your browser — no server, no build step:
open play/index.html       # macOS; on Linux use `xdg-open`, Windows `start`
```

If everything above worked you have a clean install.

---

## Repo layout

```
mega-gem/
├── megagem/                 # Game engine + AIs (the bit you import)
│   ├── __main__.py          # Terminal CLI entry point: `python -m megagem`
│   ├── cards.py             # Gem / Auction / Treasure / Loan / Invest cards
│   ├── engine.py            # setup_game, play_round, score_game
│   ├── missions.py          # Mission deck (shields / pendants / crowns)
│   ├── players.py           # Player ABC + every AI implementation
│   ├── render.py            # Pretty-printing for the CLI
│   ├── state.py             # GameState / PlayerState dataclasses
│   └── value_charts.py      # The five value charts A–E
├── play/                    # Browser frontend — open index.html, no server
│   ├── index.html           # Menu + game screens
│   ├── style.css            # Dark theme, color-coded gems
│   ├── megagem.js           # JS port of the engine + RandomAI + HeuristicAI
│   └── ui.js                # UI controller / state machine
├── scripts/                 # Standalone runnables (NOT imported by megagem/)
│   ├── evolve_hyper_adaptive.py   # GA tuner for HyperAdaptiveSplitAI
│   └── heatmap_pairwise.py        # All-vs-all win-rate matrix plot
├── tests/                   # unittest suite
│   ├── test_cards.py
│   ├── test_engine.py
│   ├── test_heuristic.py    # Big file — covers every AI's helper math
│   ├── test_missions.py
│   └── test_scoring.py
├── artifacts/               # GA plots + best_weights*.json (gitignored)
├── README.md
└── RULES.md                 # The actual game rules — read this first
```

The Python CLI, engine, and tests have **zero** third-party dependencies
— stdlib only. matplotlib is needed only for the GA + heatmap scripts in
`scripts/`. The browser frontend in `play/` has no dependencies at all.

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

# Play against the GA-evolved AI with debug mode: opponent hands AND
# AI rationale (features, per-head discounts, value estimates) printed
# after every round.
python -m megagem --ai evolved --debug
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--players {3,4,5}` | `4` | Total seats. One is human unless `--all-ai`. |
| `--chart {A,B,C,D,E}` | `A` | Which value chart to score against. |
| `--seed N` | random | Reproducibility — same seed → same deck order. |
| `--ai {random,heuristic,adaptive,hyper,hyper_adapt,evolved}` | `heuristic` | Opponent AI class. `evolved` loads GA-tuned weights from `artifacts/best_weights_{N}p.json`. |
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

## Playing in the browser

The `play/` directory is a fully self-contained vanilla-JS frontend —
no build step, no bundler, no server, no Pyodide. The Python engine is
ported faithfully to JavaScript inside `play/megagem.js` (same 30-card
gem deck, 25-card auction deck, 30 missions, identical bid resolution
including the "closest player to the left of the previous winner"
tie-break).

### Run it

```bash
# Just open the file in any modern browser:
open play/index.html               # macOS
xdg-open play/index.html           # Linux
start play/index.html              # Windows
# Or double-click it in your file manager.
```

You can also serve it with any static server (e.g.
`python -m http.server` from inside `play/`) if you'd rather hit it
over HTTP.

### Menu options

| Option | Choices | Notes |
|--------|---------|-------|
| Your name | free text | Shown on opponents' boards and the score table. |
| Players | 3 / 4 / 5 | You + 2/3/4 AI opponents. Starting coins/hand size scale per the rules. |
| AI difficulty | Random / Heuristic | "Random" is the floor; "Heuristic" is a faithful port of the Python `HeuristicAI`. |
| Value chart | A–E | Same five charts the Python engine uses. Chart E is the trickiest. |
| Random seed | optional | Leave blank for a random game; set a number to make the deck order reproducible. |

### What the game screen shows

* **Top panel** — round counter, current chart, deck sizes, the auction
  card on offer, the gems available if it's a treasure, the public
  Value Display (with current per-gem prices), and the active missions.
* **Middle panel** — opponent cards: name, coin count, hand size,
  collection size, completed-mission count, and a tally of every gem
  they own (collections are public information). The previous round's
  winner is highlighted in gold.
* **Bottom panel** — your name and coins, your hand (clickable during
  the reveal phase), your collection, your completed missions, and the
  bid input.
* **Right panel** — a running log of every round (auction, all bids,
  winner, what they took, missions completed) and the **Continue**
  button between rounds.

### How a turn flows in the UI

1. A new auction card is drawn — the bid input unlocks.
2. Type a bid (or hit **Pass**) and press **Submit**. AI bids are
   computed instantly and the winner is shown in the log.
3. If you won and have at least one card in hand, your hand becomes
   clickable — click the gem you'd like to reveal into the Value
   Display. (AI winners pick automatically.)
4. Missions auto-complete; the gem deck refills the revealed display.
5. Click **Continue** to draw the next auction card.
6. When the auction or gem deck runs out, the score screen appears
   with everyone's per-category breakdown and a **Play again** button.

> The browser AI is the JS port of `HeuristicAI`. The hypergeometric
> and GA-evolved AIs are Python-only — if you want to play against
> *those*, the terminal CLI is currently the only path.

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

All AIs live in `megagem/players.py` and implement the abstract base
class `Player`:

```python
class Player(ABC):
    name: str
    is_human: bool

    def choose_bid(self, public_state, my_state, auction) -> int: ...
    def choose_gem_to_reveal(self, public_state, my_state) -> GemCard: ...
```

`choose_bid` returns the player's intended bid for the current auction
card; `choose_gem_to_reveal` is called only on the winner each round to
pick which gem from their hand goes into the public value display.

The AIs, in roughly increasing order of strength:

### 1. `RandomAI` — `players.py:51`

Picks bids uniformly between 0 and the legal cap. Reveals a uniformly
random gem from hand. The baseline floor.

### 2. `HumanPlayer` — `players.py:75`

Interactive console UI. Prints the board (yours or `--debug`-revealed
opponents'), prompts for a bid, prompts for the gem to reveal.

### 3. `HeuristicAI` — `players.py:314`

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

### 4. `AdaptiveHeuristicAI(HeuristicAI)` — `players.py:498`

Same shape as `HeuristicAI`, but the fixed `0.75` discount is replaced
by a 5-feature linear model:

```
discount = clamp(BIAS + Σ W_i * feature_i, 0, 1)
```

Features: `progress`, `my_cash_ratio`, `avg_cash_ratio`, `top_cash_ratio`,
`variance`. The constants (`BIAS = 0.70`, `W_PROGRESS = 0.25`, …) were
hand-tuned. Adds two extra knobs that gate loans entirely
(`LOAN_CASH_RATIO_MAX`, `LOAN_DISCOUNT_MIN`).

### 5. `HypergeometricAI` — `players.py:734`

Standalone (does not subclass `HeuristicAI`). Replaces the
"uniform-share over hidden cards" estimator with a true **hypergeometric
distribution** per color. Critical for chart E because Jensen's
inequality bites hard:

> `E[chart_value(X)] ≠ chart_value(E[X])` when `chart_value` is
> non-monotonic.

The vanilla heuristic computes `chart_value(E[X])`; this AI computes
`E[chart_value(X)]`, which is the right thing.

Bid sizing is still a fixed `DISCOUNT = 0.75`, so on charts where the
estimator change isn't decisive, the AI is roughly even with
`HeuristicAI`.

### 6. `HyperAdaptiveAI(AdaptiveHeuristicAI)` — `players.py:904`

Combines the best of (4) and (5): hypergeometric value estimation under
the linear adaptive discount. Also overrides `_reserve_for_future` to
use the hyper-aware average treasure value.

This was the strongest hand-crafted AI before the GA showed up.

### 7. `HyperAdaptiveSplitAI(HyperAdaptiveAI)` — `players.py:1032`

The current champion. The single shared discount was forced to be the
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

---

## Testing

```bash
# Whole suite (currently 73 tests, ~0.5s).
python -m unittest discover

# Just the AI / heuristic tests (the biggest file).
python -m unittest tests.test_heuristic -v

# A specific test class.
python -m unittest tests.test_heuristic.HyperAdaptiveSplitBidTest -v

# A single test.
python -m unittest tests.test_heuristic.HyperAdaptiveSplitBidTest.test_default_treasure_bid_matches_old_hyper_adaptive
```

`tests/test_heuristic.py` is intentionally large because the AIs share
many helper functions (`_expected_final_display`, `_treasure_value`,
`_hyper_*` family, `_compute_discount_features`, …) and each of those
helpers gets its own focused unit test before the AI itself is
exercised. The head-to-head tests at the end are smoke checks
("X should beat 3× RandomAI on chart Y at least 60% of the time"); they
run ~60 games each and finish in well under a second.

---

## Evolving a better AI (the GA)

`scripts/evolve_hyper_adaptive.py` is a small textbook GA that tunes the
18 constants of `HyperAdaptiveSplitAI`. **Fitness = win rate vs three
`HeuristicAI` opponents**, averaged across all five value charts on a
fixed seed range.

### Why fitness is deterministic

The fitness function uses `range(games_per_chart)` for seeds — the
*same* seeds on every call. Two evaluations of the same genome therefore
return the same score. This is non-negotiable for tournament selection;
without it the GA starts chasing noise instead of signal. The same
seeds + matplotlib's `Agg` backend means the whole script runs
headlessly and reproducibly.

### Run it

```bash
# Defaults: pop 24, 30 generations, 50 games per fitness eval.
# Takes ~80s on a modern laptop.
python -m scripts.evolve_hyper_adaptive

# Quick sanity check that the wiring works (~5 seconds).
python -m scripts.evolve_hyper_adaptive \
    --population 6 --generations 3 --games-per-chart 2 \
    --output-dir artifacts/dryrun

# A bigger run.
python -m scripts.evolve_hyper_adaptive \
    --population 48 --generations 60 --games-per-chart 20 --seed 1
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--population` | `24` | Individuals per generation. |
| `--generations` | `30` | Generation count. |
| `--games-per-chart` | `10` | Per fitness eval, per chart (5 charts → 50 games/eval). |
| `--seed` | `0` | Master RNG seed for the GA itself. |
| `--output-dir` | `artifacts` | Where the plot + JSON go. |

### What it produces

After a run, `artifacts/` contains:

* **`evolve_history.png`** — best fitness per generation and population
  mean per generation. Best curve is monotonic-non-decreasing because
  elitism keeps the top 2 individuals each generation.
* **`best_weights.json`** — the winning genome plus the GA config it
  was bred under:
  ```json
  {
    "fitness": 0.88,
    "generation": 22,
    "weights": [0.4752, -0.9420, 0.4377, ..., 0.8339],
    "ga_config": { "population": 24, "generations": 30, ... }
  }
  ```
* **stdout** — per-generation `gen N: best=0.81 mean=0.62 elapsed=2.7s`
  followed by a copy-paste-ready Python block:
  ```python
  DEFAULT_TREASURE = _BidModel(+0.4752, -0.9420, +0.4377, -0.1546, +0.7196, +0.0896)
  DEFAULT_INVEST   = _BidModel(+1.1668, -0.7701, -0.5427, +0.1313, -0.0500, -0.3428)
  DEFAULT_LOAN     = _BidModel(+0.0225, -0.3077, +0.2985, -0.6757, -0.5086, +0.8339)
  ```

  If you want to bake the evolved weights into the class as the new
  defaults, paste those three lines into `HyperAdaptiveSplitAI` in
  `megagem/players.py`. The script deliberately doesn't auto-rewrite
  the source so the diff stays reviewable.

### GA configuration

The constants near the top of `scripts/evolve_hyper_adaptive.py` are
the things you'd usually tweak:

| Constant | Value | Meaning |
|----------|-------|---------|
| `INIT_LO`, `INIT_HI` | `-1.0`, `1.0` | Sampling range for the initial random population. |
| `MUTATION_SIGMA` | `0.15` | σ of the Gaussian perturbation per mutated gene. |
| `MUTATION_RATE` | `0.20` | Per-gene probability of mutation. |
| `MUTATION_CLIP` | `2.0` | Hard clip on gene magnitude after mutation. |
| `TOURNAMENT_SIZE` | `3` | k for tournament selection. |
| `ELITES` | `2` | Top-N individuals copied into the next generation unchanged. |
| `DEFAULT_SEED` | (18 floats) | Individual #0 — seeded with class defaults so the floor is non-trivial. |

The fitness cache is keyed on `tuple(round(w, 4) for w in weights)`, so
elites and rare duplicates skip re-evaluation.

### Expected results

A default run typically reaches **best fitness ≥ 0.80** by generation
20–25 and finishes around 0.85–0.88. On held-out seeds (i.e. fresh
random seeds the GA never trained on) the same evolved weights give
**~70–75% win rate** vs 3× `HeuristicAI` averaged across all charts.

That's the one number that matters: training fitness is just a signal
the GA optimises — the held-out heatmap (next section) is where you
verify the model actually generalised.

---

## Benchmarking: pairwise heatmap

`scripts/heatmap_pairwise.py` builds an `N × N` matrix where `M[row,
col]` is the win rate of one `row` AI seated against three copies of
`col`, averaged across all five charts on a held-out seed range. The
seed range is set well above the GA's training seeds (default 200..) so
EvolvedSplit's numbers reflect generalisation, not memorisation.

### Run it

```bash
# Requires artifacts/best_weights.json — run the GA first.
python -m scripts.heatmap_pairwise
```

The script prints:

1. Every cell as it's computed (`Heuristic vs 3x EvolvedSplit = 9.0%`).
2. A formatted ASCII table at the end.
3. A note where `artifacts/heatmap_pairwise.png` got saved.

### Configuring it

The constants near the top are deliberately easy to bump:

```python
CHARTS = "ABCDE"
SEED_START = 200       # held out: GA trained on 0..9
GAMES_PER_CHART = 200  # → 1000 games per cell when CHARTS == "ABCDE"
```

`GAMES_PER_CHART = 20` is enough to read trends in seconds; bump to
`200` if you want the published numbers to be tight (each cell will
average 1000 games and the matrix takes a few minutes).

### Reading the heatmap

* **Rows** = challenger. Higher row average = stronger AI overall.
* **Columns** = opponent pool. Higher column average = an *easier*
  opponent (everyone beats it). *Lower* column = a *harder* opponent.
* **Diagonal** = self-play. With a 4-player game, perfect symmetry
  predicts ~25% per seat; deviations from 25% on the diagonal hint at
  seating-order bias.
* **EvolvedSplit's column should be the hardest** — most other AIs
  should score in single digits against three EvolvedSplit opponents.
  If they don't, the GA either overfit or didn't run long enough.
* **EvolvedSplit's row should dominate** — 50–70% on every non-Random
  column is the bar. Beating Heuristic at >60% is the success
  criterion; everything above that is gravy.

---

## Adding your own AI

1. Subclass `Player` (or one of the existing AIs to inherit helpers):

   ```python
   from megagem.players import Player
   from megagem.engine import max_legal_bid

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

2. Add a unit test in `tests/test_heuristic.py` modelled on the
   existing `HeadToHeadTest` classes — at minimum, prove your AI beats
   `RandomAI` ≥60% on chart A.

3. Add it to `scripts/heatmap_pairwise.py`'s `make_factories()` if you
   want it to show up in the all-vs-all matrix.

4. (Optional) wire it into the CLI by adding an entry to `AI_TYPES` in
   `megagem/__main__.py`.

The engine never imports anything from `players.py` directly (look at
the `TYPE_CHECKING` import in `engine.py`), so adding a new AI is
purely additive — you can't break the engine by editing `players.py`.

---

## Common gotchas

* **`artifacts/best_weights.json` missing** — run
  `python -m scripts.evolve_hyper_adaptive` at least once. The heatmap
  script depends on it.
* **`ModuleNotFoundError: matplotlib`** — only the GA + heatmap scripts
  need it. `pip install matplotlib`. The CLI and tests don't.
* **Tests pass but the GA hangs** — you probably ran it inside an
  IDE/notebook that wired up matplotlib's interactive backend. The
  script forces `matplotlib.use("Agg")` before importing pyplot so this
  shouldn't happen, but if you've fiddled with backends, set
  `MPLBACKEND=Agg` in your environment.
* **GA fitness looks suspiciously high** — remember the fitness seeds
  are fixed (0..9 by default). Generalise-check with the heatmap, which
  uses seeds ≥200 by default.
* **HyperAdapt loses to plain Heuristic in the heatmap** — yes, this
  surprised us too. A single shared discount head can't simultaneously
  be the right answer for treasures, invests, and loans, so smarter
  value estimation alone doesn't help. That's why
  `HyperAdaptiveSplitAI` exists.
* **`python megagem/__main__.py` doesn't work** — use
  `python -m megagem` so the package imports resolve correctly.
* **Reveal phase is mandatory** — the auction winner *must* reveal a
  gem from their hand if their hand is non-empty. AIs that return a gem
  not in hand will get a random one substituted by the engine; this is
  a safety net, not a feature you should rely on.
* **Bid clamping is per-card, not per-coin** — `LoanCard` lets you bid
  up to `coins + loan_amount` because the loan amount is conceptually
  paid first. `clamp_bid` handles this for you.
