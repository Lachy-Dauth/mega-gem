# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` is the canonical onboarding doc and `RULES.md` is the canonical rules doc — read those for anything not covered here. This file is a curated cheat-sheet for things that are non-obvious or load-bearing for correctness.

## Commands

```bash
# Tests (stdlib unittest, no pytest, ~100 tests, sub-second)
python -m unittest discover
python -m unittest tests.test_heuristic -v
python -m unittest tests.test_evo2 -v
python -m unittest tests.test_heuristic.HyperAdaptiveSplitBidTest.test_default_treasure_bid_matches_old_hyper_adaptive

# Terminal CLI — must use `-m`, running the file directly breaks relative imports
python -m megagem
python -m megagem --all-ai --ai heuristic --seed 1 --quiet      # headless smoke test
python -m megagem --ai evolved --debug                          # opponent AI rationale + revealed hands

# GA tuners + heatmap (need matplotlib)
python -m scripts.evolve_hyper_adaptive
python -m scripts.evolve_evo2                                   # tunes Evo2AI (self-play, rotating seeds)
python -m scripts.heatmap_pairwise                              # requires artifacts/best_weights.json

# Browser frontend — no build, no server
open play/index.html
```

## Dependencies

- Python engine, CLI, and tests: **stdlib only**.
- `scripts/evolve_hyper_adaptive.py` and `scripts/heatmap_pairwise.py`: `pip install matplotlib` (forced to `Agg` backend).
- `play/` browser frontend: zero dependencies, vanilla JS.

## Architecture

### Two parallel game implementations

The single most important thing to know: this repo contains **two independent implementations of MegaGem**.

- `megagem/` — canonical Python engine + the full AI zoo.
- `play/megagem.js` — hand-written JS port of the engine plus `RandomAI` and `HeuristicAI` only. It is **not** a binding, **not** Pyodide, **not** generated. If you change a game rule in one place, you must mirror the change in the other or the two versions will silently diverge. `RULES.md` is the source of truth they should both match.

### Engine layering (`megagem/`)

- `state.py` — pure dataclasses (`GameState`, `PlayerState`). No behavior.
- `cards.py`, `missions.py`, `value_charts.py` — deck factories and the chart-A..E lookup (`value_for`).
- `engine.py` — `setup_game`, `play_round`, `score_game`, `max_legal_bid`, `clamp_bid`. Imports `players.py` only under `TYPE_CHECKING`, so the engine has zero runtime dependency on any AI. **Adding a new AI is purely additive and cannot break the engine.**
- `players.py` — `Player` ABC plus every AI in one ~42 KB file. They live together on purpose: AIs share helper math (`_expected_final_display`, `_treasure_value`, the `_hyper_*` family, `_compute_discount_features`), and tests target those helpers directly.
- `__main__.py` — argparse CLI. New AIs are wired in by adding to the `AI_FACTORIES` dict.
- `render.py`, `explain.py` — CLI pretty-printing and the `--debug` rationale-printing wrapper.

### AI hierarchy (weakest → strongest, all in `players.py`)

1. `RandomAI` — baseline floor.
2. `HumanPlayer` — interactive console.
3. `HeuristicAI` — fixed `0.75` bid discount; uniform-share display estimator. **This is the GA's fitness opponent.**
4. `AdaptiveHeuristicAI(HeuristicAI)` — replaces the fixed discount with a 5-feature linear model.
5. `HypergeometricAI` — standalone (does **not** subclass `HeuristicAI`); replaces the uniform-share display estimator with a true hypergeometric distribution per color. Matters most on chart E because Jensen's inequality makes `E[chart_value(X)] ≠ chart_value(E[X])` for non-monotonic charts.
6. `HyperAdaptiveAI(AdaptiveHeuristicAI)` — hypergeometric estimator + adaptive linear discount.
7. `HyperAdaptiveSplitAI(HyperAdaptiveAI)` — **previous champion.** Splits the single discount head into three independent `_BidModel` heads (`treasure`, `invest`, `loan`). Rationale: a single shared head cannot simultaneously be the right answer for positive-value treasures, free-money invests, and negative-cash-flow loans. 18 tunable constants total = 3 × (1 bias + 5 weights). Construct via class defaults, explicit `_BidModel`s, or `HyperAdaptiveSplitAI.from_weights(name, [18 floats])`. The GA produces individuals with the flat-vector form.
8. `Evo2AI` — **clean-slate evolved AI**, lives in its own file `megagem/players_evo2.py` (deliberate exception to the "all AIs in one file" convention so the previous champion stays bit-stable for benchmarks). Drops `_reserve_for_future`, replaces the `auctions_left/25` progress proxy with an exact closed-form `_expected_rounds_remaining` (multivariate hypergeometric over the auction-deck multiset), drops the `cash_ratio` features in favour of raw integer `my_coins / avg_opp_coins / top_opp_coins`, and gives the treasure head two new per-card features — `ev` and `std` of the prize value — derived from the same hypergeometric distribution. Treasure EV also adds a `_mission_probability_delta` term: a per-mission `(P(I win | I take the gems) − P(I win | likely opponent does)) × mission.coins`, on top of the existing hard/soft mission bonuses. **Heads output the bid in coins directly** (not a discount fraction multiplied by EV/amount): `bid = bias + Σ wᵢ·featureᵢ`, clamped to `[0, cap]` once at `choose_bid`. This frees the GA from the implicit "scale by EV" coupling the discount form baked in. 19 weights (treasure 7 + invest 6 + loan 6); tuned by `scripts/evolve_evo2.py`.

### GA tuners

Two GA scripts, intentionally separate so the new tuner doesn't perturb the old champion's weights:

**`scripts/evolve_hyper_adaptive.py`** — tunes the 18 constants of `HyperAdaptiveSplitAI`.

- Fitness = win rate vs 3× `HeuristicAI`, averaged across charts A–E, on **fixed seed range `range(games_per_chart)`**. **Do not "fix" this by randomizing seeds** — determinism is load-bearing. Without it, two evaluations of the same genome return different scores and tournament selection chases noise instead of signal.
- Elitism (`ELITES = 2`) means best-fitness is monotone non-decreasing per generation.
- Fitness cache keyed on `tuple(round(w, 4) for w in weights)`.
- Outputs `artifacts/best_weights.json` (or `best_weights_{N}p.json` for per-seat-count weights). The CLI's `--ai evolved` loads these at runtime — see `_evolved_factory` in `megagem/__main__.py`.

**`scripts/evolve_evo2.py`** — tunes the 19 constants of `Evo2AI`. Two intentional differences from the old GA:

- **Self-play, not vs HeuristicAI.** Each individual is evaluated by playing against three opponents drawn (with replacement) from the same generation's population — co-evolution rather than fixed-baseline tuning. Self-as-opponent is allowed at probability `1/pop_size`; not worth filtering out.
- **Rotating fitness seeds.** Each generation uses a fresh seed offset `(seed + gen + 1) * 9973` instead of a fixed seed range. Consequence: best-fitness is no longer monotone (a generation can land on a harder seed batch and the printed best dips). To recover a robust final winner, the script does a final held-out re-evaluation of the top-5 elites against the *last* population on a separate seed range, and writes that winner to `artifacts/best_weights_evo2_{N}p.json`. The CLI's `--ai evo2` loads these.

### Heatmap (`scripts/heatmap_pairwise.py`)

- Default `SEED_START = 200` is **held out** from the GA's training seeds (0..9). If you change the GA's training seed range, update `SEED_START` to stay above it or the generalisation check is meaningless.

## Testing conventions

- `tests/test_heuristic.py` is intentionally large. Each shared helper gets a focused unit test before the AI itself is exercised; head-to-head tests at the end run ~60 games per pair and finish in well under a second.
- When adding a new AI, follow the existing `HeadToHeadTest` pattern: at minimum, prove it beats `RandomAI` at ≥60% on chart A. If it should also appear in the all-vs-all heatmap, register it in `scripts/heatmap_pairwise.py`'s `make_factories()`.

## Gotchas

- **Always invoke as `python -m megagem`.** `python megagem/__main__.py` breaks relative imports.
- **Reveal-a-gem is mandatory.** If `choose_gem_to_reveal` returns a gem not in hand, the engine substitutes a random one — that's a safety net for buggy AIs, not a feature to rely on.
- **`LoanCard` bids exceed your coins.** `max_legal_bid` returns `coins + loan_amount` because the loan is conceptually paid first. Always route AI bids through `clamp_bid` rather than trusting raw output.
- **Browser AI is a port, not a binding.** Only `RandomAI` and `HeuristicAI` exist in `play/megagem.js`. Hypergeometric and GA-evolved AIs are Python-only — to play against them, use the terminal CLI.
- **`--ai evolved` requires a weights file.** Run the GA at least once to produce `artifacts/best_weights.json` (or the per-player-count variant); otherwise the CLI exits with a clear error.
