# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` is the canonical onboarding doc and `RULES.md` is the canonical rules doc — read those for anything not covered here. This file is a curated cheat-sheet for things that are non-obvious or load-bearing for correctness.

## Commands

```bash
# Tests (stdlib unittest, no pytest, 120 tests, sub-second)
python -m unittest discover
python -m unittest tests.test_heuristic -v
python -m unittest tests.test_evo2 -v
python -m unittest tests.test_evo3 -v
python -m unittest tests.test_heuristic.HyperAdaptiveSplitBidTest.test_default_treasure_bid_matches_old_hyper_adaptive

# Terminal CLI — must use `-m`, running the file directly breaks relative imports
python -m megagem
python -m megagem --all-ai --ai heuristic --seed 1 --quiet      # headless smoke test
python -m megagem --ai evolved --debug                          # opponent AI rationale + revealed hands
python -m megagem --ai evo3                                     # play vs the current champion

# GA tuners + heatmap (need matplotlib)
python -m scripts.evolve_hyper_adaptive                         # tunes HyperAdaptiveSplitAI vs 3× HeuristicAI
python -m scripts.evolve_evo2                                   # tunes Evo2AI (self-play by default)
python -m scripts.evolve_evo2 --opponent old_evo                # tunes Evo2AI vs HyperAdaptiveSplitAI
python -m scripts.evolve_evo3                                   # tunes Evo3AI (vs_all = avg vs all 6 prior bots)
python -m scripts.heatmap_pairwise                              # requires saved_best_weights/*.json

# Browser frontend — no build, no server
open play/index.html
```

### Weights workflow

GA scripts write their outputs to **`artifacts/`** (gitignored). The CLI, heatmap, and opponent-lookup chains (`old_evo`, `old_evo2`, `vs_evo2`) read *only* from **`saved_best_weights/`** (checked in). Promoting a fresh winner is an explicit copy:

```bash
python -m scripts.evolve_evo3
cp artifacts/best_weights_evo3_vs_all_4p.json saved_best_weights/
```

The `saved_best_weights/` folder currently holds:
- `best_weights_4p.json` — HyperAdaptiveSplitAI (89% vs 3× HeuristicAI)
- `best_weights_evo2_vs_old_4p.json` — Evo2AI trained vs HyperAdaptiveSplitAI (69%)
- `best_weights_evo3_vs_all_4p.json` — Evo3AI trained against all 6 prior bots (70% pooled)

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
- `engine.py` — `setup_game`, `play_round`, `score_game`, `max_legal_bid`, `clamp_bid`. Imports the `players` package only under `TYPE_CHECKING`, so the engine has zero runtime dependency on any AI. **Adding a new AI is purely additive and cannot break the engine.**
- `players/` — `Player` ABC (`base.py`) plus one module per AI (`random_ai.py`, `heuristic.py`, …, `evo2.py`). Shared value-estimation and discount-feature math lives in `players/helpers.py` (`_expected_final_display`, `_treasure_value`, the `_hyper_*` family, `_compute_discount_features`); tests target those helpers directly. `players/__init__.py` re-exports every public name so `from megagem.players import HeuristicAI` continues to work.
- `__main__.py` — argparse CLI. New AIs are wired in by adding to the `AI_FACTORIES` dict.
- `render.py`, `explain.py` — CLI pretty-printing and the `--debug` rationale-printing wrapper.

### AI hierarchy (weakest → strongest, one module per AI under `megagem/players/`)

1. `RandomAI` — baseline floor.
2. `HumanPlayer` — interactive console.
3. `HeuristicAI` — fixed `0.75` bid discount; uniform-share display estimator. **This is the old GA's fitness opponent.**
4. `AdaptiveHeuristicAI(HeuristicAI)` — replaces the fixed discount with a 5-feature linear model.
5. `HypergeometricAI` — standalone (does **not** subclass `HeuristicAI`); replaces the uniform-share display estimator with a true hypergeometric distribution per color. Matters most on chart E because Jensen's inequality makes `E[chart_value(X)] ≠ chart_value(E[X])` for non-monotonic charts.
6. `HyperAdaptiveAI(AdaptiveHeuristicAI)` — hypergeometric estimator + adaptive linear discount.
7. `HyperAdaptiveSplitAI(HyperAdaptiveAI)` — pre-Evo2 champion. Splits the single discount head into three independent `_BidModel` heads (`treasure`, `invest`, `loan`). Rationale: a single shared head cannot simultaneously be the right answer for positive-value treasures, free-money invests, and negative-cash-flow loans. 18 tunable constants total = 3 × (1 bias + 5 weights). Construct via class defaults, explicit `_BidModel`s, or `HyperAdaptiveSplitAI.from_weights(name, [18 floats])`. The GA produces individuals with the flat-vector form.
8. `Evo2AI` (`megagem/players/evo2.py`) — clean-slate evolved AI. Drops `_reserve_for_future`, replaces the `auctions_left/25` progress proxy with an exact closed-form `_expected_rounds_remaining` (multivariate hypergeometric over the auction-deck multiset), drops the `cash_ratio` features in favour of raw integer `my_coins / avg_opp_coins / top_opp_coins`, and gives the treasure head two new per-card features — `ev` and `std` of the prize value — derived from the same hypergeometric distribution. Treasure EV also adds a `_mission_probability_delta` term: a per-mission `(P(I win | I take the gems) − P(I win | likely opponent does)) × mission.coins`, on top of the existing hard/soft mission bonuses. **Heads output the bid in coins directly** (not a discount fraction multiplied by EV/amount): `bid = bias + Σ wᵢ·featureᵢ`, clamped to `[0, cap]` once at `choose_bid`. This frees the GA from the implicit "scale by EV" coupling the discount form baked in. 19 weights (treasure 7 + invest 6 + loan 6); tuned by `scripts/evolve_evo2.py`.
9. `Evo3AI` (`megagem/players/evo3.py`) — **current champion.** Adds an opponent-pricing signal on top of the Evo2 feature set. After every round the engine calls an optional `observe_round(public_state, my_idx, result)` hook on each player; Evo3 uses it to log `(category, max_opp_bid − baseline)` per auction. Every head reads two new features — a 4×-weighted running `mean_delta` and `std_delta` over the most recent auctions in the same category (treasure/invest/loan) — defaulting to `(0, 1)` before any history exists. **Critical detail:** `baseline` is the bid Evo3 *would* have made with the default `(0, 1)` deltas, not its actual bid. Using the actual bid creates a feedback loop where the signal depends on Evo3's own learned response; caching the default-delta bid on `self._last_default_bid` in `choose_bid` and reading it back in `observe_round` breaks that. 25 weights total (treasure 9 + invest 8 + loan 8); tuned by `scripts/evolve_evo3.py`.

### GA tuners

Three GA scripts, intentionally separate so newer tuners don't perturb older champions' weights:

**`scripts/evolve_hyper_adaptive.py`** — tunes the 18 constants of `HyperAdaptiveSplitAI`.

- Fitness = win rate vs 3× `HeuristicAI`, averaged across charts A–E, on **fixed seed range `range(games_per_chart)`**. **Do not "fix" this by randomizing seeds** — determinism is load-bearing. Without it, two evaluations of the same genome return different scores and tournament selection chases noise instead of signal.
- Elitism (`ELITES = 2`) means best-fitness is monotone non-decreasing per generation.
- Fitness cache keyed on `tuple(round(w, 4) for w in weights)`.
- Outputs `artifacts/best_weights_{N}p.json`. The CLI's `--ai evolved` loads it from `saved_best_weights/` after you copy it there.

**`scripts/evolve_evo2.py`** — tunes the 19 constants of `Evo2AI`. Two intentional differences from the old GA:

- **Self-play, not vs HeuristicAI.** Each individual is evaluated by playing against three opponents drawn (with replacement) from the same generation's population — co-evolution rather than fixed-baseline tuning. Self-as-opponent is allowed at probability `1/pop_size`; not worth filtering out. `--opponent old_evo` and `--opponent old_evo2` modes swap in fixed opponents loaded from `saved_best_weights/`.
- **Rotating fitness seeds.** Each generation uses a fresh seed offset `(seed + gen + 1) * 9973` instead of a fixed seed range. Consequence: best-fitness is no longer monotone (a generation can land on a harder seed batch and the printed best dips). To recover a robust final winner, the script does a final held-out re-evaluation of the top-5 elites against the *last* population on a separate seed range, and writes that winner to `artifacts/best_weights_evo2_{tag}_{N}p.json`.

**`scripts/evolve_evo3.py`** — tunes the 25 constants of `Evo3AI`. Three opponent modes:

- **`--opponent vs_all` (default).** For each individual, averages fitness across six providers — one per previous bot type (Random, Heuristic, Adaptive, Hyper, HyperAdapt, Evo2). Every provider fills all 3 opponent seats with its class; the challenger's pooled win rate across all 6 × (5 charts × `games_per_chart`) games is the fitness. **6× longer per generation** than single-opponent modes; the point is to avoid overfitting to any one baseline. Evo2 opponents are loaded from `saved_best_weights/` if present (otherwise class defaults). Writes `artifacts/best_weights_evo3_vs_all_{N}p.json`.
- **`--opponent vs_evo2`.** Fixed Evo2AI opponents loaded from `saved_best_weights/` (lookup chain mirrors `--ai evo2`).
- **`--opponent self_play`.** Opponents sampled from the current Evo3 population each generation.
- Uses the same rotating-seed trick as evolve_evo2, and the same final held-out re-evaluation of the top-5 elites on the same provider distribution.

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
- **`--ai evolved` / `--ai evo2` / `--ai evo3` need a weights file in `saved_best_weights/`.** `evolved` exits with a clear error if nothing is found; `evo2` and `evo3` fall back to class defaults with a stderr warning. Weights in `artifacts/` are ignored by the CLI — you must copy them over by hand.
