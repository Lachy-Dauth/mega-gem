# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` is the canonical onboarding doc and `research/RULES.md` is the canonical rules doc — read those for anything not covered here. This file is a curated cheat-sheet for things that are non-obvious or load-bearing for correctness.

## Repo layout

This is a research repo. The canonical Python engine, AI zoo, tests, GA tuners, and checked-in weights all live under **`research/`**. The static single-player web client (`web/`, deployed to GitHub Pages) lives at the repo root. **Every research-side Python command (CLI, tests, GA) must be run from inside `research/`**: those scripts resolve paths like `saved_best_weights/` and `artifacts/` relative to the current working directory. **The site build (`python web/build.py`) runs from the repo root**; it resolves paths via `Path(__file__)`.

```
mega-gem/
├── web/                # static client: runs research/megagem in Pyodide (GitHub Pages)
├── .github/workflows/  # pages.yml: tests on push/PR, deploys Pages from main
├── README.md
├── CLAUDE.md
└── research/           # Python engine + AI zoo + GA tuners + tests
    ├── megagem/
    ├── scripts/
    ├── tests/
    ├── saved_best_weights/
    └── RULES.md
```

## Commands

```bash
cd research

# Tests (stdlib unittest, no pytest, 132 tests, ~2s)
python -m unittest discover
python -m unittest tests.test_heuristic -v
python -m unittest tests.test_evo2 -v
python -m unittest tests.test_evo3 -v
python -m unittest tests.test_heuristic.HyperAdaptiveSplitBidTest.test_invest_uses_invest_model_not_treasure_model

# Terminal CLI — must use `-m`, running the file directly breaks relative imports
python -m megagem
python -m megagem --all-ai --ai heuristic --seed 1 --quiet      # headless smoke test
python -m megagem --ai evolved --debug                          # opponent AI rationale + revealed hands
python -m megagem --ai search                                   # play vs the current champion (SearchAI)
python -m megagem --ai evo4                                     # play vs the strongest GA bot

# GA tuners + heatmap (need matplotlib)
# One unified entry point for all four evolvable bots — `evo1` is the
# HyperAdaptiveSplitAI; pick a profile via --ai and an opponent via
# --opponent. Defaults: --opponent vs_all (5-way pooled fitness).
python -m scripts.evolve --ai evo1 --opponent vs_heuristic      # tunes HyperAdaptiveSplitAI vs 3× HeuristicAI
python -m scripts.evolve --ai evo2                              # tunes Evo2AI vs_all (Heuristic + evo1/3/4)
python -m scripts.evolve --ai evo2 --opponent self_play         # tunes Evo2AI via self-play
python -m scripts.evolve --ai evo3                              # tunes Evo3AI vs_all
python -m scripts.evolve --ai evo4 --opponent vs_evo3           # train evo4 against frozen evo3
python -m scripts.heatmap_pairwise                              # requires saved_best_weights/*.json
python -m scripts.generate_heatmap_json                         # web/heatmap_data.json incl. SearchAI (~1.5h on 4 cores, resumable)

```

### Web client (from the repo root)

```bash
# From mega-gem/, NOT from inside research/.
python web/build.py                 # → _site/ (gitignored): web files + megagem.zip
python -m http.server -d _site      # → http://127.0.0.1:8000/
# Offline: unpack `npm pack pyodide@0.29.5` into _site/pyodide/ and open /?pyodide=/pyodide/
```

`megagem.zip` holds `research/megagem/**/*.py` + `research/saved_best_weights/*.json`, so **rebuild after any engine/AI change or weights promotion**. CI (`.github/workflows/pages.yml`) does this on every push to `main`.

### Weights workflow

The unified `scripts.evolve` writes its outputs to **`artifacts/`** (gitignored). The CLI, heatmap, and the opponent-lookup chains used by the four GA profiles read *only* from **`saved_best_weights/`** (checked in). Promoting a fresh winner is an explicit copy:

```bash
python -m scripts.evolve --ai evo3
cp artifacts/best_weights_evo3_vs_all_4p.json saved_best_weights/
```

The `saved_best_weights/` folder currently holds one `vs_all` 4p file per profile: `best_weights_evo{1,2,3,4}_vs_all_4p.json`. The site reads the same files (packed into `megagem.zip`).

## Dependencies

- Python engine, CLI, and tests: **stdlib only**.
- `scripts/evolve/` and `scripts/heatmap_pairwise.py`: `pip install matplotlib` (forced to `Agg` backend).
- `web/` browser client: vanilla JS; loads Pyodide (`pyodide@0.29.5`, Python 3.13) from jsDelivr at runtime. `web/build.py` is stdlib only.

## Architecture

### Single canonical engine

The repo has one game engine: `research/megagem/` (Python). The terminal CLI, test suite, GA tuners, and the web client all use it. **There is no JS port of the rules**: `web/engine-worker.js` boots Pyodide in a Web Worker, unpacks `megagem.zip`, and drives `megagem.browser.BrowserSession`. `web/app.js` only renders the JSON views the session returns and posts back bids/reveals.

### Web client (`web/` + `research/megagem/browser.py`)

- `megagem/browser.py` is a frontend adapter like `__main__.py`; the engine knows nothing about it. It owns `AI_FACTORIES` for the site (all seven bot kinds, CLI key names), JSON serialization (`serialize_state` etc.; spectator view = `viewer_idx=None` shows every hand), and `BrowserSession` with a step API: `start()` / `bid(n)` / `reveal(color)` / `step()` (spectator) → view dict with `phase in {bid, reveal, ready, game_over}`.
- **Snapshot-and-replay for reveals.** The engine is synchronous and the page can't block. The human's bid is preloaded before `play_round` (bids are simultaneous). If the human wins, `_BrowserHuman.choose_gem_to_reveal` raises `_NeedReveal`; the session had deep-copied `(state, rng)` before the round, so on `reveal(color)` it restores the snapshot and replays the round with the answer preloaded. Every AI's RNG/caches and the tie-break RNG live in the snapshot, so the replay is identical. `tests/test_browser.py::ReplayTest` pins this against a direct engine run; keep it passing if you touch the engine's round flow or add AI state outside the player object.
- **Bot reasoning** (`explain=True`) calls each AI's `explain_bid` on a separate deep copy, so it can never perturb the game.
- **Weights lookup** uses the uniform `candidate_filenames` chain, then falls back to the **4p** files (all checked-in weights are 4p), then class defaults. The site never errors on missing weights.
- `web/build.py` stamps a content hash over `__MEGAGEM_BUILD__` in `.html`/`.js` for cache-busting. Use relative URLs everywhere: Pages serves the site under `/mega-gem/`.

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
4. `HyperAdaptiveSplitAI(HeuristicAI)` — pre-Evo2 champion. Inherits reveal logic from `HeuristicAI`; overrides bidding with three independent `_BidModel` heads (`treasure`, `invest`, `loan`) over hypergeometric value estimates and a 5-feature linear discount. 18 tunable constants total = 3 × (1 bias + 5 weights). Construct via class defaults, explicit `_BidModel`s, or `HyperAdaptiveSplitAI.from_weights(name, [18 floats])`. The GA produces individuals with the flat-vector form.
5. `Evo2AI` (`megagem/players/evo2.py`) — clean-slate evolved AI. Drops `_reserve_for_future`, replaces the `auctions_left/25` progress proxy with an exact closed-form `_expected_rounds_remaining` (multivariate hypergeometric over the auction-deck multiset), drops the `cash_ratio` features in favour of raw integer `my_coins / avg_opp_coins / top_opp_coins`, and gives the treasure head two new per-card features — `ev` and `std` of the prize value — derived from the same hypergeometric distribution. Treasure EV also adds a `_mission_probability_delta` term: a per-mission `(P(I win | I take the gems) − P(I win | likely opponent does)) × mission.coins`, on top of the existing hard/soft mission bonuses. **Heads output the bid in coins directly** (not a discount fraction multiplied by EV/amount): `bid = bias + Σ wᵢ·featureᵢ`, clamped to `[0, cap]` once at `choose_bid`. This frees the GA from the implicit "scale by EV" coupling the discount form baked in. 19 weights (treasure 7 + invest 6 + loan 6); tuned by `python -m scripts.evolve --ai evo2`.
6. `Evo3AI` (`megagem/players/evo3.py`) — adds an opponent-pricing signal on top of the Evo2 feature set. After every round the engine calls an optional `observe_round(public_state, my_idx, result)` hook on each player; Evo3 uses it to log `(category, max_opp_bid − baseline)` per auction. Every head reads two new features — a 4×-weighted running `mean_delta` and `std_delta` over the most recent auctions in the same category (treasure/invest/loan) — defaulting to `(0, 1)` before any history exists. **Critical detail:** `baseline` is the bid Evo3 *would* have made with the default `(0, 1)` deltas, not its actual bid. Using the actual bid creates a feedback loop where the signal depends on Evo3's own learned response; caching the default-delta bid on `self._last_default_bid` in `choose_bid` and reading it back in `observe_round` breaks that. 25 weights total (treasure 9 + invest 8 + loan 8); tuned by `python -m scripts.evolve --ai evo3`.
7. `Evo4AI` (`megagem/players/evo4.py`) — Evo3 plus a per-colour bid signal and an evolvable internal opponent-bid predictor. 35 weights. The strongest GA bot.
8. `SearchAI` (`megagem/players/search.py`) — **current champion.** Not a formula: determinized Monte Carlo search. Per bid it samples worlds (the unseen gem pool dealt into opponents' hands + deck, auction order shuffled), predicts each opponent's bid with a model bot → a *price-to-win threshold* per world, rolls out "lose" vs "win at threshold" to game end, and picks the bid maximising the mean sigmoid of final margin. Layers on top: `identify` (bank of zoo models scored per seat by bid likelihood), `shadow` (per-(seat, model) copies fed every real round so Evo3/Evo4 history features match), `infer` (importance-weight worlds by the likelihood of every past treasure bid, via per-hand bid tables), `calibrate` (gated: a seat whose bids fall outside its model's any-hand range ≥2 of its last 8 bids gets its recent errors added to predictions — this is what stops lowballing humans exploiting it; ungated it cost ~3 points vs exact bots), `rollout_snipe` (own future turns pay only the threshold). Build it with `SearchAI.standard(name, seed=, weights={"evo1".."evo4": vec})`; CLI/browser pass loaded weights. **Invariants, each pinned by a test in `tests/test_search.py`:** (a) never read hidden state, only public info + `unseen_pool` (`NoPeekingTest`); (b) rollouts must replay exactly the opponent bids used to compute the threshold, otherwise the "win" branch can lose (`BranchConsistencyTest`, a real past bug); (c) bookkeeping draws use `_aux_rng` so `explain_bid` (which replays the search on a saved RNG state) matches the real bid. It's slow by bot standards (~0.2s/bid CPython, ~3× that in Pyodide).

### GA tuners — `scripts/evolve/`

One unified package, four AI profiles, eight opponent modes. The CLI is `python -m scripts.evolve --ai {evo1,evo2,evo3,evo4} [--opponent MODE]`. Same loop, same fitness recipe, same output filename layout — only the per-AI weight count, mutation sigma, and lookup chain change between profiles.

**Package layout** (under `research/scripts/evolve/`):

- `profiles.py` — `AIProfile` dataclass + the registry of four profiles. Each profile only carries its `ai_class`, `num_weights`, and `mutation_sigma`/`mutation_clip`. Everything else is uniform: `flatten_defaults` lives as a classmethod on the AI class itself (the single source of truth for the genome layout — it's the inverse of `from_weights`), and the lookup chain is shared across all profiles via `opponents.candidate_filenames(profile_key, num_players)`. **Individual #0 of every GA run is loaded from `saved_best_weights/`** via that shared chain — the same chain used for opponents in `vs_all` / `vs_evoK` modes. That means each fresh run starts from the current champion instead of a hardcoded constant; re-running `python -m scripts.evolve --ai evo3` iteratively refines whatever weights are already checked in. If no file exists yet, the GA falls back to `profile.flatten_defaults()` → `ai_class.flatten_defaults()` (the class's hardcoded `DEFAULT_*` constants).
- `opponents.py` — the eight uniform opponent modes (`vs_all`, `vs_random`, `vs_heuristic`, `vs_evo1`, `vs_evo2`, `vs_evo3`, `vs_evo4`, `self_play`), the `build_mode_providers` dispatcher, and the shared `candidate_filenames` lookup chain used by every profile and mirrored in `megagem/__main__.py` + `megagem/browser.py`. `vs_all` returns Heuristic + every evo profile *except* the challenger's own (Random is intentionally excluded — every tuned bot trivially beats it and those free wins wash out the signal from harder opponents); `vs_evoK` loads frozen weights from `saved_best_weights/` via the shared chain.
- `ga.py` — the GA loop, fitness evaluation, progress bar, plot/json output. Fully generic over `AIProfile`. One code path for every profile/mode combination.
- `__main__.py` — argparse CLI. `--ai` is required; `--opponent` defaults to `vs_all`. Output files land in `artifacts/best_weights_{key}_{tag}_{N}p.json` and `artifacts/evolve_{key}_history_{tag}_{N}p.png`.

**Profile sizes** (mutation sigma / clip in parens):

- `evo1` = HyperAdaptiveSplitAI, 18 weights (0.15 / 2.0 — tighter than the rest because the original hyper_adaptive GA was tuned for these magnitudes).
- `evo2` = Evo2AI, 19 weights (0.05 / 5.0).
- `evo3` = Evo3AI, 25 weights (0.05 / 5.0).
- `evo4` = Evo4AI, 35 weights (0.05 / 5.0).

**Fitness strategy** (uniform across all profiles): rotating per-generation seeds `seed_offset = (seed + gen + 1) * 9973` so each generation samples a fresh slice of seed space, then a held-out re-eval of the top-5 elites against the same opponent distribution at the end. The held-out winner — not the per-gen best — is what gets written to disk. Per-generation best/mean dips are expected because the seeds shift; the held-out re-eval is the canonical "did it generalise?" check. There is no fitness cache (correct only under fixed seeds, which the unified loop never uses).

**vs_all is the default** for every profile. It pools win rate across Heuristic + every other evo class (loaded from `saved_best_weights/`, falling back to the AI class's `flatten_defaults()` classmethod). Random is intentionally excluded because every tuned bot beats it near-100%, so its slate would contribute near-perfect win rates that wash out the signal from the harder opponents. Each provider gets `len(CHARTS) × games_per_chart` games on a non-overlapping seed slice (offset by `prov_idx * 101`) so a challenger can't get lucky by hitting the same seed against every opponent.

**Output filenames follow one uniform formula**: `best_weights_{profile_key}_{tag}_{num_players}p.json`, where `tag` is one of `vs_all`, `vs_random`, `vs_heuristic`, `vs_evo1..4`, `self`. The same formula drives the shared `candidate_filenames` lookup chain in `opponents.py`, which is mirrored in `research/megagem/__main__.py` and `research/megagem/browser.py` (the latter adds a 4p fallback). There are no legacy filename fallbacks — the previously-checked-in `best_weights_4p.json` and `best_weights_evo2_vs_old_4p.json` were renamed to `best_weights_evo1_vs_heuristic_4p.json` and `best_weights_evo2_vs_evo1_4p.json` to fit the uniform scheme.

### Heatmap (`scripts/heatmap_pairwise.py`)

- Default `SEED_START = 200` is **held out** from the GA's training seeds (0..9). If you change the GA's training seed range, update `SEED_START` to stay above it or the generalisation check is meaningless.

## Testing conventions

- `tests/test_heuristic.py` is intentionally large. Each shared helper gets a focused unit test before the AI itself is exercised; head-to-head tests at the end run ~60 games per pair and finish in well under a second.
- When adding a new AI, follow the existing `HeadToHeadTest` pattern: at minimum, prove it beats `RandomAI` at ≥60% on chart A. If it should also appear in the all-vs-all heatmap, register it in `scripts/heatmap_pairwise.py`'s `make_factories()`.

## Gotchas

- **Always invoke as `python -m megagem`.** `python megagem/__main__.py` breaks relative imports.
- **Reveal-a-gem is mandatory.** If `choose_gem_to_reveal` returns a gem not in hand, the engine substitutes a random one — that's a safety net for buggy AIs, not a feature to rely on.
- **`LoanCard` bids exceed your coins.** `max_legal_bid` returns `coins + loan_amount` because the loan is conceptually paid first. Always route AI bids through `clamp_bid` rather than trusting raw output.
- **`--ai evolved` / `--ai evo2` / `--ai evo3` / `--ai evo4` need a weights file in `saved_best_weights/`.** `evolved` exits with a clear error if nothing is found; `evo2`/`evo3`/`evo4` fall back to class defaults with a stderr warning. Weights in `artifacts/` are ignored by the CLI — you must copy them over by hand.
- **`scripts.evolve` and the CLI use different profile keys.** The GA package uses `evo1`/`evo2`/`evo3`/`evo4` (`--ai evo1` runs the HyperAdaptiveSplitAI tuner). The terminal CLI for *playing* still uses the original key `evolved` (`python -m megagem --ai evolved`). The class itself is unchanged — only the GA-side label is uniformised so file paths align with `evo2`/`evo3`/`evo4`.
