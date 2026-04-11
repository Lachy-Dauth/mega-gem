# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` is the canonical onboarding doc and `research/RULES.md` is the canonical rules doc — read those for anything not covered here. This file is a curated cheat-sheet for things that are non-obvious or load-bearing for correctness.

## Repo layout

The canonical Python engine, AI zoo, tests, GA tuners, and checked-in weights all live under **`research/`**. The multiplayer stack (`server/`, `web/`) lives at the repo root. **Every research-side Python command (CLI, tests, GA) must be run from inside `research/`** — those scripts resolve paths like `saved_best_weights/` and `artifacts/` relative to the current working directory. **The multiplayer server runs from the repo root** — it resolves weights via `Path(__file__)` so it does not care about the CWD.

```
mega-gem/
├── server/             # FastAPI + WebSocket multiplayer server
├── web/                # browser client (served by server/) — quick play vs bots + multiplayer
├── requirements.txt    # fastapi / uvicorn / pydantic (server deps)
├── nixpacks.toml       # Railway build config
├── Procfile            # Railway start command
├── railway.json        # Railway service config + healthcheck
├── README.md
├── CLAUDE.md
└── research/           # Python engine + AI zoo + GA/RL tuners + tests
    ├── megagem/
    ├── scripts/
    │   ├── evolve/         # GA tuner (evo1..evo4)
    │   ├── rl/             # RL tuner (Evolution Strategies, evo4-focused)
    │   └── heatmap_pairwise.py
    ├── tests/
    ├── saved_best_weights/
    └── RULES.md
```

## Commands

```bash
cd research

# Tests (stdlib unittest, no pytest, 132 tests, sub-second)
python -m unittest discover
python -m unittest tests.test_heuristic -v
python -m unittest tests.test_evo2 -v
python -m unittest tests.test_evo3 -v
python -m unittest tests.test_heuristic.HyperAdaptiveSplitBidTest.test_invest_uses_invest_model_not_treasure_model

# Terminal CLI — must use `-m`, running the file directly breaks relative imports
python -m megagem
python -m megagem --all-ai --ai heuristic --seed 1 --quiet      # headless smoke test
python -m megagem --ai evolved --debug                          # opponent AI rationale + revealed hands
python -m megagem --ai evo3                                     # play vs the current champion

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

# RL tuner — Evolution Strategies (Salimans et al. 2017) on the same
# 35-weight Evo4 policy the GA tunes. Algorithmically distinct from
# scripts.evolve: single point θ moves via mirrored-sampling gradient
# estimates and an Adam update, no population/tournament/crossover.
python -m scripts.rl --ai evo4 --opponent vs_random             # fast smoke training
python -m scripts.rl --ai evo4 --opponent vs_all                # full run, mirrors GA defaults
python -m scripts.rl --ai evo4 --resume artifacts/rl_evo4_state_vs_all_4p.json
```

### Multiplayer server (from the repo root)

```bash
# From mega-gem/, NOT from inside research/.
pip install -r requirements.txt
uvicorn server.main:app --reload    # → http://127.0.0.1:8000/
```

`server/__init__.py` prepends `research/` to `sys.path` so
`from megagem.engine import play_round` resolves against the canonical
engine without needing an editable install. Weights are loaded from
`research/saved_best_weights/` via `Path(__file__)`, so the server
does not care about CWD.

### Weights workflow

The unified `scripts.evolve` writes its outputs to **`artifacts/`** (gitignored). The CLI, heatmap, and the opponent-lookup chains used by the four GA profiles read *only* from **`saved_best_weights/`** (checked in). Promoting a fresh winner is an explicit copy:

```bash
python -m scripts.evolve --ai evo3
cp artifacts/best_weights_evo3_vs_all_4p.json saved_best_weights/
```

The `saved_best_weights/` folder currently holds:
- `best_weights_evo1_vs_heuristic_4p.json` — HyperAdaptiveSplitAI (89% vs 3× HeuristicAI)
- `best_weights_evo2_vs_evo1_4p.json` — Evo2AI trained vs HyperAdaptiveSplitAI (69%) *(the `--opponent vs_all` retrain is in progress; promote the fresh file into `saved_best_weights/` when it lands)*
- `best_weights_evo3_vs_all_4p.json` — Evo3AI trained against all 6 prior bots (70% pooled)

## Dependencies

- Python engine, CLI, and tests: **stdlib only**.
- `scripts/evolve/` and `scripts/heatmap_pairwise.py`: `pip install matplotlib` (forced to `Agg` backend).
- `scripts/rl/`: **stdlib only for import, training, and tests.** matplotlib is only needed for `save_history_plot` and is *lazy-imported* inside that function, so `python -m unittest tests.test_rl_evo4` passes with no third-party deps. Install matplotlib only if you want the training-curve plot from `python -m scripts.rl`.
- `server/` multiplayer server: `fastapi`, `uvicorn[standard]`, `pydantic` (see `requirements.txt`).
- `web/` browser client: zero dependencies, vanilla JS.

## Architecture

### Single canonical engine

The repo has one game engine: `research/megagem/` (Python). The terminal CLI, test suite, GA tuners, and `server/` all use it. The browser client in `web/` is a **thin client** — it doesn't run any game logic. It talks to `server/` over REST + WebSocket and renders whatever state the server sends. Both quick-play (vs bots) and multiplayer games run the canonical Python engine on the server.

### Multiplayer server (`server/`)

- `server/__init__.py` prepends `research/` to `sys.path` — every other module in `server/` can then write `from megagem.engine import play_round` normally.
- `server/rooms.py` — `RoomManager` + `Room` + `Slot`. Rooms are keyed on a 5-char share code and held entirely in process memory. A Railway redeploy wipes active rooms — that's acceptable for the MVP.
- `server/session.py` — `GameSession`. The canonical engine is **synchronous**, so each room spawns a dedicated `threading.Thread` that calls `play_round` in a loop. Humans are wired up via `RemotePlayer` (`server/remote_player.py`), whose `choose_bid` and `choose_gem_to_reveal` block on thread-safe `queue.Queue` instances. The WS handler does `queue.put(amount)` when a bid arrives; the game thread unblocks and proceeds. Broadcasts go the other way via `asyncio.run_coroutine_threadsafe(room.broadcast(...), self.loop)`.
- **Before `play_round` runs**, the session peeks at `state.auction_deck[-1]` and broadcasts a `round_start` message so human clients can render the card while the engine is about to start blocking on their bid.
- **Pending-request tracking.** `GameSession._pending_requests[idx]` stores the last unanswered `request_bid` / `request_reveal` per seat. On WS reconnect, `server/main.py`'s welcome handler re-sends the current state *and* the pending request, so a mid-game page refresh picks up exactly where it left off instead of leaving the player staring at a frozen board while the engine thread is still blocked on their queue.
- `server/protocol.py` — single source of truth for JSON serialization of engine objects. The engine's `GameState` is a dataclass full of `Counter[Color]` / `AuctionCard` / `MissionCard` references; nothing in `server.session` should poke at these directly, it should always go through `serialize_state`, `serialize_auction`, etc.
- `server/ai_factory.py` — a trimmed mirror of `research/megagem/__main__.py`'s `AI_FACTORIES` dict. Only the playable kinds (`random`, `heuristic`, `evo4`) are wired up here — earlier evo generations still exist in `research/` for the GA/CLI, but the multiplayer server intentionally only exposes the current champion so players aren't overwhelmed with near-duplicate difficulty levels. Weights are loaded from `research/saved_best_weights/` via an absolute `Path(__file__)`, and missing weights fall back to class defaults with no error (unlike the CLI's `evolved` factory, which raises). The server should *never crash* because a weights file is missing.

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
6. `Evo3AI` (`megagem/players/evo3.py`) — **current champion.** Adds an opponent-pricing signal on top of the Evo2 feature set. After every round the engine calls an optional `observe_round(public_state, my_idx, result)` hook on each player; Evo3 uses it to log `(category, max_opp_bid − baseline)` per auction. Every head reads two new features — a 4×-weighted running `mean_delta` and `std_delta` over the most recent auctions in the same category (treasure/invest/loan) — defaulting to `(0, 1)` before any history exists. **Critical detail:** `baseline` is the bid Evo3 *would* have made with the default `(0, 1)` deltas, not its actual bid. Using the actual bid creates a feedback loop where the signal depends on Evo3's own learned response; caching the default-delta bid on `self._last_default_bid` in `choose_bid` and reading it back in `observe_round` breaks that. 25 weights total (treasure 9 + invest 8 + loan 8); tuned by `python -m scripts.evolve --ai evo3`.

### GA tuners — `scripts/evolve/`

One unified package, four AI profiles, eight opponent modes. The CLI is `python -m scripts.evolve --ai {evo1,evo2,evo3,evo4} [--opponent MODE]`. Same loop, same fitness recipe, same output filename layout — only the per-AI weight count, mutation sigma, and lookup chain change between profiles.

**Package layout** (under `research/scripts/evolve/`):

- `profiles.py` — `AIProfile` dataclass + the registry of four profiles. Each profile only carries its `ai_class`, `num_weights`, and `mutation_sigma`/`mutation_clip`. Everything else is uniform: `flatten_defaults` lives as a classmethod on the AI class itself (the single source of truth for the genome layout — it's the inverse of `from_weights`), and the lookup chain is shared across all profiles via `opponents.candidate_filenames(profile_key, num_players)`. **Individual #0 of every GA run is loaded from `saved_best_weights/`** via that shared chain — the same chain used for opponents in `vs_all` / `vs_evoK` modes. That means each fresh run starts from the current champion instead of a hardcoded constant; re-running `python -m scripts.evolve --ai evo3` iteratively refines whatever weights are already checked in. If no file exists yet, the GA falls back to `profile.flatten_defaults()` → `ai_class.flatten_defaults()` (the class's hardcoded `DEFAULT_*` constants).
- `opponents.py` — the eight uniform opponent modes (`vs_all`, `vs_random`, `vs_heuristic`, `vs_evo1`, `vs_evo2`, `vs_evo3`, `vs_evo4`, `self_play`), the `build_mode_providers` dispatcher, and the shared `candidate_filenames` lookup chain used by every profile and mirrored in `megagem/__main__.py` + `server/ai_factory.py`. `vs_all` returns Heuristic + every evo profile *except* the challenger's own (Random is intentionally excluded — every tuned bot trivially beats it and those free wins wash out the signal from harder opponents); `vs_evoK` loads frozen weights from `saved_best_weights/` via the shared chain.
- `ga.py` — the GA loop, fitness evaluation, progress bar, plot/json output. Fully generic over `AIProfile`. One code path for every profile/mode combination.
- `__main__.py` — argparse CLI. `--ai` is required; `--opponent` defaults to `vs_all`. Output files land in `artifacts/best_weights_{key}_{tag}_{N}p.json` and `artifacts/evolve_{key}_history_{tag}_{N}p.png`.

**Profile sizes** (mutation sigma / clip in parens):

- `evo1` = HyperAdaptiveSplitAI, 18 weights (0.15 / 2.0 — tighter than the rest because the original hyper_adaptive GA was tuned for these magnitudes).
- `evo2` = Evo2AI, 19 weights (0.05 / 5.0).
- `evo3` = Evo3AI, 25 weights (0.05 / 5.0).
- `evo4` = Evo4AI, 35 weights (0.05 / 5.0).

**Fitness strategy** (uniform across all profiles): rotating per-generation seeds `seed_offset = (seed + gen + 1) * 9973` so each generation samples a fresh slice of seed space, then a held-out re-eval of the top-5 elites against the same opponent distribution at the end. The held-out winner — not the per-gen best — is what gets written to disk. Per-generation best/mean dips are expected because the seeds shift; the held-out re-eval is the canonical "did it generalise?" check. There is no fitness cache (correct only under fixed seeds, which the unified loop never uses).

**vs_all is the default** for every profile. It pools win rate across Heuristic + every other evo class (loaded from `saved_best_weights/`, falling back to the AI class's `flatten_defaults()` classmethod). Random is intentionally excluded because every tuned bot beats it near-100%, so its slate would contribute near-perfect win rates that wash out the signal from the harder opponents. Each provider gets `len(CHARTS) × games_per_chart` games on a non-overlapping seed slice (offset by `prov_idx * 101`) so a challenger can't get lucky by hitting the same seed against every opponent.

**Output filenames follow one uniform formula**: `best_weights_{profile_key}_{tag}_{num_players}p.json`, where `tag` is one of `vs_all`, `vs_random`, `vs_heuristic`, `vs_evo1..4`, `self`. The same formula drives the shared `candidate_filenames` lookup chain in `opponents.py`, which is mirrored byte-for-byte in `research/megagem/__main__.py` and `server/ai_factory.py`. There are no legacy filename fallbacks — the previously-checked-in `best_weights_4p.json` and `best_weights_evo2_vs_old_4p.json` were renamed to `best_weights_evo1_vs_heuristic_4p.json` and `best_weights_evo2_vs_evo1_4p.json` to fit the uniform scheme.

### RL tuner — `scripts/rl/`

A second, algorithmically distinct optimizer targeting the same 35-weight Evo4 policy as the GA. **OpenAI-style Evolution Strategies** (Salimans et al. 2017) with mirrored sampling, centered rank shaping, and an Adam outer-loop update. It is not a replacement for `scripts.evolve` — it's a parallel path for experimenting with parameter-space policy-gradient RL on the existing engine.

**Why ES and not REINFORCE**: the engine is episodic (only terminal return from `score_game`, no step reward), `Evo4.choose_bid` is deterministic per state, and ES slots into the existing fork-pool worker pattern with zero changes to `Player` or `engine.py`. REINFORCE would require a new stochastic Evo4 subclass, a trajectory-buffer hook on `Player`, and log-prob math that breaks near the bid cap — not worth the surface area for what would still collapse to Monte Carlo anyway.

**Package layout** (under `research/scripts/rl/`):

- `optim.py` — stdlib `Adam` dataclass (`m`, `v`, `t`, `step`, `state_dict`/`from_state`). **Ascent direction** (`θ + α·m̂/(√v̂+ε)`), not descent, because ES maximises reward. `step` does NOT mutate its input `theta`, which the trainer relies on for its pre-update snapshot.
- `fitness.py` — `_game_reward` (normalized score margin `(mine − max_opp) / max(1, |mine| + |max_opp|)`), `rank_transform` (centered Salimans shaping in `[-0.5, +0.5]`), `evaluate_perturbations` (batch), `evaluate_theta` (held-out). Fork-pool fan-out mirrors `scripts.evolve.ga` byte-for-byte: `_WORKER_STATE` populated in parent → inherited via `fork` → cleared in `finally`. **Deterministic at any worker count** via a pre-allocated `rewards[pert_idx][game_idx_in_pert]` 2D accumulator that is summed post-pool in loop order — do NOT fold the sum into the worker callback, that reintroduces pool-order dependence.
- `trainer.py` — `run_es` loop, `ESResult` dataclass, `sample_epsilons` (mirrored pair layout `[ε₀, -ε₀, ε₁, -ε₁, …]`), `_clip`, `save_history_plot` (lazy-imports matplotlib), `save_best_weights` (GA-compatible JSON), `save_resume_state` / `load_resume_state` (`__tuple__` sentinels wrap `random.getstate()` tuples for JSON). The outer loop: sample epsilons → build `θ±σε` perturbations → evaluate → rank-shape → gradient `g[k] = (1/(N·σ)) · Σᵢ Rᵢ·εᵢ[k]` → Adam → θ-clip → held-out θ-eval.
- `__main__.py` — argparse CLI (`python -m scripts.rl --ai evo4 --opponent vs_all`). Defaults match the GA's invocation style (`--population 48`, `--generations 30`, `--games-per-chart 10`) plus ES-specific knobs (`--sigma 0.10`, `--lr 0.03`, `--theta-clip 5.0`).

**Seed layout**: per-generation training seeds use `(seed + gen + 1) * 9973` — identical to the GA's rotation. Held-out θ-eval uses `(seed + gen + 1000) * 9973` so the plotted training curve is decorrelated from the seeds the gradient estimate saw. Matches the GA's `(seed + generations + 100) * 9973` held-out convention in spirit, not byte-for-byte — different scale so ES and GA runs can coexist in the same artifact directory without reading each other's numbers.

**Output filenames use an `rl_` infix** so they never collide with the GA's `candidate_filenames` lookup chain in `opponents.py`:

```
artifacts/best_weights_evo4_rl_{tag}_{N}p.json    # GA-compatible shape + rl_config
artifacts/rl_evo4_history_{tag}_{N}p.png          # held-out reward + win-rate curve
artifacts/rl_evo4_state_{tag}_{N}p.json           # θ, Adam (m/v/t), RNG state — for --resume
```

Promotion is an explicit `cp` into `saved_best_weights/` *without* the `rl_` infix — the CLI / heatmap lookup chain won't match `rl_*` names, which is deliberate (no silent auto-promotion). The weights file's JSON layout mirrors the GA's output (`fitness`, `generation`, `weights`, plus `rl_config` instead of `ga_config`) so once renamed, it's a drop-in replacement.

**Seeding θ₀**: same as the GA — `scripts.evolve.opponents.load_profile_weights(profile, num_players)` via the shared `candidate_filenames` chain. Re-running `python -m scripts.rl --ai evo4` iteratively refines the current `saved_best_weights/best_weights_evo4_vs_all_4p.json` champion; if no file exists, falls back to `Evo4AI.flatten_defaults()`.

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
- **`scripts.rl` output files have an `rl_` infix that blocks auto-promotion.** `best_weights_evo4_rl_vs_all_4p.json` does NOT match the CLI / heatmap / GA `candidate_filenames` chain, on purpose — RL outputs never silently compete with the GA champion. Promote an RL winner by `cp`-ing the file into `saved_best_weights/` with the infix stripped (`best_weights_evo4_vs_all_4p.json`). The `rl_config` field inside the JSON is preserved for provenance.
- **`scripts.rl` determinism at `workers>1` depends on a pre-allocated 2D accumulator.** `fitness.evaluate_perturbations` sums `rewards[pert_idx][game_idx]` rows *after* the pool drains, in loop order. Folding the sum into `_worker_play` would make the mean float-sum order-dependent on completion order. The `touched[][]` validation exists to catch a silently-dropped task before it biases the mean toward 0. The `DeterminismTest` in `tests/test_rl_evo4.py` runs with `workers=1`; the parallel path is deterministic *by construction* from the same accumulator pattern and is exercised by the CLI smoke run, not by a unit test.
