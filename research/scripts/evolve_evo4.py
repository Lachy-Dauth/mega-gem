"""Genetic-algorithm tuner for ``Evo4AI``'s 26 weights.

Run from the project root::

    python -m scripts.evolve_evo4                         # default: vs all 4 previous bots
    python -m scripts.evolve_evo4 --opponent vs_evo3      # vs Evo3 only
    python -m scripts.evolve_evo4 --opponent self_play    # self-play

Outputs land in ``artifacts/`` (gitignored):

* ``evolve_evo4_history_{tag}_{N}p.png`` — best/mean fitness curve.
* ``best_weights_evo4_{tag}_{N}p.json`` — winning genome + GA config.

where ``{tag}`` is one of:

* ``vs_all``   — averaged fitness across all four previous bots (default)
* ``vs_evo3``  — fitness vs a frozen Evo3 snapshot only
* ``self``     — self-play within the current population

The CLI's ``--ai evo4`` factory checks ``vs_all`` first, then ``vs_evo3``,
then ``self``.

Three opponent modes:

1. **``--opponent vs_all`` (default).** For each individual, run
   ``games_per_chart`` games on each of the five value charts against
   ``num_players − 1`` copies of each of four previous bots
   (:class:`RandomAI`, :class:`HeuristicAI`, :class:`Evo2AI`,
   :class:`Evo3AI`). Fitness is the overall win rate across all four
   opponent types — equivalently, the average of the four per-opponent
   win rates since every type gets the same number of games. This
   takes **4× longer** per generation than the single-opponent modes
   but produces a challenger that doesn't overfit to any one baseline.

2. **``--opponent vs_evo3``.** Opponents are fixed: ``num_players − 1``
   :class:`Evo3AI` instances loaded from the best available Evo3
   weights file (lookup chain mirrors ``--ai evo3``). Fall-back is
   Evo3's class defaults if no weights file exists.

3. **``--opponent self_play``.** Opponents are sampled each generation
   from the current Evo4 population.

Like ``evolve_evo3``, fitness uses **rotating seeds** — each generation
evaluates on a fresh, non-overlapping seed range — so best-fitness per
generation is not monotone, and the final saved winner comes from a
held-out re-evaluation of the top-5 elites of the last generation on
the same opponent distribution as training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

# Headless backend so this runs over SSH / inside CI without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from megagem.engine import is_game_over, play_round, score_game, setup_game
from megagem.players import (
    Evo2AI,
    Evo3AI,
    Evo4AI,
    HeuristicAI,
    Player,
    RandomAI,
)


# An OpponentProvider builds the opponent slate for one game. It takes the
# zero-based game-slot index (within the current generation) and the game
# seed, and returns a list of ``num_players − 1`` ready-to-play Players.
OpponentProvider = Callable[[int, int], list[Player]]


# --- Fitness ----------------------------------------------------------------


def _play_one_game(
    challenger_weights: list[float],
    opponents: list[Player],
    chart: str,
    seed: int,
) -> bool:
    """Run one full game and return True iff the challenger won outright."""
    challenger = Evo4AI.from_weights("Evo4", challenger_weights, seed=seed * 7)
    players = [challenger] + opponents
    state = setup_game(players, chart=chart, seed=seed)
    game_rng = random.Random(seed)
    while not is_game_over(state):
        play_round(state, rng=game_rng)
    scores = score_game(state)
    return scores[0]["total"] > max(s["total"] for s in scores[1:])


def _make_self_play_provider(
    population: list[list[float]],
    *,
    num_players: int,
    sample_rng: random.Random,
    n_games: int,
) -> OpponentProvider:
    """Build a provider that samples opponents from ``population``.

    The opponent indices for each game slot are pre-sampled *once* per
    generation and reused for every individual — same constraint as
    :mod:`scripts.evolve_evo3`'s self-play provider.
    """
    pop_size = len(population)
    opp_indices_per_slot: list[list[int]] = [
        [sample_rng.randrange(pop_size) for _ in range(num_players - 1)]
        for _ in range(n_games)
    ]

    def provider(slot: int, game_seed: int) -> list[Player]:
        return [
            Evo4AI.from_weights(
                f"O{k + 1}",
                population[opp_indices_per_slot[slot][k]],
                seed=game_seed + k + 1,
            )
            for k in range(num_players - 1)
        ]

    return provider


def _make_evo3_opponent_provider(
    evo3_weights: list[float],
    *,
    num_players: int,
) -> OpponentProvider:
    """Build a provider that returns ``num_players − 1`` Evo3AIs from fixed weights.

    Opponents are deterministic across generations — every game runs
    against the same frozen Evo3 snapshot. Seeds vary per game so the
    Evo3 RNG (used for its own internal tie-breaks) decorrelates.
    """
    def provider(slot: int, game_seed: int) -> list[Player]:
        return [
            Evo3AI.from_weights(
                f"E3_{k + 1}",
                evo3_weights,
                seed=game_seed + k + 1,
            )
            for k in range(num_players - 1)
        ]

    return provider


def _make_vs_all_providers(
    num_players: int,
) -> list[tuple[str, OpponentProvider]]:
    """Build one provider per previous-bot type for the ``vs_all`` mode.

    Returns a list of ``(name, provider)`` pairs, one per opponent
    class. Each provider fills every opponent seat with ``num_players −
    1`` copies of the same class — matching the usual "1 challenger vs
    3× same-class opponents" setup used by the heatmap. Evo2 and Evo3
    use the best weights on disk (falling back to class defaults if
    none exist); the other two opponents (Random, Heuristic) are
    constructed directly from their class defaults since they have no
    tuneable weights.
    """
    evo2_weights, evo2_path = _load_evo2_weights(num_players)
    print(f"vs_all: Evo2 opponents loaded from {evo2_path}")
    evo3_weights, evo3_path = _load_evo3_weights(num_players)
    print(f"vs_all: Evo3 opponents loaded from {evo3_path}")

    def _simple_provider(factory):
        def provider(slot: int, game_seed: int) -> list[Player]:
            return [
                factory(f"O{k + 1}", game_seed + k + 1)
                for k in range(num_players - 1)
            ]
        return provider

    def _evo2_provider(slot: int, game_seed: int) -> list[Player]:
        return [
            Evo2AI.from_weights(
                f"E2_{k + 1}", evo2_weights, seed=game_seed + k + 1
            )
            for k in range(num_players - 1)
        ]

    def _evo3_provider(slot: int, game_seed: int) -> list[Player]:
        return [
            Evo3AI.from_weights(
                f"E3_{k + 1}", evo3_weights, seed=game_seed + k + 1
            )
            for k in range(num_players - 1)
        ]

    return [
        ("Random",     _simple_provider(lambda n, s: RandomAI(n, seed=s))),
        ("Heuristic",  _simple_provider(lambda n, s: HeuristicAI(n, seed=s))),
        ("Evo2",       _evo2_provider),
        ("Evo3",       _evo3_provider),
    ]


def _load_evo2_weights(num_players: int) -> tuple[list[float], Path]:
    """Load existing Evo2 weights for ``num_players`` to train against.

    Lookup chain mirrors ``megagem.__main__._evo2_factory`` so "Evo2"
    means whatever ``--ai evo2`` would currently pick — the strongest
    Evo2 snapshot on disk. If none exists, falls back to the class
    defaults baked into ``Evo2AI``; we flatten those into a 19-element
    vector and return a sentinel path so the caller can log "defaults".
    """
    candidates = [
        Path("saved_best_weights") / f"best_weights_evo2_vs_all_{num_players}p.json",
        Path("saved_best_weights") / f"best_weights_evo2_vs_old_evo2_{num_players}p.json",
        Path("saved_best_weights") / f"best_weights_evo2_vs_old_{num_players}p.json",
        Path("saved_best_weights") / f"best_weights_evo2_self_{num_players}p.json",
        Path("saved_best_weights") / f"best_weights_evo2_{num_players}p.json",
        Path("saved_best_weights") / "best_weights_evo2.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            weights = data["weights"]
            if len(weights) != Evo2AI.NUM_WEIGHTS:
                raise SystemExit(
                    f"{path}: expected {Evo2AI.NUM_WEIGHTS} weights, "
                    f"got {len(weights)}"
                )
            return weights, path

    # Fall back to Evo2 class defaults. Flatten in the layout
    # ``treasure(7), invest(6), loan(6)``.
    t = Evo2AI.DEFAULT_TREASURE
    i = Evo2AI.DEFAULT_INVEST
    l = Evo2AI.DEFAULT_LOAN
    weights = [
        t.bias, t.w_rounds, t.w_my, t.w_avg, t.w_top, t.w_ev, t.w_std,
        i.bias, i.w_rounds, i.w_my, i.w_avg, i.w_top, i.w_amount,
        l.bias, l.w_rounds, l.w_my, l.w_avg, l.w_top, l.w_amount,
    ]
    return weights, Path("<Evo2AI class defaults>")


def _load_evo3_weights(num_players: int) -> tuple[list[float], Path]:
    """Load existing Evo3 weights for ``num_players`` to train against.

    Lookup chain mirrors ``megagem.__main__._evo3_factory`` so "Evo3"
    means whatever ``--ai evo3`` would currently pick — the strongest
    Evo3 snapshot on disk. If none exists, falls back to the class
    defaults baked into ``Evo3AI``; we flatten those into a 25-element
    vector and return a sentinel path so the caller can log "defaults".
    """
    candidates = [
        Path("saved_best_weights") / f"best_weights_evo3_vs_all_{num_players}p.json",
        Path("saved_best_weights") / f"best_weights_evo3_vs_evo2_{num_players}p.json",
        Path("saved_best_weights") / f"best_weights_evo3_self_{num_players}p.json",
        Path("saved_best_weights") / f"best_weights_evo3_{num_players}p.json",
        Path("saved_best_weights") / "best_weights_evo3.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            weights = data["weights"]
            if len(weights) != Evo3AI.NUM_WEIGHTS:
                raise SystemExit(
                    f"{path}: expected {Evo3AI.NUM_WEIGHTS} weights, "
                    f"got {len(weights)}"
                )
            return weights, path

    # Fall back to Evo3 class defaults. Flatten in the layout
    # ``treasure(9), invest(8), loan(8)``.
    t = Evo3AI.DEFAULT_TREASURE
    i = Evo3AI.DEFAULT_INVEST
    l = Evo3AI.DEFAULT_LOAN
    weights = [
        t.bias, t.w_rounds, t.w_my, t.w_avg, t.w_top,
        t.w_ev, t.w_std, t.w_mean_delta, t.w_std_delta,
        i.bias, i.w_rounds, i.w_my, i.w_avg, i.w_top, i.w_amount,
        i.w_mean_delta, i.w_std_delta,
        l.bias, l.w_rounds, l.w_my, l.w_avg, l.w_top, l.w_amount,
        l.w_mean_delta, l.w_std_delta,
    ]
    return weights, Path("<Evo3AI class defaults>")


def evaluate_population(
    population: list[list[float]],
    *,
    opponent_provider: OpponentProvider,
    charts: str,
    games_per_chart: int,
    seed_offset: int,
) -> list[float]:
    """Score every individual in the population against the provided opponents."""
    pop_size = len(population)
    n_games = len(charts) * games_per_chart

    fitness_scores: list[float] = []
    for i in range(pop_size):
        wins = 0
        slot = 0
        for chart_idx, chart in enumerate(charts):
            for game_idx in range(games_per_chart):
                game_seed = seed_offset + chart_idx * 1000 + game_idx
                opponents = opponent_provider(slot, game_seed)
                slot += 1
                if _play_one_game(population[i], opponents, chart, game_seed):
                    wins += 1
        fitness_scores.append(wins / n_games if n_games else 0.0)
    return fitness_scores


def evaluate_population_multi(
    population: list[list[float]],
    *,
    providers: list[tuple[str, OpponentProvider]],
    charts: str,
    games_per_chart: int,
    seed_offset: int,
) -> list[float]:
    """Score every individual against *all* providers and average.

    Every individual plays ``games_per_chart × len(charts)`` games
    against each provider; the returned fitness is the overall win rate
    pooled across all providers. Since every provider contributes the
    same number of games, this equals the mean of the per-provider win
    rates. Each provider uses its own disjoint slice of the seed space
    so a challenger can't get lucky by having the same lucky seed reused
    for every opponent type.
    """
    pop_size = len(population)
    games_per_provider = len(charts) * games_per_chart
    total_games = games_per_provider * len(providers)

    fitness_scores: list[float] = []
    for i in range(pop_size):
        total_wins = 0
        for prov_idx, (_name, provider) in enumerate(providers):
            slot = 0
            prov_offset = seed_offset + prov_idx * 101
            for chart_idx, chart in enumerate(charts):
                for game_idx in range(games_per_chart):
                    game_seed = prov_offset + chart_idx * 1000 + game_idx
                    opponents = provider(slot, game_seed)
                    slot += 1
                    if _play_one_game(population[i], opponents, chart, game_seed):
                        total_wins += 1
        fitness_scores.append(total_wins / total_games if total_games else 0.0)
    return fitness_scores


def evaluate_against_multi(
    challenger: list[float],
    *,
    providers: list[tuple[str, OpponentProvider]],
    charts: str,
    games_per_chart: int,
    seed_offset: int,
) -> tuple[float, dict[str, float]]:
    """Multi-provider held-out eval; returns (pooled rate, per-provider rates)."""
    per_provider: dict[str, float] = {}
    total_wins = 0
    total_games = 0
    for prov_idx, (name, provider) in enumerate(providers):
        wins = 0
        slot = 0
        prov_offset = seed_offset + prov_idx * 101
        for chart_idx, chart in enumerate(charts):
            for game_idx in range(games_per_chart):
                game_seed = prov_offset + chart_idx * 1000 + game_idx
                opponents = provider(slot, game_seed)
                slot += 1
                if _play_one_game(challenger, opponents, chart, game_seed):
                    wins += 1
        n = len(charts) * games_per_chart
        per_provider[name] = wins / n if n else 0.0
        total_wins += wins
        total_games += n
    pooled = total_wins / total_games if total_games else 0.0
    return pooled, per_provider


def evaluate_against_fixed_provider(
    challenger: list[float],
    *,
    opponent_provider: OpponentProvider,
    charts: str,
    games_per_chart: int,
    seed_offset: int,
) -> float:
    """Win rate of one challenger across a held-out seed range."""
    wins = 0
    total = 0
    slot = 0
    for chart_idx, chart in enumerate(charts):
        for game_idx in range(games_per_chart):
            game_seed = seed_offset + chart_idx * 1000 + game_idx
            opponents = opponent_provider(slot, game_seed)
            slot += 1
            if _play_one_game(challenger, opponents, chart, game_seed):
                wins += 1
            total += 1
    return wins / total if total else 0.0


# --- GA primitives ----------------------------------------------------------


GENOME_LEN = Evo4AI.NUM_WEIGHTS  # 26

# Matches scripts.evolve_evo3 so differences in result are attributable to
# the feature set, not to GA hyperparameters.
INIT_LO, INIT_HI = -1.0, 1.0
MUTATION_SIGMA = 0.05
MUTATION_RATE = 0.20
MUTATION_CLIP = 5.0
TOURNAMENT_SIZE = 3
ELITES = 2

# Seed individual #0 with Evo4AI's class defaults — the Evo3 weights
# extended with a zero ``color_bias_influence``. Laid out in flat-vector
# form: treasure(9), invest(8), loan(8), color_bias(1).
DEFAULT_SEED = [
    # treasure: bias, w_rounds, w_my, w_avg, w_top, w_ev, w_std,
    #           w_mean_delta, w_std_delta
    0.9671062444221764,
    -0.0906995616980441,
    0.07804979550128198,
    0.05375147152736104,
    -0.04247465810129918,
    0.32783828473034604,
    -0.011838494331700117,
    0.0,
    0.0,
    # invest: bias, w_rounds, w_my, w_avg, w_top, w_amount,
    #         w_mean_delta, w_std_delta
    1.908464547879478,
    0.4300303741599258,
    -0.1201852409204779,
    -0.28421403664160627,
    0.3149361220138405,
    0.07219353469220569,
    0.0,
    0.0,
    # loan: bias, w_rounds, w_my, w_avg, w_top, w_amount,
    #       w_mean_delta, w_std_delta
    -0.4139242208454687,
    -0.31190499765072527,
    0.13966251262722051,
    0.12135141558388368,
    -0.0669196243751372,
    0.36349000133503273,
    0.0,
    0.0,
    # color_bias_influence — zero so Evo4 starts as Evo3.
    0.0,
]


def random_individual(rng: random.Random) -> list[float]:
    return [rng.uniform(INIT_LO, INIT_HI) for _ in range(GENOME_LEN)]


def tournament_select(
    population: list[list[float]],
    scores: list[float],
    rng: random.Random,
) -> list[float]:
    contenders = rng.sample(range(len(population)), TOURNAMENT_SIZE)
    best = max(contenders, key=lambda i: scores[i])
    return list(population[best])


def crossover(a: list[float], b: list[float], rng: random.Random) -> list[float]:
    return [a[i] if rng.random() < 0.5 else b[i] for i in range(GENOME_LEN)]


def mutate(genome: list[float], rng: random.Random) -> list[float]:
    out = list(genome)
    for i in range(GENOME_LEN):
        if rng.random() < MUTATION_RATE:
            out[i] += rng.gauss(0.0, MUTATION_SIGMA)
            if out[i] > MUTATION_CLIP:
                out[i] = MUTATION_CLIP
            elif out[i] < -MUTATION_CLIP:
                out[i] = -MUTATION_CLIP
    return out


# --- GA loop ----------------------------------------------------------------


@dataclass
class GAResult:
    best_per_gen: list[float] = field(default_factory=list)
    mean_per_gen: list[float] = field(default_factory=list)
    best_weights: list[float] = field(default_factory=list)
    best_fitness: float = 0.0
    best_generation: int = 0


def _render_progress(
    gen: int,
    generations: int,
    best: float,
    mean: float,
    elapsed_total: float,
    bar_width: int = 30,
) -> None:
    done = gen + 1
    frac = done / generations
    filled = int(round(frac * bar_width))
    bar = "#" * filled + "-" * (bar_width - filled)
    avg = elapsed_total / done
    eta = avg * (generations - done)
    line = (
        f"\r[{bar}] gen {done:3d}/{generations} "
        f"best={best:.3f} mean={mean:.3f} "
        f"elapsed={elapsed_total:5.1f}s eta={eta:5.1f}s"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def run_ga(
    *,
    population_size: int,
    generations: int,
    games_per_chart: int,
    seed: int,
    num_players: int = 4,
    opponent_mode: str = "vs_evo3",
) -> GAResult:
    rng = random.Random(seed)

    # --- Initial population
    population: list[list[float]] = [list(DEFAULT_SEED)]
    while len(population) < population_size:
        population.append(random_individual(rng))

    # --- Opponent mode setup
    evo3_weights: list[float] | None = None
    vs_all_providers: list[tuple[str, OpponentProvider]] | None = None
    if opponent_mode == "vs_evo3":
        evo3_weights, evo3_path = _load_evo3_weights(num_players)
        print(f"Evo3 opponents loaded from {evo3_path}")
    elif opponent_mode == "vs_all":
        vs_all_providers = _make_vs_all_providers(num_players)
        print(
            f"vs_all: averaging fitness across {len(vs_all_providers)} "
            f"opponent types ({', '.join(n for n, _ in vs_all_providers)})"
        )

    result = GAResult()
    ga_start = time.perf_counter()
    last_population: list[list[float]] = population
    last_scores: list[float] = []

    n_games_per_eval = len("ABCDE") * games_per_chart

    for gen in range(generations):
        # Fresh, non-overlapping seed range per generation. Same
        # formula as scripts.evolve_evo2 so side-by-side comparisons
        # are on comparable seed offsets.
        seed_offset = (seed + gen + 1) * 9973

        if opponent_mode == "vs_all":
            assert vs_all_providers is not None
            scores = evaluate_population_multi(
                population,
                providers=vs_all_providers,
                charts="ABCDE",
                games_per_chart=games_per_chart,
                seed_offset=seed_offset,
            )
        else:
            if opponent_mode == "self_play":
                provider = _make_self_play_provider(
                    population,
                    num_players=num_players,
                    sample_rng=rng,
                    n_games=n_games_per_eval,
                )
            else:  # vs_evo3
                assert evo3_weights is not None
                provider = _make_evo3_opponent_provider(
                    evo3_weights, num_players=num_players
                )

            scores = evaluate_population(
                population,
                opponent_provider=provider,
                charts="ABCDE",
                games_per_chart=games_per_chart,
                seed_offset=seed_offset,
            )

        best_idx = max(range(len(population)), key=lambda i: scores[i])
        best_score = scores[best_idx]
        mean_score = sum(scores) / len(scores)
        result.best_per_gen.append(best_score)
        result.mean_per_gen.append(mean_score)

        if best_score > result.best_fitness:
            result.best_fitness = best_score
            result.best_weights = list(population[best_idx])
            result.best_generation = gen

        _render_progress(
            gen,
            generations,
            best_score,
            mean_score,
            time.perf_counter() - ga_start,
        )

        last_population = population
        last_scores = scores

        if gen == generations - 1:
            break  # don't bother breeding the last gen

        # --- Breed next generation
        ranked = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
        next_pop: list[list[float]] = [list(population[i]) for i in ranked[:ELITES]]

        while len(next_pop) < population_size:
            parent_a = tournament_select(population, scores, rng)
            parent_b = tournament_select(population, scores, rng)
            child = crossover(parent_a, parent_b, rng)
            child = mutate(child, rng)
            next_pop.append(child)

        population = next_pop

    sys.stdout.write("\n")
    sys.stdout.flush()

    # --- Final held-out re-evaluation of top elites
    if last_scores:
        ranked_final = sorted(
            range(len(last_population)),
            key=lambda i: last_scores[i],
            reverse=True,
        )
        k = min(5, len(ranked_final))
        finalists = [last_population[i] for i in ranked_final[:k]]
        held_out_offset = (seed + generations + 100) * 9973
        held_out_games = max(games_per_chart, 10)
        held_out_n_games = len("ABCDE") * held_out_games

        held_out_scores: list[float] = []
        if opponent_mode == "vs_all":
            assert vs_all_providers is not None
            per_provider_breakdown: list[dict[str, float]] = []
            for cand in finalists:
                pooled, per_provider = evaluate_against_multi(
                    cand,
                    providers=vs_all_providers,
                    charts="ABCDE",
                    games_per_chart=held_out_games,
                    seed_offset=held_out_offset,
                )
                held_out_scores.append(pooled)
                per_provider_breakdown.append(per_provider)
            winner_idx = max(range(k), key=lambda i: held_out_scores[i])
            result.best_weights = list(finalists[winner_idx])
            result.best_fitness = held_out_scores[winner_idx]
            breakdown = per_provider_breakdown[winner_idx]
            print("held-out winner per-opponent win rates:")
            for name, rate in breakdown.items():
                print(f"  vs 3x {name:<12} = {rate * 100:5.1f}%")
        else:
            if opponent_mode == "self_play":
                held_out_provider = _make_self_play_provider(
                    last_population,
                    num_players=num_players,
                    sample_rng=random.Random(seed + 1),
                    n_games=held_out_n_games,
                )
            else:  # vs_evo3
                assert evo3_weights is not None
                held_out_provider = _make_evo3_opponent_provider(
                    evo3_weights, num_players=num_players
                )

            for cand in finalists:
                score = evaluate_against_fixed_provider(
                    cand,
                    opponent_provider=held_out_provider,
                    charts="ABCDE",
                    games_per_chart=held_out_games,
                    seed_offset=held_out_offset,
                )
                held_out_scores.append(score)
            winner_idx = max(range(k), key=lambda i: held_out_scores[i])
            result.best_weights = list(finalists[winner_idx])
            result.best_fitness = held_out_scores[winner_idx]

    return result


# --- Output ----------------------------------------------------------------


def save_history_plot(
    result: GAResult,
    path: Path,
    num_players: int,
    opponent_mode: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    gens = list(range(len(result.best_per_gen)))
    ax.plot(gens, result.best_per_gen, label="best (per-gen seeds)", linewidth=2.0)
    ax.plot(gens, result.mean_per_gen, label="population mean", linewidth=1.5)
    ax.set_xlabel("generation")
    opp_label = {
        "self_play": f"{num_players - 1}× sampled population",
        "vs_evo3":   f"{num_players - 1}× Evo3AI",
        "vs_all":    f"{num_players - 1}× each of 4 previous bots (avg)",
    }[opponent_mode]
    ax.set_ylabel(f"win rate vs {opp_label}")
    mode_pretty = {
        "self_play": "self-play",
        "vs_evo3":   "vs Evo3",
        "vs_all":    "vs all 4 previous bots (averaged)",
    }[opponent_mode]
    ax.set_title(f"Evo4AI GA fitness ({num_players}-player {mode_pretty})")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_best_weights(result: GAResult, path: Path, ga_config: dict) -> None:
    payload = {
        "fitness": result.best_fitness,
        "generation": result.best_generation,
        "weights": result.best_weights,
        "ga_config": ga_config,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def print_paste_ready(weights: list[float]) -> None:
    """Print evolved weights as copy-paste-ready Python."""
    if not weights:
        print("(no weights — GA produced no result)")
        return
    t = weights[0:9]
    i = weights[9:17]
    l = weights[17:25]
    color_bias = weights[25]

    def fmt(block: list[float]) -> str:
        return ", ".join(f"{w:+.4f}" for w in block)

    print()
    print("Evolved weights (paste into Evo4AI):")
    print(f"    DEFAULT_TREASURE = _Evo3TreasureModel({fmt(t)})")
    print(f"    DEFAULT_INVEST   = _Evo3InvestModel({fmt(i)})")
    print(f"    DEFAULT_LOAN     = _Evo3LoanModel({fmt(l)})")
    print(f"    DEFAULT_COLOR_BIAS_INFLUENCE = {color_bias:+.4f}")


# --- CLI -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve Evo4AI weights via GA with rotating seeds."
    )
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--games-per-chart", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-players",
        type=int,
        default=4,
        choices=(3, 4, 5),
        help="Number of seats per fitness game (1 challenger + N-1 opponents).",
    )
    parser.add_argument(
        "--opponent",
        choices=("vs_all", "vs_evo3", "self_play"),
        default="vs_all",
        help=(
            "Opponent source. Default 'vs_all' averages win rate across "
            "all four previous bots (Random, Heuristic, Evo2, Evo3) — "
            "takes 4× longer than single-opponent modes but avoids "
            "overfit. 'vs_evo3' uses fixed Evo3AI loaded from "
            "saved_best_weights/ (lookup chain matches `--ai evo3`); "
            "'self_play' samples from the current population each "
            "generation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for plot + weights output files.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tag = {
        "vs_all":    "vs_all",
        "vs_evo3":   "vs_evo3",
        "self_play": "self",
    }[args.opponent]

    ga_config = {
        "population": args.population,
        "generations": args.generations,
        "games_per_chart": args.games_per_chart,
        "num_players": args.num_players,
        "charts": "ABCDE",
        "seed": args.seed,
        "tournament_size": TOURNAMENT_SIZE,
        "elites": ELITES,
        "mutation_rate": MUTATION_RATE,
        "mutation_sigma": MUTATION_SIGMA,
        "mutation_clip": MUTATION_CLIP,
        "init_range": [INIT_LO, INIT_HI],
        "opponent_mode": args.opponent,
        "fitness_mode": f"{args.opponent}_rotating_seeds",
    }

    print(f"GA config: {json.dumps(ga_config)}")
    t0 = time.perf_counter()
    result = run_ga(
        population_size=args.population,
        generations=args.generations,
        games_per_chart=args.games_per_chart,
        seed=args.seed,
        num_players=args.num_players,
        opponent_mode=args.opponent,
    )
    total_elapsed = time.perf_counter() - t0
    print(f"\nGA finished in {total_elapsed:.1f}s")
    print(
        f"final held-out fitness {result.best_fitness:.3f} "
        f"(per-gen high first hit at generation {result.best_generation})"
    )

    suffix = f"{tag}_{args.num_players}p"
    plot_path = args.output_dir / f"evolve_evo4_history_{suffix}.png"
    weights_path = args.output_dir / f"best_weights_evo4_{suffix}.json"
    save_history_plot(
        result, plot_path,
        num_players=args.num_players,
        opponent_mode=args.opponent,
    )
    save_best_weights(result, weights_path, ga_config)
    print(f"wrote {plot_path}")
    print(f"wrote {weights_path}")

    print_paste_ready(result.best_weights)


if __name__ == "__main__":
    main()
