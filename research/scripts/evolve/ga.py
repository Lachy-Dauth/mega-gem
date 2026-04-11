"""Genetic-algorithm loop, evaluation, and output for the unified tuner.

The same loop runs every profile (evo1..evo4) and every opponent mode.
Per-profile differences enter via the :class:`scripts.evolve.profiles.AIProfile`
that the caller passes to :func:`run_ga`. Per-mode differences enter
via :func:`scripts.evolve.opponents.build_mode_providers`. There is
exactly one fitness path: rotating per-generation seeds + a final
held-out re-evaluation of the top-5 elites on the same opponent
distribution as training.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

# Headless backend so this runs over SSH / inside CI without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from megagem.engine import is_game_over, play_round, score_game, setup_game

from .opponents import (
    OpponentProvider,
    build_mode_providers,
    load_profile_weights,
)
from .profiles import AIProfile


# --- Shared GA hyperparameters ---------------------------------------------
#
# Initial sampling range for individual #1..N (individual #0 is the
# profile's DEFAULT_SEED). Mutation may push genes a bit further but
# they get clipped at ``profile.mutation_clip``.
INIT_LO, INIT_HI = -1.0, 1.0
MUTATION_RATE = 0.20
TOURNAMENT_SIZE = 3
ELITES = 2
CHARTS = "ABCDE"


# --- GA primitives ----------------------------------------------------------


def random_individual(profile: AIProfile, rng: random.Random) -> list[float]:
    return [rng.uniform(INIT_LO, INIT_HI) for _ in range(profile.num_weights)]


def tournament_select(
    population: list[list[float]],
    scores: list[float],
    rng: random.Random,
) -> list[float]:
    contenders = rng.sample(range(len(population)), TOURNAMENT_SIZE)
    best = max(contenders, key=lambda i: scores[i])
    return list(population[best])


def crossover(
    a: list[float],
    b: list[float],
    rng: random.Random,
) -> list[float]:
    return [a[i] if rng.random() < 0.5 else b[i] for i in range(len(a))]


def mutate(
    profile: AIProfile,
    genome: list[float],
    rng: random.Random,
) -> list[float]:
    sigma = profile.mutation_sigma
    clip = profile.mutation_clip
    out = list(genome)
    for i in range(len(out)):
        if rng.random() < MUTATION_RATE:
            out[i] += rng.gauss(0.0, sigma)
            if out[i] > clip:
                out[i] = clip
            elif out[i] < -clip:
                out[i] = -clip
    return out


# --- GAResult dataclass -----------------------------------------------------


@dataclass
class GAResult:
    best_per_gen: list[float] = field(default_factory=list)
    mean_per_gen: list[float] = field(default_factory=list)
    best_weights: list[float] = field(default_factory=list)
    best_fitness: float = 0.0
    best_generation: int = 0


# --- Progress bar -----------------------------------------------------------


def _render_progress(
    gen: int,
    generations: int,
    best: float,
    mean: float,
    elapsed_total: float,
    bar_width: int = 30,
) -> None:
    """In-place progress bar. Uses ``\\r`` so successive calls overwrite."""
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


# --- Fitness evaluation -----------------------------------------------------


def _play_one_game(
    profile: AIProfile,
    challenger_weights: list[float],
    opponents: list,
    chart: str,
    seed: int,
) -> bool:
    """Run one full game and return True iff the challenger won outright."""
    challenger = profile.ai_class.from_weights(
        profile.label,
        challenger_weights,
        seed=seed * 7,
    )
    players = [challenger] + opponents
    state = setup_game(players, chart=chart, seed=seed)
    game_rng = random.Random(seed)
    while not is_game_over(state):
        play_round(state, rng=game_rng)
    scores = score_game(state)
    return scores[0]["total"] > max(s["total"] for s in scores[1:])


def evaluate_population_multi(
    profile: AIProfile,
    population: list[list[float]],
    *,
    providers: list[tuple[str, OpponentProvider]],
    games_per_chart: int,
    seed_offset: int,
) -> list[float]:
    """Score every individual against *all* providers and pool win rates.

    Every individual plays ``games_per_chart × len(CHARTS)`` games
    against each provider; the returned fitness is the overall pooled
    win rate. With ``len(providers) == 1`` this collapses to a single-
    provider win rate, so the GA loop never has to special-case mode
    cardinality.

    Each provider uses its own disjoint slice of the seed space (offset
    by ``prov_idx * 101``) so a challenger can't get lucky by hitting
    the same favourable seed for every opponent type.
    """
    pop_size = len(population)
    games_per_provider = len(CHARTS) * games_per_chart
    total_games = games_per_provider * len(providers)

    fitness_scores: list[float] = []
    for i in range(pop_size):
        total_wins = 0
        for prov_idx, (_name, provider) in enumerate(providers):
            slot = 0
            prov_offset = seed_offset + prov_idx * 101
            for chart_idx, chart in enumerate(CHARTS):
                for game_idx in range(games_per_chart):
                    game_seed = prov_offset + chart_idx * 1000 + game_idx
                    opponents = provider(slot, game_seed)
                    slot += 1
                    if _play_one_game(profile, population[i], opponents, chart, game_seed):
                        total_wins += 1
        fitness_scores.append(total_wins / total_games if total_games else 0.0)
    return fitness_scores


def evaluate_against_multi(
    profile: AIProfile,
    challenger: list[float],
    *,
    providers: list[tuple[str, OpponentProvider]],
    games_per_chart: int,
    seed_offset: int,
) -> tuple[float, dict[str, float]]:
    """Held-out eval of one challenger across all providers.

    Returns the pooled win rate plus a per-provider breakdown so the
    held-out re-eval can print "vs 3x Random = 88.0%" lines for the
    chosen winner.
    """
    per_provider: dict[str, float] = {}
    total_wins = 0
    total_games = 0
    for prov_idx, (name, provider) in enumerate(providers):
        wins = 0
        slot = 0
        prov_offset = seed_offset + prov_idx * 101
        for chart_idx, chart in enumerate(CHARTS):
            for game_idx in range(games_per_chart):
                game_seed = prov_offset + chart_idx * 1000 + game_idx
                opponents = provider(slot, game_seed)
                slot += 1
                if _play_one_game(profile, challenger, opponents, chart, game_seed):
                    wins += 1
        n = len(CHARTS) * games_per_chart
        per_provider[name] = wins / n if n else 0.0
        total_wins += wins
        total_games += n
    pooled = total_wins / total_games if total_games else 0.0
    return pooled, per_provider


# --- GA loop ----------------------------------------------------------------


def run_ga(
    profile: AIProfile,
    *,
    mode_key: str,
    population_size: int,
    generations: int,
    games_per_chart: int,
    seed: int,
    num_players: int,
) -> GAResult:
    """Run one full GA tuning of ``profile`` under ``mode_key``.

    Same loop for every profile/mode combination. The two pieces of
    per-mode variation are:

    1. **Self-play rebuilds providers each generation** because the
       population evolves. The fixed-opponent and ``vs_all`` modes
       could keep one provider list, but we rebuild every gen for
       symmetry — building a provider is cheap.
    2. **The held-out re-evaluation** runs against the same mode the
       GA trained against, with a held-out seed range and per-provider
       breakdown reporting.
    """
    rng = random.Random(seed)

    # --- Initial population: the current champion + random fillers.
    # Individual #0 is loaded from saved_best_weights/ via the profile's
    # own lookup chain (the same chain used to load opponent weights for
    # vs_all / vs_evoK modes). If no weights file exists yet, we fall
    # back to the AI class's hardcoded defaults. This guarantees every
    # GA run *starts from the best known weights* instead of a stale
    # constant — you can just re-run the script to iterate.
    seed_loaded = load_profile_weights(profile, num_players)
    print(f"seed individual #0: {profile.label} from {seed_loaded.source}")
    population: list[list[float]] = [list(seed_loaded.weights)]
    while len(population) < population_size:
        population.append(random_individual(profile, rng))

    # --- Print which providers training will use (helpful for vs_all)
    n_games_per_eval = len(CHARTS) * games_per_chart
    initial_providers = build_mode_providers(
        mode_key,
        profile,
        num_players=num_players,
        population=population,
        sample_rng=rng,
        n_games=n_games_per_eval,
    )
    if mode_key == "vs_all":
        names = ", ".join(name for name, _ in initial_providers)
        print(
            f"vs_all: averaging fitness across {len(initial_providers)} "
            f"opponent types ({names})"
        )

    result = GAResult()
    ga_start = time.perf_counter()
    last_population: list[list[float]] = population
    last_scores: list[float] = []

    for gen in range(generations):
        # Each generation gets a fresh, non-overlapping seed range. The
        # offset is multiplied by a prime to spread bits and indexed by
        # generation to guarantee non-overlap with prior gens. Same
        # formula as the legacy evolve_evo2 / evo3 / evo4 scripts so
        # side-by-side comparisons are on comparable seed offsets.
        seed_offset = (seed + gen + 1) * 9973

        # Rebuild providers every generation. Required for self_play
        # (the population changes); cheap and symmetric for the rest.
        providers = build_mode_providers(
            mode_key,
            profile,
            num_players=num_players,
            population=population,
            sample_rng=rng,
            n_games=n_games_per_eval,
            quiet=True,  # only the first build prints opponent sources
        )

        scores = evaluate_population_multi(
            profile,
            population,
            providers=providers,
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
            child = mutate(profile, child, rng)
            next_pop.append(child)

        population = next_pop

    sys.stdout.write("\n")
    sys.stdout.flush()

    # --- Held-out re-evaluation of top elites
    # The per-generation scores are noisy because seeds rotate. Take the
    # top-K survivors of the last generation, rescore each against a
    # held-out seed range, and pick the winner.
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
        held_out_n_games = len(CHARTS) * held_out_games

        held_out_providers = build_mode_providers(
            mode_key,
            profile,
            num_players=num_players,
            population=last_population,
            # Independent RNG so the held-out self-play sample is
            # decorrelated from any per-gen sample.
            sample_rng=random.Random(seed + 1),
            n_games=held_out_n_games,
            quiet=True,
        )

        held_out_scores: list[float] = []
        per_provider_breakdown: list[dict[str, float]] = []
        for cand in finalists:
            pooled, per_provider = evaluate_against_multi(
                profile,
                cand,
                providers=held_out_providers,
                games_per_chart=held_out_games,
                seed_offset=held_out_offset,
            )
            held_out_scores.append(pooled)
            per_provider_breakdown.append(per_provider)

        winner_idx = max(range(k), key=lambda i: held_out_scores[i])
        result.best_weights = list(finalists[winner_idx])
        result.best_fitness = held_out_scores[winner_idx]

        # Print the per-provider breakdown for any multi-provider mode.
        # Single-provider modes have a one-line breakdown that doesn't
        # add anything beyond the pooled rate, so skip the noise.
        winner_breakdown = per_provider_breakdown[winner_idx]
        if len(winner_breakdown) > 1:
            print("held-out winner per-opponent win rates:")
            for name, rate in winner_breakdown.items():
                print(f"  vs 3x {name:<22} = {rate * 100:5.1f}%")
        # best_generation stays as the gen that first hit the per-gen high.

    return result


# --- Output -----------------------------------------------------------------


def save_history_plot(
    result: GAResult,
    path: Path,
    *,
    profile: AIProfile,
    num_players: int,
    mode_key: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    gens = list(range(len(result.best_per_gen)))
    ax.plot(gens, result.best_per_gen, label="best (per-gen seeds)", linewidth=2.0)
    ax.plot(gens, result.mean_per_gen, label="population mean", linewidth=1.5)
    ax.set_xlabel("generation")
    ax.set_ylabel(f"win rate vs {num_players - 1}× opponents ({mode_key})")
    ax.set_title(
        f"{profile.label} GA fitness "
        f"({num_players}-player, opponent={mode_key})"
    )
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
