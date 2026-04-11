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
import multiprocessing
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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
# Initial sampling range for the purely-random escape-hatch individuals
# (see ``init_from_seed`` below). The bulk of the initial population is
# seeded from the current champion instead of this uniform range.
INIT_LO, INIT_HI = -1.0, 1.0
INIT_SIGMA_SCALE = 0.5
MUTATION_RATE = 0.20
TOURNAMENT_SIZE = 3
ELITES = 2
CHARTS = "ABCDE"


# --- GA primitives ----------------------------------------------------------


def random_individual(profile: AIProfile, rng: random.Random) -> list[float]:
    return [rng.uniform(INIT_LO, INIT_HI) for _ in range(profile.num_weights)]


def init_from_seed(
    profile: AIProfile,
    seed_weights: list[float],
    rng: random.Random,
) -> list[float]:
    """Return a perturbed copy of ``seed_weights`` for the initial pop.

    Every gene gets an independent gaussian kick at
    ``profile.mutation_sigma * INIT_SIGMA_SCALE`` and is clipped to
    ``profile.mutation_clip``. Unlike :func:`mutate`, this touches
    *all* genes — the point of the initial population is diversity
    around the champion, not incremental refinement.
    """
    sigma = profile.mutation_sigma * INIT_SIGMA_SCALE
    clip = profile.mutation_clip
    out: list[float] = []
    for w in seed_weights:
        v = w + rng.gauss(0.0, sigma)
        if v > clip:
            v = clip
        elif v < -clip:
            v = -clip
        out.append(v)
    return out


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


@dataclass(frozen=True)
class _GameTask:
    """One ``(individual, provider, chart, game)`` unit of fitness work.

    Built once up-front by the evaluation functions and then either
    iterated sequentially or fanned out across a thread pool. The
    ``individual_idx`` and ``provider_idx`` fields are the merge keys
    used to accumulate wins back into per-individual / per-provider
    buckets after the pool drains. ``slot`` matches the sequential
    counter the previous nested-loop code passed to the provider, so
    self-play opponent slates line up exactly with what they would
    have been in the un-parallelised path.
    """

    individual_idx: int
    provider_idx: int
    chart: str
    game_seed: int
    slot: int


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


# --- Process-pool worker state ---------------------------------------------
#
# Opponent providers in ``scripts/evolve/opponents.py`` are inner-function
# closures and therefore not picklable, so we can't pass them as task
# arguments across the IPC boundary. Instead we stash them (plus the
# profile and population) in this module-level dict *before* building the
# :class:`ProcessPoolExecutor`. With the ``fork`` mp context, every
# worker process inherits the parent's memory via copy-on-write, so
# ``_WORKER_STATE`` appears pre-populated in each worker at zero IPC
# cost. Only a tiny ``_GameTask`` dataclass travels per call.
#
# Invariants:
# * Populated by the parent just before opening the pool.
# * Cleared in a ``finally`` block so the parent doesn't retain stale
#   references to populations / providers after the pool exits.
# * Read-only inside workers — nothing in ``_worker_run_task`` mutates
#   the captured values, and the providers in ``opponents.py`` are
#   stateless closures over frozen captured data. Do not introduce
#   stateful providers without revisiting this invariant.
_WORKER_STATE: dict = {}


def _worker_run_task(task: _GameTask) -> tuple[int, int, bool]:
    """Process-pool worker: play one fitness game.

    Reads the profile / population / providers from the module-level
    ``_WORKER_STATE`` dict that was populated in the parent before the
    pool was spawned. Returns ``(individual_idx, provider_idx, won)``
    so the parent can accumulate results into the correct bucket —
    either ``wins_by_ind`` (population eval) or ``wins_by_prov``
    (held-out eval).
    """
    profile = _WORKER_STATE["profile"]
    population = _WORKER_STATE["population"]
    providers = _WORKER_STATE["providers"]
    provider = providers[task.provider_idx][1]
    opponents = provider(task.slot, task.game_seed)
    won = _play_one_game(
        profile,
        population[task.individual_idx],
        opponents,
        task.chart,
        task.game_seed,
    )
    return (task.individual_idx, task.provider_idx, won)


def _pool_chunksize(total_tasks: int, workers: int) -> int:
    """Pick a ``chunksize`` for ``ProcessPoolExecutor.map``.

    Aim for ~4 chunks per worker: small enough that a slow game
    can't leave one core idle at the tail, large enough that IPC
    and pickling overhead stay negligible next to per-game compute.
    """
    if total_tasks <= 0 or workers <= 0:
        return 1
    return max(1, total_tasks // (workers * 4))


def evaluate_population_multi(
    profile: AIProfile,
    population: list[list[float]],
    *,
    providers: list[tuple[str, OpponentProvider]],
    games_per_chart: int,
    seed_offset: int,
    workers: int,
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

    **Parallelism.** When ``workers > 1`` the per-game work is fanned
    out across a :class:`concurrent.futures.ProcessPoolExecutor` using
    the ``fork`` multiprocessing context. Workers inherit the parent's
    profile / population / providers via copy-on-write at fork time
    (see ``_WORKER_STATE``), so the non-picklable opponent-provider
    closures in ``opponents.py`` cross the process boundary for free
    — only a tiny :class:`_GameTask` dataclass travels per call.

    Results are **bit-for-bit deterministic at any worker count**
    given the same ``--seed``: per-game seeds derive only from the
    loop indices (``seed_offset + prov_idx * 101 + chart_idx * 1000 +
    game_idx``), and merging is integer counting which is commutative,
    so completion order does not affect the final score.

    The ``workers <= 1`` branch keeps the original fully-sequential
    path available for debugging and reproducibility checks (no pool
    construction overhead, clean ``pdb`` sessions).
    """
    pop_size = len(population)
    games_per_provider = len(CHARTS) * games_per_chart
    total_games = games_per_provider * len(providers)

    tasks: list[_GameTask] = []
    for i in range(pop_size):
        for prov_idx in range(len(providers)):
            prov_offset = seed_offset + prov_idx * 101
            for chart_idx, chart in enumerate(CHARTS):
                for game_idx in range(games_per_chart):
                    game_seed = prov_offset + chart_idx * 1000 + game_idx
                    slot = chart_idx * games_per_chart + game_idx
                    tasks.append(
                        _GameTask(
                            individual_idx=i,
                            provider_idx=prov_idx,
                            chart=chart,
                            game_seed=game_seed,
                            slot=slot,
                        )
                    )

    wins_by_ind = [0] * pop_size

    if workers <= 1:
        # Sequential fallback: identical to the pre-parallel code
        # path. Kept first-class so reproducibility checks and pdb
        # sessions skip all pool overhead.
        for task in tasks:
            provider = providers[task.provider_idx][1]
            opponents = provider(task.slot, task.game_seed)
            if _play_one_game(
                profile,
                population[task.individual_idx],
                opponents,
                task.chart,
                task.game_seed,
            ):
                wins_by_ind[task.individual_idx] += 1
    else:
        _WORKER_STATE["profile"] = profile
        _WORKER_STATE["population"] = population
        _WORKER_STATE["providers"] = providers
        try:
            ctx = multiprocessing.get_context("fork")
            cs = _pool_chunksize(len(tasks), workers)
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
            ) as executor:
                for ind_idx, _prov_idx, won in executor.map(
                    _worker_run_task, tasks, chunksize=cs
                ):
                    if won:
                        wins_by_ind[ind_idx] += 1
        finally:
            _WORKER_STATE.clear()

    return [w / total_games if total_games else 0.0 for w in wins_by_ind]


def evaluate_against_multi(
    profile: AIProfile,
    challenger: list[float],
    *,
    providers: list[tuple[str, OpponentProvider]],
    games_per_chart: int,
    seed_offset: int,
    workers: int,
) -> tuple[float, dict[str, float]]:
    """Held-out eval of one challenger across all providers.

    Returns the pooled win rate plus a per-provider breakdown so the
    held-out re-eval can print "vs 3x Random = 88.0%" lines for the
    chosen winner.

    Same parallelism model as :func:`evaluate_population_multi`:
    sequential fallback when ``workers <= 1``, otherwise fan-out via a
    fork-backed :class:`concurrent.futures.ProcessPoolExecutor`. The
    single challenger is wrapped as a one-element "population" before
    being placed into ``_WORKER_STATE`` so ``_worker_run_task`` can
    use exactly the same ``population[task.individual_idx]`` lookup
    (here ``individual_idx`` is always ``0``) and we don't need a
    second worker entry point. Determinism is preserved for the same
    reasons (loop-derived seeds, commutative integer merging).
    """
    n_per_provider = len(CHARTS) * games_per_chart

    tasks: list[_GameTask] = []
    for prov_idx in range(len(providers)):
        prov_offset = seed_offset + prov_idx * 101
        for chart_idx, chart in enumerate(CHARTS):
            for game_idx in range(games_per_chart):
                game_seed = prov_offset + chart_idx * 1000 + game_idx
                slot = chart_idx * games_per_chart + game_idx
                tasks.append(
                    _GameTask(
                        individual_idx=0,
                        provider_idx=prov_idx,
                        chart=chart,
                        game_seed=game_seed,
                        slot=slot,
                    )
                )

    wins_by_prov = [0] * len(providers)

    if workers <= 1:
        for task in tasks:
            provider = providers[task.provider_idx][1]
            opponents = provider(task.slot, task.game_seed)
            if _play_one_game(
                profile, challenger, opponents, task.chart, task.game_seed
            ):
                wins_by_prov[task.provider_idx] += 1
    else:
        _WORKER_STATE["profile"] = profile
        _WORKER_STATE["population"] = [challenger]
        _WORKER_STATE["providers"] = providers
        try:
            ctx = multiprocessing.get_context("fork")
            cs = _pool_chunksize(len(tasks), workers)
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
            ) as executor:
                for _ind_idx, prov_idx, won in executor.map(
                    _worker_run_task, tasks, chunksize=cs
                ):
                    if won:
                        wins_by_prov[prov_idx] += 1
        finally:
            _WORKER_STATE.clear()

    per_provider: dict[str, float] = {}
    total_wins = 0
    total_games = 0
    for prov_idx, (name, _provider) in enumerate(providers):
        wins = wins_by_prov[prov_idx]
        per_provider[name] = wins / n_per_provider if n_per_provider else 0.0
        total_wins += wins
        total_games += n_per_provider
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
    workers: int,
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

    # --- Initial population: the current champion + perturbed copies.
    # Individual #0 is loaded from saved_best_weights/ via the profile's
    # own lookup chain (the same chain used to load opponent weights for
    # vs_all / vs_evoK modes). If no weights file exists yet, we fall
    # back to the AI class's hardcoded defaults. This guarantees every
    # GA run *starts from the best known weights* instead of a stale
    # constant — you can just re-run the script to iterate.
    #
    # The rest of the population is seeded as *perturbed copies* of the
    # champion rather than fully-random vectors. The previous "one good
    # + N-1 uniform[-1,1] noise" init meant generation 0 was effectively
    # "champion vs garbage": the champion trivially won the tournament,
    # selection couldn't distinguish the rest, and the GA burned several
    # generations just climbing back to parity with the seed. Filling
    # with gaussian perturbations around the champion puts every
    # individual in a meaningful neighbourhood on day one.
    seed_loaded = load_profile_weights(profile, num_players)
    print(f"seed individual #0: {profile.label} from {seed_loaded.source}")
    seed_weights = list(seed_loaded.weights)
    population: list[list[float]] = [list(seed_weights)]
    while len(population) < population_size:
        population.append(init_from_seed(profile, seed_weights, rng))

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
            workers=workers,
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
                workers=workers,
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
