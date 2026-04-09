"""Genetic-algorithm tuner for HyperAdaptiveSplitAI's 18 weights.

Run from the project root::

    python -m scripts.evolve_hyper_adaptive

Outputs land in ``artifacts/`` (gitignored):

* ``evolve_history.png`` — best/mean fitness curve.
* ``best_weights.json``  — winning genome + GA config.

Fitness = win rate of one ``HyperAdaptiveSplitAI`` versus three
``HeuristicAI`` opponents, averaged across all five value charts on a
fixed seed range. The seed range is reused on every fitness call so the
function is deterministic — vital for fair tournament selection.
"""

from __future__ import annotations

import argparse
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
from megagem.players import HeuristicAI, HyperAdaptiveSplitAI


# --- Fitness ----------------------------------------------------------------


def fitness(
    weights: list[float],
    *,
    charts: str = "ABCDE",
    games_per_chart: int = 10,
    num_players: int = 4,
) -> float:
    """Win rate vs (num_players - 1) HeuristicAI opponents.

    Deterministic in `weights` for a given (charts, games_per_chart,
    num_players) tuple: seeds are derived from a fixed range so two
    evaluations of the same genome give the same score. This is
    non-negotiable for tournament selection — without it the GA chases
    noise.
    """
    if num_players not in (3, 4, 5):
        raise ValueError(f"num_players must be 3, 4, or 5; got {num_players}")
    wins = 0
    total = 0
    for chart in charts:
        for seed in range(games_per_chart):
            ai = HyperAdaptiveSplitAI.from_weights("Evo", weights, seed=seed * 7)
            players = [ai]
            for k in range(num_players - 1):
                players.append(HeuristicAI(f"H{k + 1}", seed=seed * 7 + k + 1))
            state = setup_game(players, chart=chart, seed=seed)
            rng = random.Random(seed)
            while not is_game_over(state):
                play_round(state, rng=rng)
            scores = score_game(state)
            if scores[0]["total"] > max(s["total"] for s in scores[1:]):
                wins += 1
            total += 1
    return wins / total if total else 0.0


# --- GA primitives ----------------------------------------------------------


GENOME_LEN = HyperAdaptiveSplitAI.NUM_WEIGHTS  # 18

# Initial sampling range; mutation may push genes a bit further but they
# get clipped at MUTATION_CLIP.
INIT_LO, INIT_HI = -1.0, 1.0
MUTATION_SIGMA = 0.15
MUTATION_RATE = 0.20
MUTATION_CLIP = 2.0
TOURNAMENT_SIZE = 3
ELITES = 2

# Defaults to seed individual #0 with — these are the class defaults the
# GA already starts above the floor with.
DEFAULT_SEED = [
    # treasure
    0.70, 0.25, 0.35, -0.10, -0.15, -0.05,
    # invest
    0.80, 0.10, 0.20, -0.05, -0.05, 0.00,
    # loan
    0.10, 0.05, -0.40, 0.10, 0.10, -0.05,
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


def _cache_key(weights: list[float]) -> tuple[float, ...]:
    # Round so equivalent floats hit the same cache slot.
    return tuple(round(w, 4) for w in weights)


def _render_progress(
    gen: int,
    generations: int,
    best: float,
    mean: float,
    elapsed_total: float,
    bar_width: int = 30,
) -> None:
    """In-place progress bar. Uses \\r so successive calls overwrite."""
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
) -> GAResult:
    rng = random.Random(seed)
    cache: dict[tuple[float, ...], float] = {}

    def evaluate(genome: list[float]) -> float:
        key = _cache_key(genome)
        if key not in cache:
            cache[key] = fitness(
                genome,
                games_per_chart=games_per_chart,
                num_players=num_players,
            )
        return cache[key]

    # --- Initial population
    population: list[list[float]] = [list(DEFAULT_SEED)]
    while len(population) < population_size:
        population.append(random_individual(rng))

    result = GAResult()
    ga_start = time.perf_counter()

    for gen in range(generations):
        scores = [evaluate(ind) for ind in population]

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
    return result


# --- Output ----------------------------------------------------------------


def save_history_plot(result: GAResult, path: Path, num_players: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    gens = list(range(len(result.best_per_gen)))
    ax.plot(gens, result.best_per_gen, label="best", linewidth=2.0)
    ax.plot(gens, result.mean_per_gen, label="population mean", linewidth=1.5)
    ax.set_xlabel("generation")
    ax.set_ylabel(f"win rate vs {num_players - 1}× HeuristicAI")
    ax.set_title(
        f"HyperAdaptiveSplitAI GA fitness ({num_players}-player games)"
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


def print_paste_ready(weights: list[float]) -> None:
    """Print evolved weights as copy-paste-ready Python."""
    if not weights:
        print("(no weights — GA produced no result)")
        return
    t = weights[0:6]
    i = weights[6:12]
    l = weights[12:18]

    def fmt(block: list[float]) -> str:
        return ", ".join(f"{w:+.4f}" for w in block)

    print()
    print("Evolved weights (paste into HyperAdaptiveSplitAI):")
    print(f"    DEFAULT_TREASURE = _BidModel({fmt(t)})")
    print(f"    DEFAULT_INVEST   = _BidModel({fmt(i)})")
    print(f"    DEFAULT_LOAN     = _BidModel({fmt(l)})")


# --- CLI -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve HyperAdaptiveSplitAI weights via GA."
    )
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--games-per-chart", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-players",
        type=int,
        default=4,
        choices=(3, 4, 5),
        help="Number of seats per fitness game (1 challenger + N-1 HeuristicAI).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for evolve_history_{N}p.png and best_weights_{N}p.json.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    }

    print(f"GA config: {json.dumps(ga_config)}")
    t0 = time.perf_counter()
    result = run_ga(
        population_size=args.population,
        generations=args.generations,
        games_per_chart=args.games_per_chart,
        seed=args.seed,
        num_players=args.num_players,
    )
    total_elapsed = time.perf_counter() - t0
    print(f"\nGA finished in {total_elapsed:.1f}s")
    print(
        f"best fitness {result.best_fitness:.3f} "
        f"(generation {result.best_generation})"
    )

    suffix = f"{args.num_players}p"
    plot_path = args.output_dir / f"evolve_history_{suffix}.png"
    weights_path = args.output_dir / f"best_weights_{suffix}.json"
    save_history_plot(result, plot_path, num_players=args.num_players)
    save_best_weights(result, weights_path, ga_config)
    print(f"wrote {plot_path}")
    print(f"wrote {weights_path}")

    print_paste_ready(result.best_weights)


if __name__ == "__main__":
    main()
