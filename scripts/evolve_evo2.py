"""Genetic-algorithm tuner for ``Evo2AI``'s 19 weights.

Run from the project root::

    python -m scripts.evolve_evo2                         # default: self-play
    python -m scripts.evolve_evo2 --opponent old_evo      # vs HyperAdaptiveSplitAI
    python -m scripts.evolve_evo2 --opponent old_evo2     # vs the existing best Evo2AI

Outputs land in ``artifacts/`` (gitignored):

* ``evolve_evo2_history_{tag}_{N}p.png`` — best/mean fitness curve.
* ``best_weights_evo2_{tag}_{N}p.json`` — winning genome + GA config.

where ``{tag}`` is ``self`` for self-play, ``vs_old`` for old-evo
mode, and ``vs_old_evo2`` for old-evo2 mode. The CLI's ``--ai evo2``
factory checks all three.

Three opponent modes:

1. **``--opponent self_play`` (default).** Each individual is
   evaluated against ``num_players − 1`` opponents *sampled randomly
   from the current population*. Within a generation, all individuals
   see the same pre-sampled opponent slate for each (chart, game) slot,
   so tournament selection stays fair. Across generations, opponents
   change as the population evolves.

2. **``--opponent old_evo``.** Opponents are fixed: ``num_players − 1``
   instances of ``HyperAdaptiveSplitAI`` loaded from
   ``artifacts/best_weights_{N}p.json``. Use this to evolve a
   challenger that *specifically* beats the pre-Evo2 champion. Errors
   if the old-evo weights file for the chosen seat count doesn't exist.

3. **``--opponent old_evo2``.** Opponents are fixed: ``num_players − 1``
   instances of ``Evo2AI`` loaded from the existing best Evo2 weights
   for this seat count (same lookup chain as ``--ai evo2``, minus the
   file this run will write). Use this to evolve a strict refinement
   of the current best Evo2. Errors if no prior Evo2 weights exist.

In all three modes the GA also uses **rotating seeds**: each generation
gets a fresh, non-overlapping range of game seeds, preventing
overfitting to a specific 50-seed slice. Side effect: best-fitness
per generation is **not** monotone non-decreasing — the elite from
gen N is re-evaluated in gen N+1 against new seeds and may drop. The
plot reflects raw per-generation scores; the saved "best weights"
come from a held-out re-evaluation across many seeds at the end of
the run, applied to the top elites.
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
from megagem.players import Evo2AI, HyperAdaptiveSplitAI, Player


# An OpponentProvider builds the opponent slate for one game. It takes the
# zero-based game-slot index (within the current generation) and the game
# seed, and returns a list of ``num_players − 1`` ready-to-play Players.
# This indirection lets the GA loop be agnostic to whether opponents come
# from the current population (self-play) or a fixed external AI.
OpponentProvider = Callable[[int, int], list[Player]]


# --- Fitness ----------------------------------------------------------------


def _play_one_game(
    challenger_weights: list[float],
    opponents: list[Player],
    chart: str,
    seed: int,
) -> bool:
    """Run one full game and return True iff the challenger won outright."""
    challenger = Evo2AI.from_weights("Evo", challenger_weights, seed=seed * 7)
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
    generation and reused for every individual. Across individuals the
    seeds and opponent slate are identical; only the challenger differs.
    This is what lets tournament selection compare scores honestly.

    Self-as-opponent is allowed (probability 1/pop_size) — filtering it
    would bias sampling per individual or require resampling loops.
    """
    pop_size = len(population)
    opp_indices_per_slot: list[list[int]] = [
        [sample_rng.randrange(pop_size) for _ in range(num_players - 1)]
        for _ in range(n_games)
    ]

    def provider(slot: int, game_seed: int) -> list[Player]:
        return [
            Evo2AI.from_weights(
                f"O{k + 1}",
                population[opp_indices_per_slot[slot][k]],
                seed=game_seed + k + 1,
            )
            for k in range(num_players - 1)
        ]

    return provider


def _make_old_evo_provider(
    old_evo_weights: list[float],
    *,
    num_players: int,
) -> OpponentProvider:
    """Build a provider that returns ``num_players − 1`` HyperAdaptiveSplitAIs.

    Opponents are deterministic across generations (the same fixed
    weights every game), so the only thing that varies between games
    is the seed. This is the right setup when you want to evolve a
    challenger that *specifically* beats the previous champion.
    """
    def provider(slot: int, game_seed: int) -> list[Player]:
        return [
            HyperAdaptiveSplitAI.from_weights(
                f"OE{k + 1}",
                old_evo_weights,
                seed=game_seed + k + 1,
            )
            for k in range(num_players - 1)
        ]

    return provider


def _make_old_evo2_provider(
    old_evo2_weights: list[float],
    *,
    num_players: int,
) -> OpponentProvider:
    """Build a provider that returns ``num_players − 1`` Evo2AIs from fixed weights.

    Same shape as :func:`_make_old_evo_provider` but for Evo2 — used by
    the ``old_evo2`` training mode where the opponents are a frozen
    snapshot of the strongest existing Evo2AI.
    """
    def provider(slot: int, game_seed: int) -> list[Player]:
        return [
            Evo2AI.from_weights(
                f"OE2{k + 1}",
                old_evo2_weights,
                seed=game_seed + k + 1,
            )
            for k in range(num_players - 1)
        ]

    return provider


def _load_old_evo_weights(num_players: int) -> list[float]:
    """Load the previous champion's weights for ``num_players``-seat games.

    Prefers ``best_weights_{N}p.json``, falls back to the unsuffixed
    ``best_weights.json`` so the script still works on older artifact
    layouts. Errors loudly if neither exists.
    """
    filenames = [
        f"best_weights_{num_players}p.json",
        "best_weights.json",
    ]
    candidates = [Path(d) / f for f in filenames for d in ("artifacts", "saved_best_weights")]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            weights = data["weights"]
            if len(weights) != HyperAdaptiveSplitAI.NUM_WEIGHTS:
                raise SystemExit(
                    f"{path}: expected {HyperAdaptiveSplitAI.NUM_WEIGHTS} "
                    f"weights, got {len(weights)}"
                )
            return weights
    raise SystemExit(
        f"Old-evo opponent requested but no weights file found. Run "
        f"`python -m scripts.evolve_hyper_adaptive` first to produce "
        f"artifacts/best_weights_{num_players}p.json."
    )


def _load_old_evo2_weights(num_players: int) -> tuple[list[float], Path]:
    """Load existing Evo2 weights for ``num_players`` to train against.

    The lookup chain mirrors :func:`megagem.__main__._evo2_factory` so
    "old evo2" means "whatever ``--ai evo2`` would currently pick" —
    i.e. the strongest Evo2 you've already produced. Returns the
    weights *and* the source path so the caller can log which file
    was loaded (handy for telling iterative runs apart).

    Note: this deliberately does **not** consider the file the new
    run will write (``best_weights_evo2_vs_old_evo2_{N}p.json``), so
    re-running ``--opponent old_evo2`` always trains against the same
    pre-existing baseline rather than chasing its own tail. To do
    iterative refinement, copy the new file over a higher-priority
    name yourself.
    """
    filenames = [
        f"best_weights_evo2_vs_old_{num_players}p.json",
        f"best_weights_evo2_self_{num_players}p.json",
        f"best_weights_evo2_{num_players}p.json",
        "best_weights_evo2.json",
    ]
    candidates = [Path(d) / f for f in filenames for d in ("artifacts", "saved_best_weights")]
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
    raise SystemExit(
        "Old-evo2 opponent requested but no Evo2 weights file found. "
        "Run `python -m scripts.evolve_evo2` (default self-play mode) "
        f"first to produce artifacts/best_weights_evo2_self_{num_players}p.json."
    )


def evaluate_population(
    population: list[list[float]],
    *,
    opponent_provider: OpponentProvider,
    charts: str,
    games_per_chart: int,
    seed_offset: int,
) -> list[float]:
    """Score every individual in the population against the provided opponents.

    The ``opponent_provider`` decides where opponents come from — see
    :func:`_make_self_play_provider` and :func:`_make_old_evo_provider`.
    From the GA's perspective, every individual plays the same set of
    games (same seeds, same provider call sequence), so tournament
    selection compares like with like.
    """
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


def evaluate_against_fixed_provider(
    challenger: list[float],
    *,
    opponent_provider: OpponentProvider,
    charts: str,
    games_per_chart: int,
    seed_offset: int,
) -> float:
    """Win rate of one challenger across a held-out seed range.

    Used at the end of the GA to re-evaluate the top elites on seeds
    they haven't trained against. The provider is whatever the run
    used during training (so for old-evo mode the held-out games are
    also against old-evo, and for self-play mode they're against a
    snapshot of the final population sampled the same way).
    """
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


GENOME_LEN = Evo2AI.NUM_WEIGHTS  # 19

# Initial sampling range; mutation may push genes a bit further but they
# get clipped at MUTATION_CLIP. Match the existing GA so any difference
# in result is attributable to the new feature set, not the GA hyperparams.
INIT_LO, INIT_HI = -1.0, 1.0
MUTATION_SIGMA = 0.05
MUTATION_RATE = 0.20
MUTATION_CLIP = 5.0
TOURNAMENT_SIZE = 3
ELITES = 2

# Defaults to seed individual #0 with — Evo2AI's class defaults laid out
# in flat-vector form. Order: treasure(7), invest(6), loan(6).
# Heads now output the *raw bid in coins* directly, so weights are sized
# accordingly (see Evo2AI.DEFAULT_TREASURE etc. for the rationale).
DEFAULT_SEED = [
    # treasure: bias, w_rounds, w_my, w_avg, w_top, w_ev, w_std
    0.9671062444221764,
    -0.0906995616980441,
    0.07804979550128198,
    0.05375147152736104,
    -0.04247465810129918,
    0.32783828473034604,
    -0.011838494331700117,
    # invest: bias, w_rounds, w_my, w_avg, w_top, w_amount
    1.908464547879478,
    0.4300303741599258,
    -0.1201852409204779,
    -0.28421403664160627,
    0.3149361220138405,
    0.07219353469220569,
    # loan: bias, w_rounds, w_my, w_avg, w_top, w_amount
    -0.4139242208454687,
    -0.31190499765072527,
    0.13966251262722051,
    0.12135141558388368,
    -0.0669196243751372,
    0.36349000133503273,
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


def run_ga(
    *,
    population_size: int,
    generations: int,
    games_per_chart: int,
    seed: int,
    num_players: int = 4,
    opponent_mode: str = "self_play",
) -> GAResult:
    rng = random.Random(seed)

    # --- Initial population
    population: list[list[float]] = [list(DEFAULT_SEED)]
    while len(population) < population_size:
        population.append(random_individual(rng))

    # --- Opponent mode setup
    old_evo_weights: list[float] | None = None
    old_evo2_weights: list[float] | None = None
    if opponent_mode == "old_evo":
        old_evo_weights = _load_old_evo_weights(num_players)
    elif opponent_mode == "old_evo2":
        old_evo2_weights, old_evo2_path = _load_old_evo2_weights(num_players)
        print(f"old_evo2 opponents loaded from {old_evo2_path}")

    result = GAResult()
    ga_start = time.perf_counter()
    last_population: list[list[float]] = population
    last_scores: list[float] = []

    n_games_per_eval = len("ABCDE") * games_per_chart

    for gen in range(generations):
        # Each generation gets a fresh seed range. The offset jumps in
        # blocks of (charts × games_per_chart) so consecutive generations
        # never overlap. Multiplied by a prime to spread the bits.
        seed_offset = (seed + gen + 1) * 9973

        # Build the per-generation opponent provider. Self-play needs a
        # fresh provider per generation because the population evolves;
        # the fixed-opponent modes (old_evo, old_evo2) have stable
        # providers but we still rebuild for symmetry.
        if opponent_mode == "self_play":
            provider = _make_self_play_provider(
                population,
                num_players=num_players,
                sample_rng=rng,
                n_games=n_games_per_eval,
            )
        elif opponent_mode == "old_evo":
            assert old_evo_weights is not None
            provider = _make_old_evo_provider(
                old_evo_weights, num_players=num_players
            )
        else:  # old_evo2
            assert old_evo2_weights is not None
            provider = _make_old_evo2_provider(
                old_evo2_weights, num_players=num_players
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
    # The per-generation scores are noisy because seeds rotate. Take the
    # top-K survivors of the last generation and rescore each against a
    # held-out seed range. Pick the winner.
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

        # Held-out provider mirrors the training mode so the final pick
        # is judged the same way the GA was selecting all along.
        if opponent_mode == "self_play":
            held_out_provider = _make_self_play_provider(
                last_population,
                num_players=num_players,
                sample_rng=random.Random(seed + 1),
                n_games=held_out_n_games,
            )
        elif opponent_mode == "old_evo":
            assert old_evo_weights is not None
            held_out_provider = _make_old_evo_provider(
                old_evo_weights, num_players=num_players
            )
        else:  # old_evo2
            assert old_evo2_weights is not None
            held_out_provider = _make_old_evo2_provider(
                old_evo2_weights, num_players=num_players
            )

        held_out_scores: list[float] = []
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
        # best_generation stays as the gen that first hit the per-gen high.

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
        "old_evo": f"{num_players - 1}× HyperAdaptiveSplitAI (old evo)",
        "old_evo2": f"{num_players - 1}× Evo2AI (old evo2)",
    }[opponent_mode]
    ax.set_ylabel(f"win rate vs {opp_label}")
    mode_pretty = {
        "self_play": "self-play",
        "old_evo": "vs old evo",
        "old_evo2": "vs old evo2",
    }[opponent_mode]
    ax.set_title(f"Evo2AI GA fitness ({num_players}-player {mode_pretty})")
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
    t = weights[0:7]
    i = weights[7:13]
    l = weights[13:19]

    def fmt(block: list[float]) -> str:
        return ", ".join(f"{w:+.4f}" for w in block)

    print()
    print("Evolved weights (paste into Evo2AI):")
    print(f"    DEFAULT_TREASURE = _TreasureModel({fmt(t)})")
    print(f"    DEFAULT_INVEST   = _InvestModel({fmt(i)})")
    print(f"    DEFAULT_LOAN     = _LoanModel({fmt(l)})")


# --- CLI -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve Evo2AI weights via GA with rotating seeds."
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
        choices=("self_play", "old_evo", "old_evo2"),
        default="self_play",
        help=(
            "Opponent source. 'self_play' samples from the current "
            "population each generation; 'old_evo' uses fixed "
            "HyperAdaptiveSplitAI loaded from artifacts/best_weights*; "
            "'old_evo2' uses fixed Evo2AI loaded from "
            "artifacts/best_weights_evo2_* (lookup chain matches "
            "`--ai evo2`)."
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
        "self_play": "self",
        "old_evo": "vs_old",
        "old_evo2": "vs_old_evo2",
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
    plot_path = args.output_dir / f"evolve_evo2_history_{suffix}.png"
    weights_path = args.output_dir / f"best_weights_evo2_{suffix}.json"
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
