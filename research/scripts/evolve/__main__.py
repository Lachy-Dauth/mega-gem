"""Unified CLI entry point: ``python -m scripts.evolve``.

Tunes one of the four evolvable AIs (``evo1``..``evo4``) with one of
the eight uniform opponent modes (``vs_all``, ``vs_random``,
``vs_heuristic``, ``vs_evo1``..``vs_evo4``, ``self_play``). All four
AIs share the same opponent registry, the same GA loop, and the same
output filename convention::

    artifacts/best_weights_evo{K}_{tag}_{N}p.json
    artifacts/evolve_evo{K}_history_{tag}_{N}p.png

where ``{K}`` is 1..4, ``{tag}`` is the opponent mode tag (``vs_all``,
``vs_random``, …, ``self``), and ``{N}`` is the player count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .ga import (
    ELITES,
    INIT_HI,
    INIT_LO,
    MUTATION_RATE,
    TOURNAMENT_SIZE,
    run_ga,
    save_best_weights,
    save_history_plot,
)
from .opponents import MODE_FILENAME_TAGS, MODE_KEYS
from .profiles import AI_PROFILES, PROFILE_KEYS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts.evolve",
        description=(
            "Unified GA tuner for the four evolvable AIs. "
            "Pick a profile via --ai and an opponent mode via --opponent."
        ),
    )
    parser.add_argument(
        "--ai",
        required=True,
        choices=PROFILE_KEYS,
        help=(
            "Which AI profile to tune. "
            "evo1 = HyperAdaptiveSplitAI (18 weights), "
            "evo2 = Evo2AI (19 weights), "
            "evo3 = Evo3AI (25 weights), "
            "evo4 = Evo4AI (35 weights)."
        ),
    )
    parser.add_argument(
        "--opponent",
        choices=MODE_KEYS,
        default="vs_all",
        help=(
            "Opponent mode. 'vs_all' (default) pools fitness across "
            "Heuristic + every other evo profile. "
            "'vs_random'/'vs_heuristic'/'vs_evo1'..'vs_evo4' fix the "
            "opponents to one class (loaded from saved_best_weights/ "
            "for the evo* targets). 'self_play' samples opponents from "
            "the current population each generation."
        ),
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
        help="Seats per fitness game (1 challenger + N-1 opponents).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help=(
            "Number of threads used for fitness evaluation. Defaults "
            "to all CPU cores. Pass --workers 1 for the fully-"
            "sequential path (debugging, reproducibility checks). "
            "Results are deterministic for any worker count given the "
            "same --seed. Note: stock CPython's GIL limits CPU-bound "
            "speedup; free-threaded builds (python3.13t) get true "
            "parallelism with no code changes."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for the plot + weights output files.",
    )
    args = parser.parse_args(argv)

    profile = AI_PROFILES[args.ai]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ga_config = {
        "ai": args.ai,
        "label": profile.label,
        "num_weights": profile.num_weights,
        "population": args.population,
        "generations": args.generations,
        "games_per_chart": args.games_per_chart,
        "num_players": args.num_players,
        "charts": "ABCDE",
        "seed": args.seed,
        "tournament_size": TOURNAMENT_SIZE,
        "elites": ELITES,
        "mutation_rate": MUTATION_RATE,
        "mutation_sigma": profile.mutation_sigma,
        "mutation_clip": profile.mutation_clip,
        "init_range": [INIT_LO, INIT_HI],
        "opponent_mode": args.opponent,
        "fitness_mode": f"{args.opponent}_rotating_seeds",
        "workers": args.workers,
    }

    print(f"GA config: {json.dumps(ga_config)}")

    t0 = time.perf_counter()
    result = run_ga(
        profile,
        mode_key=args.opponent,
        population_size=args.population,
        generations=args.generations,
        games_per_chart=args.games_per_chart,
        seed=args.seed,
        num_players=args.num_players,
        workers=args.workers,
    )
    total_elapsed = time.perf_counter() - t0
    print(f"\nGA finished in {total_elapsed:.1f}s")
    print(
        f"final held-out fitness {result.best_fitness:.3f} "
        f"(per-gen high first hit at generation {result.best_generation})"
    )

    tag = MODE_FILENAME_TAGS[args.opponent]
    suffix = f"{tag}_{args.num_players}p"
    plot_path = args.output_dir / f"evolve_{profile.key}_history_{suffix}.png"
    weights_path = args.output_dir / f"best_weights_{profile.key}_{suffix}.json"
    save_history_plot(
        result,
        plot_path,
        profile=profile,
        num_players=args.num_players,
        mode_key=args.opponent,
    )
    save_best_weights(result, weights_path, ga_config)
    print(f"wrote {plot_path}")
    print(f"wrote {weights_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
