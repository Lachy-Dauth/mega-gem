"""CLI entry point for the RL (Evolution Strategies) trainer.

Invocation::

    python -m scripts.rl --ai evo4 --opponent vs_all

Defaults target Evo4 because that's the current champion and the
project's stated RL target. Other profiles (``evo1``..``evo3``) are
accepted for symmetry with :mod:`scripts.evolve`; the 35-vector path
generalizes to 18/19/25-vector profiles without further changes.

Output layout — all under ``artifacts/`` by default, with an ``rl_``
infix so files never collide with GA outputs or accidentally feed
into :func:`scripts.evolve.opponents.candidate_filenames`::

    artifacts/best_weights_evo4_rl_{tag}_{N}p.json
    artifacts/rl_evo4_history_{tag}_{N}p.png
    artifacts/rl_evo4_state_{tag}_{N}p.json   # for --resume

where ``{tag}`` is :data:`scripts.evolve.opponents.MODE_FILENAME_TAGS`
for the chosen ``--opponent`` mode, and ``{N}`` is the player count.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from scripts.evolve.opponents import MODE_FILENAME_TAGS, MODE_KEYS
from scripts.evolve.profiles import AI_PROFILES, PROFILE_KEYS

from .trainer import (
    load_resume_state,
    run_es,
    save_best_weights,
    save_history_plot,
    save_resume_state,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts.rl",
        description=(
            "Reinforcement-learning tuner via Evolution Strategies "
            "(Salimans et al. 2017). Parameter-space policy gradient "
            "on the 35-weight Evo4 policy (or any other AIProfile). "
            "Algorithmically distinct from scripts.evolve (GA): a "
            "single point θ moves via mirrored-sampling gradient "
            "estimates and an Adam update, no population/tournament/"
            "crossover. Outputs match the GA's file format so trained "
            "weights can be promoted into saved_best_weights/ and "
            "loaded by the existing CLI / heatmap unchanged."
        ),
    )
    parser.add_argument(
        "--ai",
        choices=PROFILE_KEYS,
        default="evo4",
        help=(
            "Which AI profile to tune. Default: evo4 (35 weights). "
            "evo1 = HyperAdaptiveSplitAI (18), evo2 = Evo2AI (19), "
            "evo3 = Evo3AI (25)."
        ),
    )
    parser.add_argument(
        "--opponent",
        choices=MODE_KEYS,
        default="vs_all",
        help=(
            "Opponent mode. 'vs_all' (default) pools reward across "
            "Heuristic + every other evo profile. 'vs_random' / "
            "'vs_heuristic' / 'vs_evo1..4' fix the opponents to one "
            "class. 'self_play' samples opponents from small gaussian "
            "perturbations of the current θ."
        ),
    )
    parser.add_argument(
        "--population",
        type=int,
        default=48,
        help=(
            "Perturbation batch size. MUST be even (mirrored pairs). "
            "Default 48."
        ),
    )
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--games-per-chart", type=int, default=10)
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.10,
        help=(
            "ES noise std. NOT the GA's mutation sigma — ES sigma is "
            "the search radius of the gaussian policy. Default 0.10."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.03,
        help="Adam learning rate for the outer parameter update.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-players",
        type=int,
        default=4,
        choices=(3, 4, 5),
        help="Seats per game (1 challenger + N-1 opponents).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help=(
            "Number of worker processes for perturbation evaluation. "
            "Pass --workers 1 for the fully-sequential path "
            "(debugging, reproducibility tests). Results are "
            "deterministic for any worker count given the same --seed."
        ),
    )
    parser.add_argument(
        "--theta-clip",
        type=float,
        default=5.0,
        help="Component-wise θ clip applied after every Adam step.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for the plot + weights + resume files.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Optional resume file produced by a previous run. "
            "θ, Adam moments, and RNG state are restored; the loop "
            "runs --generations additional generations on top."
        ),
    )
    args = parser.parse_args(argv)

    if args.population % 2 != 0 or args.population < 2:
        parser.error(
            f"--population must be a positive even integer, "
            f"got {args.population}"
        )

    profile = AI_PROFILES[args.ai]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rl_config = {
        "ai": args.ai,
        "label": profile.label,
        "num_weights": profile.num_weights,
        "population": args.population,
        "generations": args.generations,
        "games_per_chart": args.games_per_chart,
        "num_players": args.num_players,
        "charts": "ABCDE",
        "seed": args.seed,
        "sigma": args.sigma,
        "lr": args.lr,
        "theta_clip": args.theta_clip,
        "opponent_mode": args.opponent,
        "fitness_mode": f"{args.opponent}_rank_shaped_margin",
        "workers": args.workers,
        "optimizer": "adam",
        "algorithm": "openai_es_mirrored",
    }

    print(f"RL config: {json.dumps(rl_config)}")

    resume_state = None
    if args.resume is not None:
        resume_state = load_resume_state(args.resume)
        if resume_state.get("profile_key", profile.key) != profile.key:
            parser.error(
                f"--resume file is for profile "
                f"{resume_state.get('profile_key')!r}, not {profile.key!r}"
            )

    t0 = time.perf_counter()
    result = run_es(
        profile,
        mode_key=args.opponent,
        population_size=args.population,
        generations=args.generations,
        games_per_chart=args.games_per_chart,
        sigma=args.sigma,
        lr=args.lr,
        seed=args.seed,
        num_players=args.num_players,
        workers=args.workers,
        theta_clip=args.theta_clip,
        resume=resume_state,
    )
    total_elapsed = time.perf_counter() - t0
    print(f"\nES finished in {total_elapsed:.1f}s")
    print(
        f"θ₀ reward={result.initial_mean_reward:+.3f} "
        f"win_rate={result.initial_win_rate:.2f}"
    )
    print(
        f"best held-out reward={result.best_mean_reward:+.3f} "
        f"win_rate={result.best_win_rate:.2f} "
        f"at generation {result.best_generation}"
    )

    tag = MODE_FILENAME_TAGS[args.opponent]
    suffix = f"rl_{tag}_{args.num_players}p"
    plot_path = args.output_dir / f"rl_{profile.key}_history_{tag}_{args.num_players}p.png"
    weights_path = args.output_dir / f"best_weights_{profile.key}_{suffix}.json"
    state_path = args.output_dir / f"rl_{profile.key}_state_{tag}_{args.num_players}p.json"

    save_history_plot(
        result,
        plot_path,
        profile=profile,
        num_players=args.num_players,
        mode_key=args.opponent,
    )
    save_best_weights(result, weights_path, rl_config)
    save_resume_state(
        result,
        state_path,
        profile=profile,
        mode_key=args.opponent,
        num_players=args.num_players,
        rl_config=rl_config,
    )
    print(f"wrote {plot_path}")
    print(f"wrote {weights_path}")
    print(f"wrote {state_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
