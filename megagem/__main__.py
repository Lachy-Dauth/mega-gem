"""Entry point: `python -m megagem [...]`."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable

from . import render
from .engine import is_game_over, play_round, score_game, setup_game
from .explain import ExplainingPlayer, render_round_rationales
from .players import (
    AdaptiveHeuristicAI,
    HeuristicAI,
    HumanPlayer,
    HyperAdaptiveAI,
    HyperAdaptiveSplitAI,
    HypergeometricAI,
    Player,
    RandomAI,
)
from .players_evo2 import Evo2AI


# Factory signature: (name, *, seed, num_players) -> Player.
# Wrapped in a dict so the CLI can validate `--ai` and the evolved factory
# can pull a player-count-specific weight set out of artifacts/.
AIFactory = Callable[..., Player]


def _evolved_factory(name: str, *, seed: int, num_players: int) -> Player:
    """Build a HyperAdaptiveSplitAI with GA-evolved weights for this seat count.

    Looks for ``artifacts/best_weights_{N}p.json`` first (per-player-count
    files written by ``scripts/evolve_hyper_adaptive.py``), then falls back
    to the un-suffixed ``best_weights.json``. Raises if neither exists so
    the user gets a clear error instead of a silent default.
    """
    candidates = [
        Path(f"artifacts/best_weights_{num_players}p.json"),
        Path("artifacts/best_weights.json"),
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            return HyperAdaptiveSplitAI.from_weights(name, data["weights"], seed=seed)
    raise SystemExit(
        "No evolved weights found in artifacts/. Run "
        "`python -m scripts.evolve_hyper_adaptive` first."
    )


def _evo2_factory(name: str, *, seed: int, num_players: int) -> Player:
    """Build an Evo2AI with GA-evolved weights for this seat count.

    Lookup order (first match wins):

    1. ``artifacts/best_weights_evo2_vs_old_{N}p.json`` — trained against
       the previous champion (HyperAdaptiveSplitAI). Preferred when present
       since beating the old evo is the bar that matters for the heatmap.
    2. ``artifacts/best_weights_evo2_self_{N}p.json`` — trained via
       self-play within the Evo2 population.
    3. ``artifacts/best_weights_evo2_{N}p.json`` — legacy un-tagged path.
    4. ``artifacts/best_weights_evo2.json`` — global fallback.

    If none exist, falls back to ``Evo2AI``'s class defaults with a
    one-time stderr warning so first-time users can play immediately
    without running the GA.
    """
    candidates = [
        Path(f"artifacts/best_weights_evo2_vs_old_{num_players}p.json"),
        Path(f"artifacts/best_weights_evo2_self_{num_players}p.json"),
        Path(f"artifacts/best_weights_evo2_{num_players}p.json"),
        Path("artifacts/best_weights_evo2.json"),
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            return Evo2AI.from_weights(name, data["weights"], seed=seed)
    print(
        "warning: no evo2 weights in artifacts/ — using class defaults. "
        "Run `python -m scripts.evolve_evo2` for evolved weights.",
        file=sys.stderr,
    )
    return Evo2AI(name, seed=seed)


AI_FACTORIES: dict[str, AIFactory] = {
    "random":     lambda name, *, seed, num_players: RandomAI(name, seed=seed),
    "heuristic":  lambda name, *, seed, num_players: HeuristicAI(name, seed=seed),
    "adaptive":   lambda name, *, seed, num_players: AdaptiveHeuristicAI(name, seed=seed),
    "hyper":      lambda name, *, seed, num_players: HypergeometricAI(name, seed=seed),
    "hyper_adapt": lambda name, *, seed, num_players: HyperAdaptiveAI(name, seed=seed),
    "evolved":    _evolved_factory,
    "evo2":       _evo2_factory,
}


def build_players(
    num_players: int,
    all_ai: bool,
    debug: bool,
    seed: int | None,
    ai_type: str,
) -> list[Player]:
    rng = random.Random(seed)
    ai_names = ["Avery", "Blair", "Casey", "Dylan", "Elliot"]
    factory = AI_FACTORIES[ai_type]
    players: list[Player] = []
    if all_ai:
        for i in range(num_players):
            players.append(factory(ai_names[i], seed=rng.randrange(2**31), num_players=num_players))
    else:
        players.append(HumanPlayer("You", debug=debug))
        for i in range(num_players - 1):
            players.append(factory(ai_names[i], seed=rng.randrange(2**31), num_players=num_players))

    # Debug mode: wrap every non-human seat in an ExplainingPlayer so we
    # can show the AI's reasoning between rounds. The wrapper is purely
    # observational — it forwards every decision unchanged.
    if debug:
        players = [
            p if p.is_human else ExplainingPlayer(p)
            for p in players
        ]
    return players


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="megagem", description="Play MegaGem in the terminal.")
    parser.add_argument("--players", type=int, default=4, choices=[3, 4, 5],
                        help="Total number of players (3-5). One is human unless --all-ai.")
    parser.add_argument("--chart", type=str, default="A", choices=list("ABCDE"),
                        help="Which value chart to use.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed RNG for reproducible games.")
    parser.add_argument("--debug", action="store_true",
                        help=("Debug mode: reveal every player's hand AND print "
                              "AI rationale (features, discounts, reserve, value "
                              "estimate) for each seat after every round."))
    parser.add_argument("--all-ai", action="store_true",
                        help="Replace the human with another AI (smoke testing).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-round output (handy with --all-ai).")
    parser.add_argument("--ai", type=str, default="heuristic",
                        choices=list(AI_FACTORIES),
                        help=("Which AI to use for opponents (and --all-ai players). "
                              "`evolved` loads GA-tuned weights from artifacts/."))
    args = parser.parse_args(argv)

    players = build_players(args.players, args.all_ai, args.debug, args.seed, args.ai)
    state = setup_game(players, chart=args.chart, seed=args.seed)

    rng = random.Random(args.seed)

    if not args.quiet:
        print(f"Starting MegaGem with {args.players} players, chart {args.chart}, AI = {args.ai}.")
        if args.debug:
            print("Debug mode: opponent hands visible, AI rationale printed each round.")
        print()

    while not is_game_over(state):
        result = play_round(state, rng=rng)
        if not args.quiet:
            print()
            print(render.render_round_result(result, state))
            if args.debug:
                rationale_block = render_round_rationales(
                    state, players, result["bids"]
                )
                if rationale_block:
                    print()
                    print(rationale_block)
            print()

    scores = score_game(state)
    print()
    print(render.render_scores(scores))
    return 0


if __name__ == "__main__":
    sys.exit(main())
