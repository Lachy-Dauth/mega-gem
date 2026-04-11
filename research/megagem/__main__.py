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
    Evo2AI,
    Evo3AI,
    Evo4AI,
    HeuristicAI,
    HumanPlayer,
    HyperAdaptiveSplitAI,
    Player,
    RandomAI,
)


# Factory signature: (name, *, seed, num_players) -> Player.
# Wrapped in a dict so the CLI can validate `--ai` and the evolved factory
# can pull a player-count-specific weight set out of artifacts/.
AIFactory = Callable[..., Player]


# Weights are loaded ONLY from `saved_best_weights/` (checked-in
# snapshots). `artifacts/` is transient GA output — promote a fresh
# run by copying its weights file into `saved_best_weights/` yourself.
_WEIGHTS_DIR = Path("saved_best_weights")


def _candidate_weight_paths(profile_key: str, num_players: int) -> list[Path]:
    """Uniform lookup chain for every evo profile.

    Mirrors :func:`scripts.evolve.opponents.candidate_filenames` so the
    CLI and the GA tuner see the same files in the same order. First
    existing match wins.
    """
    tags = (
        "vs_all", "vs_random", "vs_heuristic",
        "vs_evo1", "vs_evo2", "vs_evo3", "vs_evo4",
        "self",
    )
    return [
        _WEIGHTS_DIR / f"best_weights_{profile_key}_{tag}_{num_players}p.json"
        for tag in tags
    ] + [_WEIGHTS_DIR / f"best_weights_{profile_key}_{num_players}p.json"]


def _load_evo_weights(profile_key: str, num_players: int) -> list[float] | None:
    """Return the first matching weights file's vector, or None if nothing exists."""
    for path in _candidate_weight_paths(profile_key, num_players):
        if path.exists():
            data = json.loads(path.read_text())
            return data["weights"]
    return None


def _evolved_factory(name: str, *, seed: int, num_players: int) -> Player:
    """Build a HyperAdaptiveSplitAI with GA-evolved weights for this seat count.

    Walks the uniform lookup chain (``best_weights_evo1_{tag}_{N}p.json``
    for every tag). Raises if nothing exists so the user gets a clear
    error instead of a silent default.
    """
    weights = _load_evo_weights("evo1", num_players)
    if weights is None:
        raise SystemExit(
            "No evo1 weights found in saved_best_weights/. Run "
            "`python -m scripts.evolve --ai evo1` and copy "
            "artifacts/best_weights_evo1_*.json into saved_best_weights/."
        )
    return HyperAdaptiveSplitAI.from_weights(name, weights, seed=seed)


def _evo2_factory(name: str, *, seed: int, num_players: int) -> Player:
    """Build an Evo2AI from the uniform evo2 lookup chain; class defaults if missing."""
    weights = _load_evo_weights("evo2", num_players)
    if weights is None:
        print(
            "warning: no evo2 weights in saved_best_weights/ — using class "
            "defaults. Run `python -m scripts.evolve --ai evo2` and copy "
            "the result into saved_best_weights/ for evolved weights.",
            file=sys.stderr,
        )
        return Evo2AI(name, seed=seed)
    return Evo2AI.from_weights(name, weights, seed=seed)


def _evo3_factory(name: str, *, seed: int, num_players: int) -> Player:
    """Build an Evo3AI from the uniform evo3 lookup chain; class defaults if missing."""
    weights = _load_evo_weights("evo3", num_players)
    if weights is None:
        print(
            "warning: no evo3 weights in saved_best_weights/ — using class "
            "defaults. Run `python -m scripts.evolve --ai evo3` and copy "
            "the result into saved_best_weights/ for evolved weights.",
            file=sys.stderr,
        )
        return Evo3AI(name, seed=seed)
    return Evo3AI.from_weights(name, weights, seed=seed)


def _evo4_factory(name: str, *, seed: int, num_players: int) -> Player:
    """Build an Evo4AI from the uniform evo4 lookup chain; class defaults if missing.

    Class defaults reduce to Evo3 behaviour (zero color-bias influence),
    so the fallback is the same strength as ``--ai evo3`` with defaults
    — no regression vs the previous AI.
    """
    weights = _load_evo_weights("evo4", num_players)
    if weights is None:
        print(
            "warning: no evo4 weights in saved_best_weights/ — using class "
            "defaults (equivalent to Evo3). Run `python -m scripts.evolve --ai evo4` "
            "and copy the result into saved_best_weights/ for evolved weights.",
            file=sys.stderr,
        )
        return Evo4AI(name, seed=seed)
    return Evo4AI.from_weights(name, weights, seed=seed)


AI_FACTORIES: dict[str, AIFactory] = {
    "random":     lambda name, *, seed, num_players: RandomAI(name, seed=seed),
    "heuristic":  lambda name, *, seed, num_players: HeuristicAI(name, seed=seed),
    "evolved":    _evolved_factory,
    "evo2":       _evo2_factory,
    "evo3":       _evo3_factory,
    "evo4":       _evo4_factory,
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
                              "`evolved`, `evo2`, `evo3`, and `evo4` load GA-tuned "
                              "weights from saved_best_weights/."))
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
