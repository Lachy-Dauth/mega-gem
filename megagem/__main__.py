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
    Evo2AI,
    Evo3AI,
    HeuristicAI,
    HumanPlayer,
    HyperAdaptiveAI,
    HyperAdaptiveSplitAI,
    HypergeometricAI,
    Player,
    RandomAI,
)


# Factory signature: (name, *, seed, num_players) -> Player.
# Wrapped in a dict so the CLI can validate `--ai` and the evolved factory
# can pull a player-count-specific weight set out of artifacts/.
AIFactory = Callable[..., Player]


# Weights are looked up in `artifacts/` first (where fresh GA runs land — the
# directory is gitignored), then `saved_best_weights/` (checked-in snapshots
# so fresh clones can play against trained AIs without running the GA).
_WEIGHTS_DIRS: tuple[str, ...] = ("artifacts", "saved_best_weights")


def _expand_weight_candidates(filenames: list[str]) -> list[Path]:
    """Return every filename joined with every weights dir, in priority order."""
    return [Path(d) / f for f in filenames for d in _WEIGHTS_DIRS]


def _evolved_factory(name: str, *, seed: int, num_players: int) -> Player:
    """Build a HyperAdaptiveSplitAI with GA-evolved weights for this seat count.

    Looks for ``best_weights_{N}p.json`` first (per-player-count files
    written by ``scripts/evolve_hyper_adaptive.py``), then falls back to
    the un-suffixed ``best_weights.json``. Each filename is tried in
    ``artifacts/`` then ``saved_best_weights/``. Raises if nothing
    exists so the user gets a clear error instead of a silent default.
    """
    candidates = _expand_weight_candidates([
        f"best_weights_{num_players}p.json",
        "best_weights.json",
    ])
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            return HyperAdaptiveSplitAI.from_weights(name, data["weights"], seed=seed)
    raise SystemExit(
        "No evolved weights found in artifacts/ or saved_best_weights/. Run "
        "`python -m scripts.evolve_hyper_adaptive` first."
    )


def _evo3_factory(name: str, *, seed: int, num_players: int) -> Player:
    """Build an Evo3AI with GA-evolved weights for this seat count.

    Lookup order (first match wins), each filename checked in
    ``artifacts/`` then ``saved_best_weights/``:

    1. ``best_weights_evo3_vs_all_{N}p.json`` — trained with the
       fitness averaged across all six previous bots. Top priority:
       that is the training regime ``scripts/evolve_evo3.py`` uses by
       default.
    2. ``best_weights_evo3_vs_evo2_{N}p.json`` — trained against a
       frozen Evo2 snapshot.
    3. ``best_weights_evo3_self_{N}p.json`` — self-play.
    4. ``best_weights_evo3_{N}p.json`` — legacy un-tagged.
    5. ``best_weights_evo3.json`` — global fallback.

    If none exist, falls back to ``Evo3AI``'s class defaults with a
    one-time stderr warning so ``--ai evo3`` still works without
    running the GA.
    """
    candidates = _expand_weight_candidates([
        f"best_weights_evo3_vs_all_{num_players}p.json",
        f"best_weights_evo3_vs_evo2_{num_players}p.json",
        f"best_weights_evo3_self_{num_players}p.json",
        f"best_weights_evo3_{num_players}p.json",
        "best_weights_evo3.json",
    ])
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            return Evo3AI.from_weights(name, data["weights"], seed=seed)
    print(
        "warning: no evo3 weights in artifacts/ or saved_best_weights/ — "
        "using class defaults. Run `python -m scripts.evolve_evo3` for "
        "evolved weights.",
        file=sys.stderr,
    )
    return Evo3AI(name, seed=seed)


def _evo2_factory(name: str, *, seed: int, num_players: int) -> Player:
    """Build an Evo2AI with GA-evolved weights for this seat count.

    Lookup order (first match wins), each filename checked in
    ``artifacts/`` then ``saved_best_weights/``:

    1. ``best_weights_evo2_vs_old_evo2_{N}p.json`` — trained against
       an earlier Evo2AI snapshot. Top priority because this is a
       strict refinement of "best Evo2 we've ever produced".
    2. ``best_weights_evo2_vs_old_{N}p.json`` — trained against the
       pre-Evo2 champion (HyperAdaptiveSplitAI).
    3. ``best_weights_evo2_self_{N}p.json`` — trained via self-play
       within the Evo2 population.
    4. ``best_weights_evo2_{N}p.json`` — legacy un-tagged path.
    5. ``best_weights_evo2.json`` — global fallback.

    If none exist, falls back to ``Evo2AI``'s class defaults with a
    one-time stderr warning so first-time users can play immediately
    without running the GA.
    """
    candidates = _expand_weight_candidates([
        f"best_weights_evo2_vs_old_evo2_{num_players}p.json",
        f"best_weights_evo2_vs_old_{num_players}p.json",
        f"best_weights_evo2_self_{num_players}p.json",
        f"best_weights_evo2_{num_players}p.json",
        "best_weights_evo2.json",
    ])
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            return Evo2AI.from_weights(name, data["weights"], seed=seed)
    print(
        "warning: no evo2 weights in artifacts/ or saved_best_weights/ — "
        "using class defaults. Run `python -m scripts.evolve_evo2` for "
        "evolved weights.",
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
    "evo3":       _evo3_factory,
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
