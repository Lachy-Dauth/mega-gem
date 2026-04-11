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


def _expand_weight_candidates(filenames: list[str]) -> list[Path]:
    """Return each filename joined with the canonical weights dir."""
    return [_WEIGHTS_DIR / f for f in filenames]


def _evolved_factory(name: str, *, seed: int, num_players: int) -> Player:
    """Build a HyperAdaptiveSplitAI with GA-evolved weights for this seat count.

    Lookup order (first match wins), all paths rooted in
    ``saved_best_weights/``:

    1. ``best_weights_evo1_vs_all_{N}p.json`` — written by the unified
       ``python -m scripts.evolve --ai evo1`` tuner (default mode).
    2. ``best_weights_evo1_vs_heuristic_{N}p.json`` — single-opponent
       fallback.
    3. ``best_weights_evo1_self_{N}p.json`` — self-play.
    4. ``best_weights_evo1_{N}p.json`` — un-tagged.
    5. ``best_weights_{N}p.json`` — legacy file written by the
       pre-unified ``evolve_hyper_adaptive`` script (still in the repo).
    6. ``best_weights.json`` — global legacy fallback.

    Raises if nothing exists so the user gets a clear error instead of
    a silent default.
    """
    candidates = _expand_weight_candidates([
        f"best_weights_evo1_vs_all_{num_players}p.json",
        f"best_weights_evo1_vs_heuristic_{num_players}p.json",
        f"best_weights_evo1_self_{num_players}p.json",
        f"best_weights_evo1_{num_players}p.json",
        f"best_weights_{num_players}p.json",
        "best_weights.json",
    ])
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            return HyperAdaptiveSplitAI.from_weights(name, data["weights"], seed=seed)
    raise SystemExit(
        "No evolved weights found in saved_best_weights/. Run "
        "`python -m scripts.evolve --ai evo1` and copy "
        "artifacts/best_weights_evo1_*.json into saved_best_weights/."
    )


def _evo3_factory(name: str, *, seed: int, num_players: int) -> Player:
    """Build an Evo3AI with GA-evolved weights for this seat count.

    Lookup order (first match wins), all paths rooted in
    ``saved_best_weights/``:

    1. ``best_weights_evo3_vs_all_{N}p.json`` — trained with the
       fitness averaged across every other bot (the
       ``python -m scripts.evolve --ai evo3`` default).
    2. ``best_weights_evo3_vs_evo1_{N}p.json`` — vs frozen evo1.
    3. ``best_weights_evo3_vs_evo2_{N}p.json`` — vs frozen evo2.
    4. ``best_weights_evo3_vs_evo4_{N}p.json`` — vs frozen evo4.
    5. ``best_weights_evo3_self_{N}p.json`` — self-play.
    6. ``best_weights_evo3_{N}p.json`` — legacy un-tagged.
    7. ``best_weights_evo3.json`` — global fallback.

    If none exist, falls back to ``Evo3AI``'s class defaults with a
    one-time stderr warning so ``--ai evo3`` still works without
    running the GA.
    """
    candidates = _expand_weight_candidates([
        f"best_weights_evo3_vs_all_{num_players}p.json",
        f"best_weights_evo3_vs_evo1_{num_players}p.json",
        f"best_weights_evo3_vs_evo2_{num_players}p.json",
        f"best_weights_evo3_vs_evo4_{num_players}p.json",
        f"best_weights_evo3_self_{num_players}p.json",
        f"best_weights_evo3_{num_players}p.json",
        "best_weights_evo3.json",
    ])
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            return Evo3AI.from_weights(name, data["weights"], seed=seed)
    print(
        "warning: no evo3 weights in saved_best_weights/ — using class "
        "defaults. Run `python -m scripts.evolve --ai evo3` and copy "
        "the result into saved_best_weights/ for evolved weights.",
        file=sys.stderr,
    )
    return Evo3AI(name, seed=seed)


def _evo2_factory(name: str, *, seed: int, num_players: int) -> Player:
    """Build an Evo2AI with GA-evolved weights for this seat count.

    Lookup order (first match wins), all paths rooted in
    ``saved_best_weights/``:

    1. ``best_weights_evo2_vs_all_{N}p.json`` — trained with the
       fitness averaged across every other bot (the
       ``python -m scripts.evolve --ai evo2`` default).
    2. ``best_weights_evo2_vs_evo1_{N}p.json`` — vs frozen evo1.
    3. ``best_weights_evo2_vs_evo3_{N}p.json`` — vs frozen evo3.
    4. ``best_weights_evo2_vs_evo4_{N}p.json`` — vs frozen evo4.
    5. ``best_weights_evo2_self_{N}p.json`` — self-play.
    6. ``best_weights_evo2_vs_old_evo2_{N}p.json`` — legacy
       (pre-unified ``--opponent old_evo2`` mode).
    7. ``best_weights_evo2_vs_old_{N}p.json`` — legacy
       (pre-unified ``--opponent old_evo`` mode; the existing
       checked-in 4-player file lives here).
    8. ``best_weights_evo2_{N}p.json`` — legacy un-tagged path.
    9. ``best_weights_evo2.json`` — global fallback.

    If none exist, falls back to ``Evo2AI``'s class defaults with a
    one-time stderr warning so first-time users can play immediately
    without running the GA.
    """
    candidates = _expand_weight_candidates([
        f"best_weights_evo2_vs_all_{num_players}p.json",
        f"best_weights_evo2_vs_evo1_{num_players}p.json",
        f"best_weights_evo2_vs_evo3_{num_players}p.json",
        f"best_weights_evo2_vs_evo4_{num_players}p.json",
        f"best_weights_evo2_self_{num_players}p.json",
        f"best_weights_evo2_vs_old_evo2_{num_players}p.json",
        f"best_weights_evo2_vs_old_{num_players}p.json",
        f"best_weights_evo2_{num_players}p.json",
        "best_weights_evo2.json",
    ])
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            return Evo2AI.from_weights(name, data["weights"], seed=seed)
    print(
        "warning: no evo2 weights in saved_best_weights/ — using class "
        "defaults. Run `python -m scripts.evolve --ai evo2` and copy "
        "the result into saved_best_weights/ for evolved weights.",
        file=sys.stderr,
    )
    return Evo2AI(name, seed=seed)


def _evo4_factory(name: str, *, seed: int, num_players: int) -> Player:
    """Build an Evo4AI with GA-evolved weights for this seat count.

    Lookup order (first match wins), all paths rooted in
    ``saved_best_weights/``:

    1. ``best_weights_evo4_vs_all_{N}p.json`` — trained with the
       fitness averaged across every other bot (the
       ``python -m scripts.evolve --ai evo4`` default).
    2. ``best_weights_evo4_vs_evo1_{N}p.json`` — vs frozen evo1.
    3. ``best_weights_evo4_vs_evo2_{N}p.json`` — vs frozen evo2.
    4. ``best_weights_evo4_vs_evo3_{N}p.json`` — vs frozen evo3.
    5. ``best_weights_evo4_self_{N}p.json`` — self-play.
    6. ``best_weights_evo4_{N}p.json`` — legacy un-tagged.
    7. ``best_weights_evo4.json`` — global fallback.

    If none exist, falls back to ``Evo4AI``'s class defaults with a
    one-time stderr warning so ``--ai evo4`` still works without
    running the GA. Class defaults reduce to Evo3 behaviour (zero
    color-bias influence), so the fallback is the same strength as
    ``--ai evo3`` with defaults — no regression vs the previous AI.
    """
    candidates = _expand_weight_candidates([
        f"best_weights_evo4_vs_all_{num_players}p.json",
        f"best_weights_evo4_vs_evo1_{num_players}p.json",
        f"best_weights_evo4_vs_evo2_{num_players}p.json",
        f"best_weights_evo4_vs_evo3_{num_players}p.json",
        f"best_weights_evo4_self_{num_players}p.json",
        f"best_weights_evo4_{num_players}p.json",
        "best_weights_evo4.json",
    ])
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            return Evo4AI.from_weights(name, data["weights"], seed=seed)
    print(
        "warning: no evo4 weights in saved_best_weights/ — using class "
        "defaults (equivalent to Evo3). Run `python -m scripts.evolve --ai evo4` "
        "and copy the result into saved_best_weights/ for evolved weights.",
        file=sys.stderr,
    )
    return Evo4AI(name, seed=seed)


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
