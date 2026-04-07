"""Entry point: `python -m megagem [...]`."""

from __future__ import annotations

import argparse
import random
import sys

from . import render
from .engine import is_game_over, play_round, score_game, setup_game
from .players import HeuristicAI, HumanPlayer, Player, RandomAI


AI_TYPES = {
    "random": RandomAI,
    "heuristic": HeuristicAI,
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
    ai_cls = AI_TYPES[ai_type]
    players: list[Player] = []
    if all_ai:
        for i in range(num_players):
            players.append(ai_cls(ai_names[i], seed=rng.randrange(2**31)))
    else:
        players.append(HumanPlayer("You", debug=debug))
        for i in range(num_players - 1):
            players.append(ai_cls(ai_names[i], seed=rng.randrange(2**31)))
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
                        help="Show all opponents' hands when rendering the board.")
    parser.add_argument("--all-ai", action="store_true",
                        help="Replace the human with another RandomAI (smoke testing).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-round output (handy with --all-ai).")
    parser.add_argument("--ai", type=str, default="heuristic", choices=list(AI_TYPES),
                        help="Which AI to use for opponents (and --all-ai players).")
    args = parser.parse_args(argv)

    players = build_players(args.players, args.all_ai, args.debug, args.seed, args.ai)
    state = setup_game(players, chart=args.chart, seed=args.seed)

    rng = random.Random(args.seed)
    has_human = any(p.is_human for p in players)

    if not args.quiet:
        print(f"Starting MegaGem with {args.players} players, chart {args.chart}.")
        print()

    while not is_game_over(state):
        if has_human and not args.quiet:
            # Show the board to the human player before bids are collected.
            # (HumanPlayer.choose_bid will print it again from their perspective,
            # but we want a quick "between rounds" view too.)
            pass
        result = play_round(state, rng=rng)
        if not args.quiet:
            print()
            print(render.render_round_result(result, state))
            print()

    scores = score_game(state)
    print()
    print(render.render_scores(scores))
    return 0


if __name__ == "__main__":
    sys.exit(main())
