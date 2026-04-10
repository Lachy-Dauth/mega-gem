"""Generate pairwise heatmap data as JSON for the web bots page.

Reuses the same simulation logic as heatmap_pairwise.py but outputs
a JSON file instead of a matplotlib image. Run from research/:

    python -m scripts.generate_heatmap_json

Writes to ../web/heatmap_data.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from megagem.engine import is_game_over, play_round, score_game, setup_game
from megagem.players import (
    Evo2AI,
    Evo3AI,
    HeuristicAI,
    HyperAdaptiveSplitAI,
    RandomAI,
)

CHARTS = "ABCDE"
SEED_START = 200
GAMES_PER_CHART = 40  # 200 games per cell (5 charts × 40 seeds)

_WEIGHTS_DIR = Path("saved_best_weights")


def _try_load(*filenames: str) -> list[float] | None:
    for filename in filenames:
        path = _WEIGHTS_DIR / filename
        if path.exists():
            return json.loads(path.read_text())["weights"]
    return None


def make_factories() -> dict:
    factories: dict = {
        "Random": lambda name, seed: RandomAI(name, seed=seed),
        "Heuristic": lambda name, seed: HeuristicAI(name, seed=seed),
    }

    evolved = _try_load("best_weights_4p.json", "best_weights.json")
    if evolved is not None:
        factories["Evolved"] = lambda name, seed: HyperAdaptiveSplitAI.from_weights(
            name, evolved, seed=seed
        )

    evo2 = _try_load(
        "best_weights_evo2_vs_all_4p.json",
        "best_weights_evo2_vs_old_4p.json",
        "best_weights_evo2_self_4p.json",
    )
    if evo2 is not None:
        factories["Evo2"] = lambda name, seed: Evo2AI.from_weights(name, evo2, seed=seed)
    else:
        factories["Evo2"] = lambda name, seed: Evo2AI(name, seed=seed)

    evo3 = _try_load(
        "best_weights_evo3_vs_all_4p.json",
        "best_weights_evo3_vs_evo2_4p.json",
        "best_weights_evo3_self_4p.json",
    )
    if evo3 is not None:
        factories["Evo3"] = lambda name, seed: Evo3AI.from_weights(name, evo3, seed=seed)
    else:
        factories["Evo3"] = lambda name, seed: Evo3AI(name, seed=seed)

    return factories


def play_one(challenger_factory, opponent_factory, chart: str, seed: int) -> bool:
    players = [
        challenger_factory("C", seed * 7),
        opponent_factory("O1", seed * 7 + 1),
        opponent_factory("O2", seed * 7 + 2),
        opponent_factory("O3", seed * 7 + 3),
    ]
    state = setup_game(players, chart=chart, seed=seed)
    rng = random.Random(seed)
    while not is_game_over(state):
        play_round(state, rng=rng)
    scores = score_game(state)
    return scores[0]["total"] > max(s["total"] for s in scores[1:])


def winrate(challenger_factory, opponent_factory) -> float:
    wins = 0
    total = 0
    for chart in CHARTS:
        for s in range(SEED_START, SEED_START + GAMES_PER_CHART):
            if play_one(challenger_factory, opponent_factory, chart, s):
                wins += 1
            total += 1
    return wins / total


def main() -> None:
    factories = make_factories()
    names = list(factories.keys())
    n = len(names)
    total_games = GAMES_PER_CHART * len(CHARTS)

    print(f"Building {n}x{n} pairwise matrix ({total_games} games/cell)...")

    matrix: list[list[float]] = []
    for i, row in enumerate(names):
        row_data: list[float] = []
        for j, col in enumerate(names):
            wr = winrate(factories[row], factories[col])
            row_data.append(round(wr, 3))
            print(f"  {row:>13} vs 3x {col:<13} = {wr * 100:5.1f}%")
        matrix.append(row_data)

    data = {
        "names": names,
        "matrix": matrix,
        "games_per_cell": total_games,
        "charts": CHARTS,
        "seed_start": SEED_START,
        "games_per_chart": GAMES_PER_CHART,
        "description": "Win rate of 1 challenger (row) vs 3 copies of opponent (column)",
    }

    out = Path(__file__).resolve().parent.parent.parent / "web" / "heatmap_data.json"
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
