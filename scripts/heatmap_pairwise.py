"""Pairwise win-rate heatmap for every AI vs every other AI.

Each cell M[row, col] is the win rate of one ``row`` AI seated against
three copies of ``col``, averaged across all five value charts on a
fixed seed range. The seed range starts at ``SEED_START`` and is held
out from every GA's training seeds (0..9 for the old GA, rotating
seeds based on ``(seed + gen) * 9973`` for the evo2 GA), so the
numbers reflect generalisation rather than memorisation.

The matrix runs 4-player games (1 challenger + 3 opponents), so we
load every evolved AI's *4-player* weights file in preference to
unsuffixed legacy files.

Run::

    python -m scripts.heatmap_pairwise
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from megagem.engine import is_game_over, play_round, score_game, setup_game
from megagem.players import (
    AdaptiveHeuristicAI,
    Evo2AI,
    Evo3AI,
    HeuristicAI,
    HyperAdaptiveAI,
    HyperAdaptiveSplitAI,
    HypergeometricAI,
    RandomAI,
)

CHARTS = "ABCDE"
SEED_START = 200       # held out: GA trained on 0..9
GAMES_PER_CHART = 200  # 1000 games per cell (5 charts × 200 seeds)


# Weights are looked up ONLY in `saved_best_weights/` (checked-in
# snapshots). `artifacts/` holds intermediate GA output that may not
# reflect the canonical "best ever" — promote a fresh run by copying
# its weights file into `saved_best_weights/`.
_WEIGHTS_DIR = Path("saved_best_weights")


def _try_load(*filenames: str) -> list[float] | None:
    """Return the weights from the first existing candidate path, else None."""
    for filename in filenames:
        path = _WEIGHTS_DIR / filename
        if path.exists():
            return json.loads(path.read_text())["weights"]
    return None


def _load_evolved() -> list[float] | None:
    """Load HyperAdaptiveSplitAI 4-player weights if present.

    Prefers ``best_weights_4p.json`` (per-seat-count) over the legacy
    unsuffixed ``best_weights.json``. Returns None (rather than raising)
    so the heatmap can still run when the old GA hasn't been trained —
    the cell for the old evolved AI is simply omitted.
    """
    return _try_load(
        "best_weights_4p.json",
        "best_weights.json",
    )


def make_factories() -> dict:
    factories: dict = {
        "Random":       lambda name, seed: RandomAI(name, seed=seed),
        "Heuristic":    lambda name, seed: HeuristicAI(name, seed=seed),
        "Adaptive":     lambda name, seed: AdaptiveHeuristicAI(name, seed=seed),
        "Hyper":        lambda name, seed: HypergeometricAI(name, seed=seed),
        "HyperAdapt":   lambda name, seed: HyperAdaptiveAI(name, seed=seed),
    }

    evolved = _load_evolved()
    if evolved is not None:
        factories["EvolvedSplit"] = lambda name, seed: HyperAdaptiveSplitAI.from_weights(
            name, evolved, seed=seed
        )

    # Evo2: prefer the per-seat-count file but fall back to class defaults
    # so the column is always present in the heatmap — otherwise a fresh
    # clone without an Evo2 GA run would drop the most interesting row.
    evo2 = _try_load(
        "best_weights_evo2_vs_all_4p.json",
        "best_weights_evo2_vs_old_evo2_4p.json",
        "best_weights_evo2_vs_old_4p.json",
        "best_weights_evo2_self_4p.json",
        "best_weights_evo2_4p.json",
        "best_weights_evo2.json",
    )
    if evo2 is not None:
        factories["Evo2"] = lambda name, seed: Evo2AI.from_weights(name, evo2, seed=seed)
    else:
        factories["Evo2"] = lambda name, seed: Evo2AI(name, seed=seed)

    # Evo3: same deal — always include it so the whole point of this
    # script (comparing Evo3's opponent-pricing head vs everything else)
    # is visible even on a fresh checkout.
    evo3 = _try_load(
        "best_weights_evo3_vs_all_4p.json",
        "best_weights_evo3_vs_evo2_4p.json",
        "best_weights_evo3_self_4p.json",
        "best_weights_evo3_4p.json",
        "best_weights_evo3.json",
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


def build_matrix(factories: dict) -> tuple[list[str], np.ndarray]:
    names = list(factories.keys())
    n = len(names)
    M = np.zeros((n, n))
    for i, row in enumerate(names):
        for j, col in enumerate(names):
            M[i, j] = winrate(factories[row], factories[col])
            print(f"  {row:>13} vs 3x {col:<13} = {M[i, j] * 100:5.1f}%")
    return names, M


def save_heatmap(names: list[str], M: np.ndarray, path: Path) -> None:
    n = len(names)
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("3× opponent")
    ax.set_ylabel("1× challenger")
    ax.set_title(
        "Pairwise win rates: 1 challenger vs 3× opponent\n"
        f"({GAMES_PER_CHART * len(CHARTS)} games/cell, charts {CHARTS}, "
        f"seeds {SEED_START}..{SEED_START + GAMES_PER_CHART - 1})"
    )

    for i in range(n):
        for j in range(n):
            v = M[i, j]
            # White text on dark cells, black on light.
            color = "white" if v < 0.25 or v > 0.75 else "black"
            ax.text(j, i, f"{int(round(v * 100))}%",
                    ha="center", va="center", color=color, fontsize=11)

    fig.colorbar(im, ax=ax, label="win rate", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def print_table(names: list[str], M: np.ndarray) -> None:
    col_w = max(len(n) for n in names) + 2
    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in names)
    print("\n" + header)
    print("-" * len(header))
    for i, row in enumerate(names):
        line = f"{row:<{col_w}}"
        for j in range(len(names)):
            line += f"{int(round(M[i, j] * 100)):>{col_w - 1}}%"
        print(line)


def main() -> None:
    factories = make_factories()
    print(f"Building {len(factories)}x{len(factories)} pairwise matrix "
          f"({GAMES_PER_CHART * len(CHARTS)} games/cell)...")
    names, M = build_matrix(factories)
    print_table(names, M)

    out = Path("artifacts/heatmap_pairwise.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_heatmap(names, M, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
