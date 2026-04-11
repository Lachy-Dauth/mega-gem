"""AI factory for multiplayer seats.

This is a trimmed version of the logic in
``research/megagem/__main__.py`` — the CLI lives in an ``argparse``
script so we can't import its factories without side effects. We keep
the same AI names so the browser client and the CLI share vocabulary.

Weights are loaded from ``research/saved_best_weights/`` (since the
server's working directory at deploy time is the repo root, we
resolve the path explicitly). Missing weights fall back to class
defaults — the server should never *crash* because Evo3 weights
aren't available, it just runs with vanilla constants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from megagem.players import (
    Evo2AI,
    Evo3AI,
    Evo4AI,
    HeuristicAI,
    HyperAdaptiveSplitAI,
    Player,
    RandomAI,
)


# research/ sits one level up from server/ — the weights are checked
# into research/saved_best_weights/.
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "research" / "saved_best_weights"

AIFactory = Callable[..., Player]


def _candidate_weight_paths(profile_key: str, num_players: int) -> list[Path]:
    """Uniform lookup chain for every evo profile.

    Mirrors :func:`scripts.evolve.opponents.candidate_filenames` and
    :func:`megagem.__main__._candidate_weight_paths` so the server, CLI,
    and GA tuner all agree on which files to search and in what order.
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
    weights = _load_evo_weights("evo1", num_players)
    if weights is None:
        # No pre-Evo2 GA weights available — fall back to the
        # hand-tuned defaults so the seat still has a plausible bot.
        return HyperAdaptiveSplitAI(name, seed=seed)
    return HyperAdaptiveSplitAI.from_weights(name, weights, seed=seed)


def _evo2_factory(name: str, *, seed: int, num_players: int) -> Player:
    weights = _load_evo_weights("evo2", num_players)
    if weights is None:
        return Evo2AI(name, seed=seed)
    return Evo2AI.from_weights(name, weights, seed=seed)


def _evo3_factory(name: str, *, seed: int, num_players: int) -> Player:
    weights = _load_evo_weights("evo3", num_players)
    if weights is None:
        return Evo3AI(name, seed=seed)
    return Evo3AI.from_weights(name, weights, seed=seed)


def _evo4_factory(name: str, *, seed: int, num_players: int) -> Player:
    weights = _load_evo_weights("evo4", num_players)
    if weights is None:
        return Evo4AI(name, seed=seed)
    return Evo4AI.from_weights(name, weights, seed=seed)


AI_FACTORIES: dict[str, AIFactory] = {
    "random":      lambda name, *, seed, num_players: RandomAI(name, seed=seed),
    "heuristic":   lambda name, *, seed, num_players: HeuristicAI(name, seed=seed),
    "evolved":     _evolved_factory,
    "evo2":        _evo2_factory,
    "evo3":        _evo3_factory,
    "evo4":        _evo4_factory,
}


AI_KINDS = tuple(AI_FACTORIES.keys())


def build_ai(kind: str, name: str, *, seed: int, num_players: int) -> Player:
    if kind not in AI_FACTORIES:
        raise ValueError(f"Unknown AI kind: {kind!r}")
    return AI_FACTORIES[kind](name, seed=seed, num_players=num_players)
