"""Opponent provider registry for the unified GA tuner.

The eight modes — ``vs_all``, ``vs_random``, ``vs_heuristic``,
``vs_evo1``, ``vs_evo2``, ``vs_evo3``, ``vs_evo4``, and ``self_play`` —
are uniform across all four AIs. Every mode resolves to a list of
``(label, OpponentProvider)`` pairs that the GA loop pools fitness
across; single-opponent modes return a one-element list, the multi-
opponent ``vs_all`` mode returns one entry per opponent class minus the
challenger's own class. Self-play rebuilds its provider each generation
(opponents are sampled from the live population).

The legacy per-AI quirks (``old_evo`` only for evo2, ``vs_evo3`` only
for evo4, etc.) are gone — every AI supports the full mode set, so
training Evo3 against a frozen Evo2 *and* training Evo4 against a
frozen Evo2 are spelled the same way: ``--opponent vs_evo2``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from megagem.players import (
    HeuristicAI,
    Player,
    RandomAI,
)

from .profiles import AI_PROFILES, AIProfile


# An OpponentProvider builds the opponent slate for one game. It takes
# the zero-based game-slot index (within the current generation) and
# the game seed, and returns ``num_players − 1`` ready-to-play Players.
OpponentProvider = Callable[[int, int], list[Player]]


# The eight uniform mode keys. CLI ``--opponent`` only accepts these.
MODE_KEYS = (
    "vs_all",
    "vs_random",
    "vs_heuristic",
    "vs_evo1",
    "vs_evo2",
    "vs_evo3",
    "vs_evo4",
    "self_play",
)


# Filename tag per mode. Plugged into the artifact filenames as
# ``best_weights_evo{K}_{tag}_{N}p.json``. ``self_play`` shortens to
# ``self`` to match the legacy convention.
MODE_FILENAME_TAGS: dict[str, str] = {
    "vs_all":       "vs_all",
    "vs_random":    "vs_random",
    "vs_heuristic": "vs_heuristic",
    "vs_evo1":      "vs_evo1",
    "vs_evo2":      "vs_evo2",
    "vs_evo3":      "vs_evo3",
    "vs_evo4":      "vs_evo4",
    "self_play":    "self",
}


# --- Weight loading ---------------------------------------------------------


# Saved weights live under ``research/saved_best_weights/``. The CLI
# script is invoked from ``research/`` (per the README), so a relative
# path resolves to the right place.
_WEIGHTS_DIR = Path("saved_best_weights")


def candidate_filenames(profile_key: str, num_players: int) -> list[str]:
    """Return the uniform lookup chain for any profile.

    Same chain for every evo profile — just the profile key varies.
    First existing match wins; the un-tagged ``best_weights_{key}_{N}p``
    entry is the fallback for pre-tag files (none of the
    checked-in weights actually use it right now, but we keep it so the
    GA can pick up a weights file the user dropped in by hand without
    worrying about mode tags).
    """
    return [
        f"best_weights_{profile_key}_vs_all_{num_players}p.json",
        f"best_weights_{profile_key}_vs_random_{num_players}p.json",
        f"best_weights_{profile_key}_vs_heuristic_{num_players}p.json",
        f"best_weights_{profile_key}_vs_evo1_{num_players}p.json",
        f"best_weights_{profile_key}_vs_evo2_{num_players}p.json",
        f"best_weights_{profile_key}_vs_evo3_{num_players}p.json",
        f"best_weights_{profile_key}_vs_evo4_{num_players}p.json",
        f"best_weights_{profile_key}_self_{num_players}p.json",
        f"best_weights_{profile_key}_{num_players}p.json",
    ]


@dataclass(frozen=True)
class LoadedWeights:
    """Result of looking up a profile's weights file."""

    weights: list[float]
    source: Path  # real path on disk OR a sentinel like "<class defaults>"


def load_profile_weights(
    profile: AIProfile,
    num_players: int,
) -> LoadedWeights:
    """Return the best available weights for ``profile`` at this seat count.

    Walks the uniform :func:`candidate_filenames` chain (rooted in
    ``saved_best_weights/``), returning the first existing match. If
    none exist, falls back to ``profile.flatten_defaults()`` (which
    delegates to the AI class's own classmethod) and reports a
    sentinel source path so the caller can log "class defaults".

    Validates that any loaded file matches the profile's expected
    ``num_weights`` — a length mismatch means the file is for a
    different AI version, and silently ignoring it would lead to a
    confusing crash inside the GA loop.
    """
    for filename in candidate_filenames(profile.key, num_players):
        path = _WEIGHTS_DIR / filename
        if path.exists():
            data = json.loads(path.read_text())
            weights = data["weights"]
            if len(weights) != profile.num_weights:
                raise SystemExit(
                    f"{path}: expected {profile.num_weights} weights "
                    f"for {profile.label}, got {len(weights)}"
                )
            return LoadedWeights(weights=weights, source=path)
    # No file on disk — fall back to the AI class's hardcoded defaults.
    print(
        f"no saved weights found for {profile.label} at {num_players}p, "
        f"falling back to class defaults"
    )
    return LoadedWeights(
        weights=profile.flatten_defaults(),
        source=Path(f"<{profile.label} class defaults>"),
    )


# --- Provider builders ------------------------------------------------------


def _make_class_provider(
    ai_class: type[Player],
    *,
    num_players: int,
) -> OpponentProvider:
    """Provider that fills opponent seats with class-default instances.

    Used for :class:`RandomAI` and :class:`HeuristicAI`, which have no
    tuneable weights. Each game gets a fresh seed-derived RNG so the
    class's internal tie-breaks decorrelate across games.
    """
    def provider(slot: int, game_seed: int) -> list[Player]:
        return [
            ai_class(f"O{k + 1}", seed=game_seed + k + 1)
            for k in range(num_players - 1)
        ]
    return provider


def _make_weight_provider(
    profile: AIProfile,
    weights: list[float],
    *,
    num_players: int,
) -> OpponentProvider:
    """Provider that fills opponent seats with one frozen-weight evo bot.

    Constructs each opponent via ``profile.ai_class.from_weights`` so
    the same code path handles all four evo profiles uniformly.
    """
    def provider(slot: int, game_seed: int) -> list[Player]:
        return [
            profile.ai_class.from_weights(
                f"{profile.key}_{k + 1}",
                weights,
                seed=game_seed + k + 1,
            )
            for k in range(num_players - 1)
        ]
    return provider


def _make_self_play_provider(
    profile: AIProfile,
    population: list[list[float]],
    *,
    num_players: int,
    sample_rng: random.Random,
    n_games: int,
) -> OpponentProvider:
    """Provider that samples opponent slates from the current population.

    Opponent indices for each game slot are pre-sampled *once* per
    generation and reused for every individual. Across individuals the
    seeds and slate are identical; only the challenger differs. This is
    what lets tournament selection compare scores honestly.

    Self-as-opponent is allowed (probability 1/pop_size) — filtering it
    would bias sampling per individual or require resampling loops.
    """
    pop_size = len(population)
    opp_indices_per_slot: list[list[int]] = [
        [sample_rng.randrange(pop_size) for _ in range(num_players - 1)]
        for _ in range(n_games)
    ]

    def provider(slot: int, game_seed: int) -> list[Player]:
        return [
            profile.ai_class.from_weights(
                f"O{k + 1}",
                population[opp_indices_per_slot[slot][k]],
                seed=game_seed + k + 1,
            )
            for k in range(num_players - 1)
        ]

    return provider


# --- Mode dispatcher --------------------------------------------------------


def build_mode_providers(
    mode_key: str,
    profile: AIProfile,
    *,
    num_players: int,
    population: list[list[float]] | None = None,
    sample_rng: random.Random | None = None,
    n_games: int = 0,
    quiet: bool = False,
) -> list[tuple[str, OpponentProvider]]:
    """Build the ``(label, provider)`` list for one opponent mode.

    The GA loop calls this once per generation (rebuilt every gen for
    self-play because the population evolves; rebuilt for symmetry in
    fixed-opponent modes too).

    Single-opponent modes return a one-element list. The ``vs_all``
    mode returns one entry per opponent class — Random, Heuristic, and
    every evo profile *other than* the challenger's own. The pooled
    fitness then naturally averages across opponent types because every
    provider gets the same number of games.
    """
    if mode_key not in MODE_KEYS:
        raise ValueError(f"unknown opponent mode: {mode_key!r}")

    if mode_key == "vs_random":
        return [("Random", _make_class_provider(RandomAI, num_players=num_players))]

    if mode_key == "vs_heuristic":
        return [("Heuristic", _make_class_provider(HeuristicAI, num_players=num_players))]

    if mode_key in ("vs_evo1", "vs_evo2", "vs_evo3", "vs_evo4"):
        target_key = mode_key[len("vs_"):]
        target_profile = AI_PROFILES[target_key]
        loaded = load_profile_weights(target_profile, num_players)
        if not quiet:
            print(f"{mode_key}: {target_profile.label} opponents loaded from {loaded.source}")
        return [(
            target_profile.label,
            _make_weight_provider(target_profile, loaded.weights, num_players=num_players),
        )]

    if mode_key == "self_play":
        if population is None or sample_rng is None or n_games <= 0:
            raise ValueError(
                "self_play requires population, sample_rng, and n_games"
            )
        return [(
            f"{profile.label}-self",
            _make_self_play_provider(
                profile,
                population,
                num_players=num_players,
                sample_rng=sample_rng,
                n_games=n_games,
            ),
        )]

    # vs_all: Random + Heuristic + every evo profile EXCEPT the challenger.
    providers: list[tuple[str, OpponentProvider]] = [
        ("Random",    _make_class_provider(RandomAI, num_players=num_players)),
        ("Heuristic", _make_class_provider(HeuristicAI, num_players=num_players)),
    ]
    for other_key in ("evo1", "evo2", "evo3", "evo4"):
        if other_key == profile.key:
            continue
        other_profile = AI_PROFILES[other_key]
        loaded = load_profile_weights(other_profile, num_players)
        if not quiet:
            print(
                f"vs_all: {other_profile.label} opponents loaded from {loaded.source}"
            )
        providers.append((
            other_profile.label,
            _make_weight_provider(other_profile, loaded.weights, num_players=num_players),
        ))
    return providers
