"""Fitness evaluation for the ES trainer.

Two public entry points:

* :func:`evaluate_perturbations` — evaluates a batch of perturbed
  parameter vectors (the ``θ ± σ·εᵢ`` pairs the outer loop needs for a
  single gradient estimate). Returns one mean reward per perturbation.
* :func:`evaluate_theta` — evaluates a single parameter vector on a
  held-out seed range. Returns ``(mean_margin_reward, binary_win_rate)``
  so the trainer can log both the ES-relevant margin curve and a
  GA-comparable win-rate line on the same plot.

Parallelism mirrors :mod:`scripts.evolve.ga`:

* ``ProcessPoolExecutor`` backed by ``multiprocessing.get_context("fork")``
  — workers inherit the parent's module-level ``_WORKER_STATE`` via
  copy-on-write, so the opponent-provider closures in
  :mod:`scripts.evolve.opponents` (which are not picklable) cross the
  process boundary for free.
* ``_WORKER_STATE`` is populated in the parent *before* the pool is
  opened and cleared in a ``finally`` block so the parent does not
  retain stale references after the pool exits.
* Every game seed derives from pure loop indices (``seed_offset +
  prov_idx * 101 + chart_idx * 1000 + game_idx``), so results are
  deterministic at any worker count for a fixed ``seed_offset``.

Reward shaping:

* Per-game reward is the **normalized score margin**
  ``(mine − max_opp) / max(1, |mine| + |max_opp|)``, in roughly ``[-1, 1]``.
  Denser than the binary win signal the GA uses — important for ES,
  because rank-shaping in the trainer then has something more
  informative than a ``{0, 1}`` vector to sort.
* Binary win rate is computed in parallel and returned from
  :func:`evaluate_theta` for the GA-comparable plot line only. The
  gradient itself only ever sees the margin.
"""

from __future__ import annotations

import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from megagem.engine import is_game_over, play_round, score_game, setup_game

from scripts.evolve.opponents import OpponentProvider
from scripts.evolve.profiles import AIProfile


# Charts the GA trains on, duplicated here so the RL trainer doesn't
# have to import from ``scripts.evolve.ga`` (which drags in matplotlib).
CHARTS = "ABCDE"


# ---------------------------------------------------------------------------
# Per-game reward
# ---------------------------------------------------------------------------


def _game_reward(scores: list[dict], my_idx: int = 0) -> float:
    """Normalized score margin for one finished game.

    Returns ``(mine − max_opp) / max(1, |mine| + |max_opp|)``, bounded
    in roughly ``[-1, 1]``. Denser than the GA's binary win signal,
    which is the whole point of using ES with rank shaping on top.

    ``my_idx`` defaults to 0 because the trainer always seats the
    challenger in slot 0, same as :mod:`scripts.evolve.ga`.
    """
    mine = scores[my_idx]["total"]
    opp_best = max(
        s["total"] for i, s in enumerate(scores) if i != my_idx
    )
    denom = max(1.0, float(abs(mine) + abs(opp_best)))
    return (float(mine) - float(opp_best)) / denom


def _play_one_game(
    profile: AIProfile,
    weights: list[float],
    opponents: list,
    chart: str,
    seed: int,
) -> tuple[bool, float]:
    """Play one full game from the challenger's POV.

    Returns ``(won, reward)`` — the binary outright-win flag (for
    plotting) and the normalized score margin (for the ES gradient).
    Mirrors :func:`scripts.evolve.ga._play_one_game` but returns both
    signals so the trainer doesn't have to replay the same game twice.
    """
    challenger = profile.ai_class.from_weights(
        profile.label,
        weights,
        seed=seed * 7,
    )
    players = [challenger] + opponents
    state = setup_game(players, chart=chart, seed=seed)
    game_rng = random.Random(seed)
    while not is_game_over(state):
        play_round(state, rng=game_rng)
    scores = score_game(state)
    won = scores[0]["total"] > max(s["total"] for s in scores[1:])
    reward = _game_reward(scores, my_idx=0)
    return won, reward


# ---------------------------------------------------------------------------
# Rank transformation (Salimans-style fitness shaping)
# ---------------------------------------------------------------------------


def rank_transform(values: list[float]) -> list[float]:
    """Centered linear rank in ``[-0.5, +0.5]``.

    Salimans et al. 2017 call this "fitness shaping" and it is the
    default for NES/ES implementations. Concretely, we sort the
    input values, assign rank ``k`` to the ``k``-th smallest, then map
    rank ``k ∈ [0, N-1]`` to ``k/(N-1) − 0.5`` so the output is
    centered and unit-range.

    Invariant to monotone transforms of the reward, which is why we
    can use raw score margins (potentially noisy or skewed) as the
    input without worrying about outliers dominating the gradient.

    Ties are broken by index — the ``k``-th occurrence of a repeated
    value gets the ``k``-th available rank. This keeps the
    transformation deterministic (stable Python sort) without
    requiring the caller to perturb equal values.

    Single-element inputs return ``[0.0]`` rather than dividing by
    zero.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    # Stable sort by (value, original_index) so ties are broken
    # deterministically by index.
    order = sorted(range(n), key=lambda i: (values[i], i))
    out = [0.0] * n
    for rank, idx in enumerate(order):
        out[idx] = rank / (n - 1) - 0.5
    return out


# ---------------------------------------------------------------------------
# Fork-pool worker plumbing (mirrors scripts.evolve.ga)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Task:
    """One ``(perturbation, provider, chart, game)`` unit of fitness work.

    Mirrors :class:`scripts.evolve.ga._GameTask`. ``pert_idx`` and
    ``game_idx_in_pert`` are the merge keys used to slot results into
    the pre-allocated ``rewards[pert_idx][game_idx]`` accumulator so
    the post-pool sum is done in deterministic loop order.
    """

    pert_idx: int
    game_idx_in_pert: int
    provider_idx: int
    chart: str
    game_seed: int
    slot: int


# Module-level state read by fork-inherited workers. The parent
# populates this dict before opening the pool and clears it in a
# ``finally`` block. Workers treat it as read-only; the providers
# stored here are stateless closures over frozen data.
_WORKER_STATE: dict = {}


def _worker_play(task: _Task) -> tuple[int, int, bool, float]:
    """Process-pool worker body. One game per call.

    Reads profile / perturbed-thetas / providers from ``_WORKER_STATE``
    (inherited via fork copy-on-write), plays one game, returns the
    ``(pert_idx, game_idx_in_pert, won, reward)`` tuple for the parent
    to bucket.
    """
    profile: AIProfile = _WORKER_STATE["profile"]
    perturbed_thetas: list[list[float]] = _WORKER_STATE["perturbed_thetas"]
    providers = _WORKER_STATE["providers"]

    provider = providers[task.provider_idx][1]
    opponents = provider(task.slot, task.game_seed)
    won, reward = _play_one_game(
        profile,
        perturbed_thetas[task.pert_idx],
        opponents,
        task.chart,
        task.game_seed,
    )
    return (task.pert_idx, task.game_idx_in_pert, won, reward)


def _pool_chunksize(total_tasks: int, workers: int) -> int:
    """Pick a ``chunksize`` for ``ProcessPoolExecutor.map``.

    Same heuristic as :func:`scripts.evolve.ga._pool_chunksize`: aim
    for ~4 chunks per worker so a slow tail game can't leave a core
    idle, but chunks are big enough that IPC overhead stays
    negligible next to per-game compute.
    """
    if total_tasks <= 0 or workers <= 0:
        return 1
    return max(1, total_tasks // (workers * 4))


# ---------------------------------------------------------------------------
# Perturbation batch evaluation
# ---------------------------------------------------------------------------


def _build_tasks(
    perturbed_thetas: list[list[float]],
    providers: list[tuple[str, OpponentProvider]],
    *,
    games_per_chart: int,
    seed_offset: int,
) -> tuple[list[_Task], int]:
    """Enumerate all fitness work. Returns ``(tasks, games_per_pert)``.

    Factored out so both :func:`evaluate_perturbations` and
    :func:`evaluate_theta` can reuse the exact same seed layout.
    ``games_per_pert`` is the total number of games each perturbation
    plays across all providers × all charts, i.e. the pre-allocated
    row length of the reward accumulator.
    """
    tasks: list[_Task] = []
    games_per_pert = len(providers) * len(CHARTS) * games_per_chart
    for pert_idx in range(len(perturbed_thetas)):
        game_idx_in_pert = 0
        for prov_idx in range(len(providers)):
            prov_offset = seed_offset + prov_idx * 101
            for chart_idx, chart in enumerate(CHARTS):
                for game_idx in range(games_per_chart):
                    game_seed = (
                        prov_offset + chart_idx * 1000 + game_idx
                    )
                    slot = chart_idx * games_per_chart + game_idx
                    tasks.append(
                        _Task(
                            pert_idx=pert_idx,
                            game_idx_in_pert=game_idx_in_pert,
                            provider_idx=prov_idx,
                            chart=chart,
                            game_seed=game_seed,
                            slot=slot,
                        )
                    )
                    game_idx_in_pert += 1
    return tasks, games_per_pert


def evaluate_perturbations(
    profile: AIProfile,
    perturbed_thetas: list[list[float]],
    *,
    providers: list[tuple[str, OpponentProvider]],
    games_per_chart: int,
    seed_offset: int,
    workers: int,
) -> list[float]:
    """Return one mean-margin-reward per perturbation.

    Deterministic at any worker count: results accumulate into a
    pre-allocated ``list[list[float]]`` indexed by
    ``(pert_idx, game_idx_in_pert)``, then summed *after* the pool
    drains in loop order. Float sums are order-independent by
    construction, not by luck.

    Determinism-preserving structure is the whole reason this is a
    two-pass algorithm (build tasks → run pool → sum rows) rather than
    a one-pass accumulator. Do not "optimize" it by folding the sum
    into the worker callback — that reintroduces pool-order dependence.
    """
    if not perturbed_thetas:
        return []

    tasks, games_per_pert = _build_tasks(
        perturbed_thetas,
        providers,
        games_per_chart=games_per_chart,
        seed_offset=seed_offset,
    )

    # Pre-allocated accumulator. ``rewards[pert_idx][game_idx_in_pert]``
    # is set once per task; unset slots are a bug that will show up as
    # the default 0.0 and bias the mean, so we validate every slot is
    # touched below.
    rewards: list[list[float]] = [
        [0.0] * games_per_pert for _ in range(len(perturbed_thetas))
    ]
    touched: list[list[bool]] = [
        [False] * games_per_pert for _ in range(len(perturbed_thetas))
    ]

    if workers <= 1:
        for task in tasks:
            provider = providers[task.provider_idx][1]
            opponents = provider(task.slot, task.game_seed)
            _won, reward = _play_one_game(
                profile,
                perturbed_thetas[task.pert_idx],
                opponents,
                task.chart,
                task.game_seed,
            )
            rewards[task.pert_idx][task.game_idx_in_pert] = reward
            touched[task.pert_idx][task.game_idx_in_pert] = True
    else:
        _WORKER_STATE["profile"] = profile
        _WORKER_STATE["perturbed_thetas"] = perturbed_thetas
        _WORKER_STATE["providers"] = providers
        try:
            ctx = multiprocessing.get_context("fork")
            cs = _pool_chunksize(len(tasks), workers)
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
            ) as executor:
                for pert_idx, game_idx, _won, reward in executor.map(
                    _worker_play, tasks, chunksize=cs
                ):
                    rewards[pert_idx][game_idx] = reward
                    touched[pert_idx][game_idx] = True
        finally:
            _WORKER_STATE.clear()

    # Sanity-check: every slot must have been written. A miss means
    # the pool dropped a task silently (would bias the mean toward 0).
    for pert_idx, row in enumerate(touched):
        if not all(row):
            missing = [i for i, v in enumerate(row) if not v]
            raise RuntimeError(
                f"evaluate_perturbations: perturbation {pert_idx} "
                f"missing game results at indices {missing[:5]}"
            )

    # Sum each row in deterministic loop order.
    return [sum(row) / games_per_pert for row in rewards]


def evaluate_theta(
    profile: AIProfile,
    theta: list[float],
    *,
    providers: list[tuple[str, OpponentProvider]],
    games_per_chart: int,
    seed_offset: int,
    workers: int,
) -> tuple[float, float]:
    """Evaluate a single parameter vector on a held-out seed range.

    Returns ``(mean_margin_reward, binary_win_rate)`` — the margin is
    the ES-relevant training curve; the win rate is the GA-comparable
    plot line the trainer shows alongside it.

    Reuses :func:`_build_tasks` so the seed layout is bit-for-bit
    identical to :func:`evaluate_perturbations` — which is what lets
    the trainer use a *different* ``seed_offset`` for the held-out
    evaluation without re-implementing the enumeration.
    """
    # Single perturbation, so the "perturbations" list has one entry.
    tasks, games_per_pert = _build_tasks(
        [theta],
        providers,
        games_per_chart=games_per_chart,
        seed_offset=seed_offset,
    )
    rewards = [0.0] * games_per_pert
    wins = [False] * games_per_pert
    touched = [False] * games_per_pert

    if workers <= 1:
        for task in tasks:
            provider = providers[task.provider_idx][1]
            opponents = provider(task.slot, task.game_seed)
            won, reward = _play_one_game(
                profile,
                theta,
                opponents,
                task.chart,
                task.game_seed,
            )
            rewards[task.game_idx_in_pert] = reward
            wins[task.game_idx_in_pert] = won
            touched[task.game_idx_in_pert] = True
    else:
        _WORKER_STATE["profile"] = profile
        _WORKER_STATE["perturbed_thetas"] = [theta]
        _WORKER_STATE["providers"] = providers
        try:
            ctx = multiprocessing.get_context("fork")
            cs = _pool_chunksize(len(tasks), workers)
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
            ) as executor:
                for _pert_idx, game_idx, won, reward in executor.map(
                    _worker_play, tasks, chunksize=cs
                ):
                    rewards[game_idx] = reward
                    wins[game_idx] = won
                    touched[game_idx] = True
        finally:
            _WORKER_STATE.clear()

    if not all(touched):
        missing = [i for i, v in enumerate(touched) if not v]
        raise RuntimeError(
            f"evaluate_theta: missing game results at indices {missing[:5]}"
        )

    mean_reward = sum(rewards) / games_per_pert
    win_rate = sum(1 for w in wins if w) / games_per_pert
    return mean_reward, win_rate
