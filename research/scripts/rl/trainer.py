"""Evolution-Strategies policy-gradient loop.

This is the core of :mod:`scripts.rl`. :func:`run_es` tunes the weights
of an :class:`~scripts.evolve.profiles.AIProfile` (default: ``evo4``)
against one of the eight opponent modes from
:mod:`scripts.evolve.opponents` using parameter-space ES with mirrored
sampling, centered rank shaping, and an Adam update.

Algorithm (one generation):

1. Sample ``N/2`` noise vectors ``εᵢ ~ 𝒩(0, I)`` via stdlib
   :func:`random.gauss` and the trainer's seeded RNG. ``N`` is the
   requested population size, which must be even.
2. Build ``N`` perturbed parameter vectors ``θ±ᵢ = θ ± σ·εᵢ``, clipped
   component-wise to ``[-theta_clip, +theta_clip]``.
3. Evaluate each perturbed parameter across ``games_per_chart × 5 ×
   len(providers)`` games via :func:`scripts.rl.fitness.evaluate_perturbations`.
   Per-game reward is the normalized score margin.
4. Apply centered rank shaping via :func:`scripts.rl.fitness.rank_transform`.
5. Gradient estimate: ``g[k] = (1 / (N·σ)) · Σᵢ Rᵢ · (pertᵢ[k] - θ[k]) / σ``
   — equivalent to the ``ε`` form but reconstructs ``ε`` from the
   stored perturbations so it is robust to re-clipping. (In practice
   we use the stored ``ε`` directly; the reconstruction is only a
   fallback mental model.)
6. Adam step on ``θ``.
7. Re-clip ``θ`` to ``[-theta_clip, +theta_clip]``.
8. Evaluate the updated ``θ`` on a held-out seed range for the
   training curve.

Unlike the GA — which rotates game seeds per generation — the ES
trainer uses a **constant** seed slate for the whole run. Training
seeds are ``(seed + 1) * 9973`` and held-out θ-eval seeds are
``(seed + 1000) * 9973``, both fixed across every generation.

This is the Salimans et al. 2017 convention, and it is not an
implementation detail — it is load-bearing for the gradient
estimator. The ES gradient ``g[k] = (1/(N·σ)) · Σᵢ Rᵢ·εᵢ[k]``
implicitly assumes every perturbation's reward is drawn from the
*same* distribution; rotating seeds between generations mixes "θ
improved" with "games got easier/harder" and the optimizer chases
seed noise. Constant seeds remove that confound — a reward
improvement across generations is then *actually* an improvement
on the same games, and the held-out curve becomes a clean
"objective value at θ" rather than a randomized re-eval.

The trade-off: θ can in principle overfit to the specific seed
slate. In practice the 48 perturbations × ``games_per_chart`` × 5
charts × ``len(providers)`` games per generation is a large enough
effective sample that overfitting is not observed within a
30-generation run. If a run does show drift, rerun with a larger
``--games-per-chart`` or a different ``--seed``.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# NOTE: matplotlib is *not* imported at module load time. It's imported
# lazily inside :func:`save_history_plot` so the trainer can be
# imported — and its pure-Python ES loop tested — in stdlib-only
# environments where matplotlib isn't installed. This mirrors the
# CLAUDE.md dependency contract: the engine / CLI / tests are
# stdlib-only; only the plot-producing scripts need matplotlib.

from scripts.evolve.opponents import (
    OpponentProvider,
    build_mode_providers,
    load_profile_weights,
)
from scripts.evolve.profiles import AIProfile

from .fitness import (
    CHARTS,
    evaluate_perturbations,
    evaluate_theta,
    rank_transform,
)
from .optim import Adam


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ESResult:
    """Return value of :func:`run_es`.

    Contains everything the CLI, the test suite, and a resume
    operation need: the final ``θ``, the per-generation curves for
    plotting, and the Adam / RNG state for exact resumption.
    """

    theta: list[float]
    mean_reward_per_gen: list[float] = field(default_factory=list)
    win_rate_per_gen: list[float] = field(default_factory=list)
    training_mean_per_gen: list[float] = field(default_factory=list)
    best_mean_reward: float = -float("inf")
    best_generation: int = 0
    best_theta: list[float] = field(default_factory=list)
    best_win_rate: float = 0.0
    initial_mean_reward: float = 0.0
    initial_win_rate: float = 0.0
    adam_state: dict[str, Any] = field(default_factory=dict)
    rng_state: Any = None
    generations_completed: int = 0


# ---------------------------------------------------------------------------
# Epsilon sampling
# ---------------------------------------------------------------------------


def sample_epsilons(
    n_pairs: int,
    dim: int,
    rng: random.Random,
) -> list[list[float]]:
    """Return ``2·n_pairs`` noise vectors in mirrored order.

    Layout: ``[ε₀, -ε₀, ε₁, -ε₁, …]``. Callers consume them by pair
    index via ``epsilons[2*k]`` / ``epsilons[2*k + 1]``. Interleaving
    (rather than concatenating all positive vectors then all negative)
    keeps the per-pair locality tight for any future code that
    iterates pairs — and makes the gradient estimation loop a trivial
    ``for i in range(2·n_pairs)``.

    Uses :meth:`random.Random.gauss` so the sampling is seedable via
    the trainer's own RNG.
    """
    out: list[list[float]] = []
    for _ in range(n_pairs):
        eps = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        neg = [-x for x in eps]
        out.append(eps)
        out.append(neg)
    return out


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------


def _render_progress(
    gen: int,
    generations: int,
    mean_reward: float,
    win_rate: float,
    elapsed_total: float,
    bar_width: int = 30,
) -> None:
    """In-place progress bar. Mirrors :func:`scripts.evolve.ga._render_progress`.

    Uses ``\\r`` so successive calls overwrite the same terminal line.
    Shows mean reward (the ES objective) and win rate (GA-comparable
    signal) of the current ``θ`` on the held-out seeds.
    """
    done = gen + 1
    frac = done / generations
    filled = int(round(frac * bar_width))
    bar = "#" * filled + "-" * (bar_width - filled)
    avg = elapsed_total / done
    eta = avg * (generations - done)
    line = (
        f"\r[{bar}] gen {done:3d}/{generations} "
        f"θ-eval reward={mean_reward:+.3f} win={win_rate:.2f} "
        f"elapsed={elapsed_total:5.1f}s eta={eta:5.1f}s"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Theta clipping
# ---------------------------------------------------------------------------


def _clip(theta: list[float], clip: float) -> list[float]:
    """Component-wise clip to ``[-clip, +clip]``. Returns a new list."""
    out = [0.0] * len(theta)
    for i, w in enumerate(theta):
        if w > clip:
            out[i] = clip
        elif w < -clip:
            out[i] = -clip
        else:
            out[i] = w
    return out


# ---------------------------------------------------------------------------
# Core ES loop
# ---------------------------------------------------------------------------


def run_es(
    profile: AIProfile,
    *,
    mode_key: str,
    population_size: int,
    generations: int,
    games_per_chart: int,
    sigma: float,
    lr: float,
    seed: int,
    num_players: int,
    workers: int,
    theta_clip: float = 5.0,
    resume: dict[str, Any] | None = None,
    quiet: bool = False,
) -> ESResult:
    """Tune ``profile`` via ES. See module docstring for the algorithm.

    Parameters mirror the GA's :func:`scripts.evolve.ga.run_ga` where
    possible so invocations are interchangeable in muscle memory:

    * ``population_size`` must be even (mirrored pairs). An odd value
      raises :class:`ValueError`.
    * ``sigma`` is the ES noise std — *not* the GA's mutation sigma.
      Surfaced separately because the two concepts are different
      (GA: per-gene child kick; ES: search radius of the gaussian
      policy).
    * ``lr`` is the outer-loop Adam learning rate.
    * ``theta_clip`` caps ``θ`` component-wise after each update, same
      magnitude as ``profile.mutation_clip`` for consistency with the
      GA's own clip.
    * ``resume`` — if given, a dict from a previous
      :func:`save_resume_state` file. ``θ``, Adam state, RNG state,
      and generation counter are restored; the loop runs
      ``generations`` *additional* generations on top of the resumed
      state.
    * ``quiet`` — suppress the progress bar and informational prints.
      The tests pass ``quiet=True`` so unittest output stays clean.
    """
    if population_size % 2 != 0 or population_size < 2:
        raise ValueError(
            f"population_size must be a positive even integer, "
            f"got {population_size}"
        )
    if generations < 1:
        raise ValueError(f"generations must be ≥ 1, got {generations}")
    if games_per_chart < 1:
        raise ValueError(
            f"games_per_chart must be ≥ 1, got {games_per_chart}"
        )
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    rng = random.Random(seed)
    adam = Adam(lr=lr)
    adam.init(profile.num_weights)

    # Seed θ₀ from saved_best_weights/ via the same lookup chain the GA
    # uses. Falls back to profile.flatten_defaults() if no file exists.
    if resume is not None:
        theta: list[float] = list(resume["theta"])
        if len(theta) != profile.num_weights:
            raise ValueError(
                f"resume theta has {len(theta)} weights, "
                f"profile expects {profile.num_weights}"
            )
        adam = Adam.from_state(resume["adam"])
        adam.init(profile.num_weights)  # no-op if lengths match
        rng.setstate(tuple(_deserialize_rng_state(resume["rng"])))
        start_gen = int(resume.get("generations_completed", 0))
        if not quiet:
            print(
                f"resume: continuing from generation {start_gen} "
                f"with theta from {resume.get('source', '<unknown>')}"
            )
    else:
        seed_loaded = load_profile_weights(profile, num_players)
        theta = list(seed_loaded.weights)
        if len(theta) != profile.num_weights:
            raise ValueError(
                f"loaded theta has {len(theta)} weights, "
                f"profile expects {profile.num_weights}"
            )
        start_gen = 0
        if not quiet:
            print(
                f"seed θ₀: {profile.label} from {seed_loaded.source}"
            )

    theta = _clip(theta, theta_clip)

    # Build the provider slate once (same as GA's initial_providers).
    # For modes that don't depend on the population (everything except
    # self_play), this is all we ever need. For self_play we rebuild
    # per-generation below.
    n_games_per_eval = len(CHARTS) * games_per_chart
    dummy_population: list[list[float]] = [theta]  # only used by self_play
    providers = build_mode_providers(
        mode_key,
        profile,
        num_players=num_players,
        population=dummy_population,
        sample_rng=rng,
        n_games=n_games_per_eval,
        quiet=quiet,
    )
    if not quiet and mode_key == "vs_all":
        names = ", ".join(name for name, _ in providers)
        print(
            f"vs_all: averaging reward across {len(providers)} "
            f"opponent types ({names})"
        )

    result = ESResult(theta=list(theta))

    n_pairs = population_size // 2
    ga_start = time.perf_counter()

    # Constant seed slate across every generation — see module
    # docstring. ES wants the gradient estimator to compare
    # perturbations on the *same* games, not on a fresh seed slice
    # each generation, otherwise "θ improved" gets confounded with
    # "games got easier". These two offsets therefore do NOT depend
    # on ``gen``.
    train_offset = (seed + 1) * 9973
    holdout_offset = (seed + 1000) * 9973

    # --- Sanity evaluation of θ₀ --------------------------------------
    # Evaluate the starting parameters on the *same* held-out offset
    # the per-generation loop uses below, so ``initial_mean_reward``
    # is directly comparable to ``mean_reward_per_gen[k]``. This is
    # what lets the best-theta tracker below seed from θ₀ and only
    # upgrade when a post-step θ genuinely beats the starting point.
    init_reward, init_win_rate = evaluate_theta(
        profile,
        theta,
        providers=providers,
        games_per_chart=games_per_chart,
        seed_offset=holdout_offset,
        workers=workers,
    )
    result.initial_mean_reward = init_reward
    result.initial_win_rate = init_win_rate

    # Seed the best-theta tracker from θ₀ itself. If every subsequent
    # generation is worse (common when θ₀ is already near a local
    # optimum — e.g. a GA champion being fine-tuned), ``best_theta``
    # stays at θ₀ and the saved weights file is the starting point
    # rather than a degraded variant. ``best_generation = -1`` is
    # the sentinel meaning "initial θ₀, no improvement".
    result.best_mean_reward = init_reward
    result.best_win_rate = init_win_rate
    result.best_theta = list(theta)
    result.best_generation = -1

    if not quiet:
        print(
            f"θ₀ held-out eval: reward={init_reward:+.3f} "
            f"win_rate={init_win_rate:.2f}"
        )

    for gen_offset in range(generations):
        gen = start_gen + gen_offset

        # Rebuild providers each generation for self_play (population
        # changes). For the other modes this is cheap and symmetric.
        if mode_key == "self_play":
            # Self-play opponents come from a single-point "population"
            # around θ itself. The cleanest rig: sample small
            # perturbations of the current θ to seed the opponent
            # slate for this generation. That keeps self-play meaningful
            # for a gradient-based trainer, even though it differs from
            # the GA's population-based self_play.
            synthetic_pop = [
                _clip(
                    [
                        t + rng.gauss(0.0, sigma)
                        for t in theta
                    ],
                    theta_clip,
                )
                for _ in range(max(4, population_size // 4))
            ]
            providers = build_mode_providers(
                mode_key,
                profile,
                num_players=num_players,
                population=synthetic_pop,
                sample_rng=rng,
                n_games=n_games_per_eval,
                quiet=True,
            )

        # --- Sample epsilons and build perturbations -----------------
        epsilons = sample_epsilons(n_pairs, profile.num_weights, rng)
        perturbed_thetas = [
            _clip([theta[k] + sigma * eps[k] for k in range(len(theta))], theta_clip)
            for eps in epsilons
        ]

        # --- Evaluate all perturbations ------------------------------
        rewards = evaluate_perturbations(
            profile,
            perturbed_thetas,
            providers=providers,
            games_per_chart=games_per_chart,
            seed_offset=train_offset,
            workers=workers,
        )
        training_mean = sum(rewards) / len(rewards)
        result.training_mean_per_gen.append(training_mean)

        # --- Rank-shape the rewards ----------------------------------
        ranked = rank_transform(rewards)

        # --- Gradient estimate via the ε form ------------------------
        # g[k] = (1 / (N·σ)) · Σᵢ Rᵢ · εᵢ[k]
        # We use the *stored* ε, not the reconstructed (pert - θ)/σ,
        # because the stored ε is pre-clip. After clipping some pert
        # components have zero sensitivity to ε, but the mathematical
        # gradient of the uncut objective still uses the raw ε — that
        # is what the ES literature uses, and the trainer handles
        # boundary components via the re-clip step 7.
        grad = [0.0] * profile.num_weights
        scale = 1.0 / (population_size * sigma)
        for i, eps in enumerate(epsilons):
            r = ranked[i]
            for k in range(profile.num_weights):
                grad[k] += r * eps[k]
        for k in range(profile.num_weights):
            grad[k] *= scale

        # --- Adam step and re-clip -----------------------------------
        theta = adam.step(theta, grad)
        theta = _clip(theta, theta_clip)

        # --- θ-eval on held-out seeds --------------------------------
        heldout_reward, heldout_win_rate = evaluate_theta(
            profile,
            theta,
            providers=providers,
            games_per_chart=games_per_chart,
            seed_offset=holdout_offset,
            workers=workers,
        )
        result.mean_reward_per_gen.append(heldout_reward)
        result.win_rate_per_gen.append(heldout_win_rate)

        if heldout_reward > result.best_mean_reward:
            result.best_mean_reward = heldout_reward
            result.best_generation = gen
            result.best_theta = list(theta)
            result.best_win_rate = heldout_win_rate

        if not quiet:
            _render_progress(
                gen_offset,
                generations,
                heldout_reward,
                heldout_win_rate,
                time.perf_counter() - ga_start,
            )

        result.generations_completed = gen + 1

    if not quiet:
        sys.stdout.write("\n")
        sys.stdout.flush()

    result.theta = list(theta)
    # best_theta / best_mean_reward were seeded from θ₀ above and the
    # per-generation loop only upgrades them on a strict improvement,
    # so if every generation was worse than θ₀ the saved best is
    # genuinely θ₀ and best_generation is -1. The old "fall back to
    # final theta if best is empty" branch is gone — it was papering
    # over the empty-initial-best bug that is now fixed.
    result.adam_state = adam.state_dict()
    result.rng_state = _serialize_rng_state(rng.getstate())
    return result


# ---------------------------------------------------------------------------
# Output / resume
# ---------------------------------------------------------------------------


def save_history_plot(
    result: ESResult,
    path: Path,
    *,
    profile: AIProfile,
    num_players: int,
    mode_key: str,
) -> None:
    """Write the training-curve PNG to ``path``.

    Two lines on one axis: the held-out mean margin reward (the ES
    objective) and the held-out binary win rate (GA-comparable). The
    two use different y-scales conceptually — margin is in ``[-1, 1]``,
    win rate is in ``[0, 1]`` — so we use a secondary axis for the
    win rate to keep both curves legible.

    Lazy-imports matplotlib so the rest of the trainer is importable
    without it (see module docstring).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    gens = list(range(len(result.mean_reward_per_gen)))
    ax.plot(
        gens,
        result.mean_reward_per_gen,
        label="held-out reward (margin)",
        linewidth=2.0,
        color="C0",
    )
    ax.plot(
        gens,
        result.training_mean_per_gen,
        label="train mean (noisy)",
        linewidth=1.0,
        color="C2",
        alpha=0.6,
    )
    ax.axhline(
        result.initial_mean_reward,
        color="grey",
        linestyle="--",
        linewidth=1.0,
        label=f"θ₀ reward ({result.initial_mean_reward:+.3f})",
    )
    ax.set_xlabel("generation")
    ax.set_ylabel("score margin (normalized)")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(
        gens,
        result.win_rate_per_gen,
        label="held-out win rate",
        linewidth=1.5,
        color="C1",
        linestyle="-",
    )
    ax2.set_ylabel(f"win rate vs {num_players - 1}× opponents ({mode_key})")
    ax2.set_ylim(0.0, 1.0)

    # Combine legends from both axes.
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower right")

    ax.set_title(
        f"{profile.label} ES training "
        f"({num_players}-player, opponent={mode_key})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_best_weights(
    result: ESResult,
    path: Path,
    rl_config: dict[str, Any],
) -> None:
    """Write a JSON weights file compatible with the GA output shape.

    Same top-level keys as :func:`scripts.evolve.ga.save_best_weights`
    (``fitness``, ``generation``, ``weights``) plus ``rl_config``
    instead of ``ga_config`` so the CLI / heatmap can read both files
    via the same code path once the winning file is promoted to
    ``saved_best_weights/``.

    ``fitness`` is the best held-out **win rate** (binary,
    GA-comparable), not the margin — the GA stores a win rate there,
    so writing a margin would make the field incomparable across
    trainer types. The RL-specific margin is preserved separately in
    ``rl_config["best_margin_reward"]`` for debugging.
    """
    payload = {
        "fitness": result.best_win_rate,
        "generation": result.best_generation,
        "weights": result.best_theta,
        "rl_config": {
            **rl_config,
            "initial_mean_reward": result.initial_mean_reward,
            "initial_win_rate": result.initial_win_rate,
            "best_margin_reward": result.best_mean_reward,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def save_resume_state(
    result: ESResult,
    path: Path,
    *,
    profile: AIProfile,
    mode_key: str,
    num_players: int,
    rl_config: dict[str, Any],
) -> None:
    """Write the full state needed for ``--resume``.

    Separate from the weights file so the weights file stays a clean
    drop-in replacement for GA artifacts — the resume file carries
    Adam moments, RNG state, and generation counter which would be
    noise in a weights file consumed by the CLI.
    """
    payload = {
        "theta": result.theta,
        "adam": result.adam_state,
        "rng": result.rng_state,
        "generations_completed": result.generations_completed,
        "profile_key": profile.key,
        "mode_key": mode_key,
        "num_players": num_players,
        "rl_config": rl_config,
        "source": str(path),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_resume_state(path: Path) -> dict[str, Any]:
    """Read a file produced by :func:`save_resume_state`.

    Thin wrapper around ``json.loads`` that also tags the payload with
    its source path for the trainer's log line.
    """
    data = json.loads(path.read_text())
    data["source"] = str(path)
    return data


# ---------------------------------------------------------------------------
# RNG state serialization
# ---------------------------------------------------------------------------
#
# ``random.Random().getstate()`` returns a tuple containing a large
# tuple of Mersenne-Twister ints. JSON can't serialize tuples — they
# become lists, which we have to convert back on load. These two
# helpers own that conversion so the rest of the trainer can pretend
# the state is JSON-native.


def _serialize_rng_state(state: Any) -> Any:
    """Convert an ``rng.getstate()`` tuple into a JSON-safe structure."""
    if isinstance(state, tuple):
        return ["__tuple__", [_serialize_rng_state(x) for x in state]]
    return state


def _deserialize_rng_state(state: Any) -> Any:
    """Inverse of :func:`_serialize_rng_state`."""
    if isinstance(state, list) and len(state) == 2 and state[0] == "__tuple__":
        return tuple(_deserialize_rng_state(x) for x in state[1])
    return state
