"""Run repeated cycles of evo1 → evo2 → evo3 → evo4 GA tuning.

Each step invokes ``python -m scripts.evolve --ai evoK [args...]`` as a
subprocess, immediately promotes the resulting weights file from
``artifacts/`` into ``saved_best_weights/`` (so the next step sees the
fresh champion in its ``vs_all`` opponent mix), and then measures every
AI's pooled ``vs_all`` win rate so we can graph progression across the
whole sequence.

Unknown arguments are forwarded verbatim to every underlying
``scripts.evolve`` call, so you can tweak things like ``--generations``,
``--population``, or ``--games-per-chart`` once and have them applied
to all four profiles::

    # 3 full cycles (12 evolve steps), default vs_all / 4p
    python -m scripts.cycle_evolve --cycles 3

    # 2 cycles, 50 gens each, smaller population
    python -m scripts.cycle_evolve --cycles 2 --generations 50 --population 16

The cycle script owns ``--ai`` (set per step) and also wraps
``--opponent`` / ``--num-players`` so it can compute the artifact
filename to promote. Everything else is passthrough.

Outputs (under ``--output-dir``, default ``artifacts/``):

* ``cycle_evolve_history_{tag}_{N}p.json`` — raw per-step win rates
* ``cycle_evolve_history_{tag}_{N}p.png``  — four-line progression plot
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

# Headless backend so this runs over SSH / inside CI without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.evolve.ga import evaluate_against_multi
from scripts.evolve.opponents import (
    MODE_FILENAME_TAGS,
    MODE_KEYS,
    build_mode_providers,
    load_profile_weights,
)
from scripts.evolve.profiles import AI_PROFILES


# The cycle order is semantic: weakest → strongest, matching the zoo
# hierarchy in CLAUDE.md. Explicit rather than reading AI_PROFILES so
# adding an evo5 requires an explicit decision about where it slots in.
ORDER: tuple[str, ...] = ("evo1", "evo2", "evo3", "evo4")

_ARTIFACTS_DIR = Path("artifacts")
_WEIGHTS_DIR = Path("saved_best_weights")


def parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="scripts.cycle_evolve",
        description=(
            "Repeatedly run evo1 → evo2 → evo3 → evo4 GA tuning, promoting "
            "each winner into saved_best_weights/ between steps and "
            "measuring all four AIs' vs_all win rates so the run can be "
            "graphed. Unknown arguments are forwarded to every "
            "scripts.evolve subprocess unchanged."
        ),
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="How many full evo1→evo2→evo3→evo4 sequences to run.",
    )
    parser.add_argument(
        "--opponent",
        choices=MODE_KEYS,
        default="vs_all",
        help=(
            "Opponent mode forwarded to every evolve call. Controls the "
            "artifact filename cycle_evolve promotes after each step."
        ),
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=4,
        choices=(3, 4, 5),
        help="Seats per fitness/eval game (forwarded to every evolve call).",
    )
    parser.add_argument(
        "--eval-games-per-chart",
        type=int,
        default=40,
        help=(
            "Games per chart for the post-step vs_all win-rate measurement "
            "of each AI. 40 × 5 charts × 4 providers = 800 games per AI "
            "per step, per AI measured."
        ),
    )
    parser.add_argument(
        "--eval-seed-offset",
        type=int,
        default=99_000_000,
        help=(
            "Seed offset used for every post-step win-rate measurement "
            "(baseline + every cycle step). Every progression-graph "
            "point replays the *exact same* fixture of games, so any "
            "movement in the plot reflects a real change in AI strength "
            "rather than sampling noise across different seed draws. "
            "Kept well above the GA's training + held-out seed range "
            "so the graph reflects generalisation rather than "
            "memorisation."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help=(
            "Number of threads used for fitness evaluation, both for "
            "this script's post-step win-rate measurements and forwarded "
            "to every scripts.evolve subprocess. Defaults to all CPU "
            "cores. Pass --workers 1 for the fully-sequential path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ARTIFACTS_DIR,
        help="Directory for the cycle history JSON and progression plot.",
    )
    args, passthrough = parser.parse_known_args(argv)

    # --ai is controlled per-step by cycle_evolve; forwarding a user value
    # would silently collide with our injected --ai on each subprocess.
    if any(a == "--ai" or a.startswith("--ai=") for a in passthrough):
        parser.error(
            "--ai is controlled by cycle_evolve and must not be passed explicitly"
        )
    # --workers is owned by cycle_evolve and forwarded to every subprocess
    # below, so a user-supplied passthrough --workers would collide.
    if any(a == "--workers" or a.startswith("--workers=") for a in passthrough):
        parser.error(
            "--workers is controlled by cycle_evolve and must not be passed explicitly"
        )

    return args, passthrough


def run_evolve_step(
    ai_key: str,
    passthrough: list[str],
    *,
    opponent: str,
    num_players: int,
    workers: int,
) -> None:
    """Invoke ``python -m scripts.evolve`` for one AI. Raises on failure."""
    cmd = [
        sys.executable,
        "-m",
        "scripts.evolve",
        "--ai", ai_key,
        "--opponent", opponent,
        "--num-players", str(num_players),
        "--workers", str(workers),
        *passthrough,
    ]
    print(f"\n=== running: {' '.join(cmd)} ===", flush=True)
    subprocess.run(cmd, check=True)


def promote_weights(ai_key: str, *, opponent: str, num_players: int) -> Path:
    """Copy the evolve output from artifacts/ into saved_best_weights/."""
    tag = MODE_FILENAME_TAGS[opponent]
    filename = f"best_weights_{ai_key}_{tag}_{num_players}p.json"
    src = _ARTIFACTS_DIR / filename
    if not src.exists():
        raise SystemExit(
            f"{src} missing — scripts.evolve did not produce expected weights file"
        )
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = _WEIGHTS_DIR / filename
    shutil.copyfile(src, dst)
    return dst


def evaluate_all_ais(
    *,
    num_players: int,
    games_per_chart: int,
    seed_offset: int,
    workers: int,
) -> dict[str, float]:
    """Load each AI's current saved weights and measure vs_all win rate.

    Providers are rebuilt fresh for every AI so that opponent weights
    promoted earlier in the same cycle are reflected. Uses vs_all (the
    canonical "general strength" measure) regardless of the opponent
    mode the GA itself is training against, so the graph always shows
    comparable pooled win rates.
    """
    results: dict[str, float] = {}
    for key in ORDER:
        profile = AI_PROFILES[key]
        loaded = load_profile_weights(profile, num_players)
        providers = build_mode_providers(
            "vs_all",
            profile,
            num_players=num_players,
            quiet=True,
        )
        pooled, _ = evaluate_against_multi(
            profile,
            list(loaded.weights),
            providers=providers,
            games_per_chart=games_per_chart,
            seed_offset=seed_offset,
            workers=workers,
        )
        results[key] = pooled
    return results


def print_winrates(label: str, winrates: dict[str, float]) -> None:
    summary = "  ".join(
        f"{key}={rate * 100:5.1f}%" for key, rate in winrates.items()
    )
    print(f"  {label}: {summary}")


def save_history(
    history: list[dict],
    *,
    cycles: int,
    opponent: str,
    num_players: int,
    eval_games_per_chart: int,
    path: Path,
) -> None:
    payload = {
        "cycles": cycles,
        "opponent": opponent,
        "num_players": num_players,
        "eval_games_per_chart": eval_games_per_chart,
        "history": history,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def save_history_plot(
    history: list[dict],
    path: Path,
    *,
    num_players: int,
    opponent: str,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8.0, 0.7 * len(history) + 3), 5.5))
    xs = list(range(len(history)))
    labels = [entry["label"] for entry in history]
    for key in ORDER:
        ys = [entry["winrates"][key] for entry in history]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=AI_PROFILES[key].label)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(f"vs_all win rate ({num_players}-player)")
    ax.set_xlabel("evolve step")
    ax.set_title(
        f"cycle_evolve progression "
        f"({num_players}p, GA opponent={opponent})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tag = MODE_FILENAME_TAGS[args.opponent]
    suffix = f"{tag}_{args.num_players}p"
    history_path = args.output_dir / f"cycle_evolve_history_{suffix}.json"
    plot_path = args.output_dir / f"cycle_evolve_history_{suffix}.png"

    print(
        f"cycle_evolve: {args.cycles} cycle(s) of {' → '.join(ORDER)} "
        f"(opponent={args.opponent}, {args.num_players}p)"
    )
    if passthrough:
        print(f"forwarding to scripts.evolve: {' '.join(passthrough)}")

    history: list[dict] = []

    # --- Baseline measurement (before any GA step runs)
    baseline = evaluate_all_ais(
        num_players=args.num_players,
        games_per_chart=args.eval_games_per_chart,
        seed_offset=args.eval_seed_offset,
        workers=args.workers,
    )
    print("baseline win rates:")
    print_winrates("baseline", baseline)
    history.append({
        "label": "baseline",
        "cycle": 0,
        "ai": None,
        "winrates": baseline,
    })
    save_history(
        history,
        cycles=args.cycles,
        opponent=args.opponent,
        num_players=args.num_players,
        eval_games_per_chart=args.eval_games_per_chart,
        path=history_path,
    )
    save_history_plot(
        history,
        plot_path,
        num_players=args.num_players,
        opponent=args.opponent,
    )

    # --- Main cycle loop
    t0 = time.perf_counter()
    for cycle_idx in range(1, args.cycles + 1):
        for ai_key in ORDER:
            run_evolve_step(
                ai_key,
                passthrough,
                opponent=args.opponent,
                num_players=args.num_players,
                workers=args.workers,
            )
            promoted = promote_weights(
                ai_key,
                opponent=args.opponent,
                num_players=args.num_players,
            )
            print(f"promoted → {promoted}")

            # Every progression-graph point — baseline and every
            # cycle step — uses the *same* eval_seed_offset, so every
            # measurement replays the identical fixture of games.
            # Within that fixture every provider still gets its own
            # disjoint slice of seed space (the +101 per-provider
            # offset inside evaluate_against_multi) so challengers
            # can't hit the same favourable seed against every
            # opponent. With the fixture held constant across
            # measurements, a win-rate change on the plot now
            # reflects an actual strength delta in the AI whose
            # weights were just promoted, instead of being tangled up
            # with a fresh sampling draw.
            winrates = evaluate_all_ais(
                num_players=args.num_players,
                games_per_chart=args.eval_games_per_chart,
                seed_offset=args.eval_seed_offset,
                workers=args.workers,
            )
            step_label = f"{ai_key} c{cycle_idx}"
            print(f"after {step_label}:")
            print_winrates(step_label, winrates)

            history.append({
                "label": step_label,
                "cycle": cycle_idx,
                "ai": ai_key,
                "winrates": winrates,
            })

            # Persist progressively so a mid-run Ctrl+C keeps useful output.
            save_history(
                history,
                cycles=args.cycles,
                opponent=args.opponent,
                num_players=args.num_players,
                eval_games_per_chart=args.eval_games_per_chart,
                path=history_path,
            )
            save_history_plot(
                history,
                plot_path,
                num_players=args.num_players,
                opponent=args.opponent,
            )

    elapsed = time.perf_counter() - t0
    print(
        f"\ncycle_evolve finished {args.cycles} cycle(s) "
        f"({len(history) - 1} evolve steps) in {elapsed:.1f}s"
    )
    print(f"wrote {history_path}")
    print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
