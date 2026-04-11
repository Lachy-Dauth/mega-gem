"""Per-AI profile registry for the unified GA tuner.

Each :class:`AIProfile` carries the constants and helpers that differ
between the four evolvable bots — DEFAULT_SEED genome, mutation sigma,
flatten-defaults closure, weights-file lookup chain, and paste-ready
printer. The GA loop in :mod:`scripts.evolve.ga` is fully parameterised
by an :class:`AIProfile`, so all four AIs share one code path.

The four profile keys are ``evo1`` (HyperAdaptiveSplitAI), ``evo2``,
``evo3``, and ``evo4``. The ``evo1`` label is a convenience — the class
itself is still :class:`HyperAdaptiveSplitAI`; we use the ``evo1`` name
inside the GA so file paths and CLI choices line up symmetrically with
the later bots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from megagem.players import (
    Evo2AI,
    Evo3AI,
    Evo4AI,
    HyperAdaptiveSplitAI,
    Player,
)


# --- AIProfile dataclass ----------------------------------------------------


@dataclass(frozen=True)
class AIProfile:
    """Everything that differs between the four evolvable bots.

    The GA loop is generic over this dataclass — given a profile, it
    knows the genome length, how to spawn the AI, where to load opponent
    weights from, and how to format the final paste-ready dump.
    """

    key: str                                          # "evo1" | "evo2" | "evo3" | "evo4"
    label: str                                        # human-readable, e.g. "HyperAdaptiveSplitAI"
    ai_class: type[Player]
    num_weights: int
    default_seed: list[float]                         # initial individual #0 of the GA
    mutation_sigma: float
    mutation_clip: float
    flatten_defaults: Callable[[], list[float]]      # fallback when no saved weights file exists
    candidate_filenames: Callable[[int], list[str]]   # files to try when loading this AI's weights
    paste_ready: Callable[[list[float]], None]


# --- Flatten-defaults closures ----------------------------------------------
#
# Used by :func:`scripts.evolve.opponents._load_profile_weights` when no
# saved weights file exists for a given profile. Each function reads the
# class-level ``DEFAULT_*`` model constants and lays them out in the
# exact flat-vector form the matching ``from_weights`` classmethod expects.


def _flatten_evo1_defaults() -> list[float]:
    t = HyperAdaptiveSplitAI.DEFAULT_TREASURE
    i = HyperAdaptiveSplitAI.DEFAULT_INVEST
    l = HyperAdaptiveSplitAI.DEFAULT_LOAN
    return [
        t.bias, t.w_progress, t.w_my_cash, t.w_avg_cash, t.w_top_cash, t.w_variance,
        i.bias, i.w_progress, i.w_my_cash, i.w_avg_cash, i.w_top_cash, i.w_variance,
        l.bias, l.w_progress, l.w_my_cash, l.w_avg_cash, l.w_top_cash, l.w_variance,
    ]


def _flatten_evo2_defaults() -> list[float]:
    t = Evo2AI.DEFAULT_TREASURE
    i = Evo2AI.DEFAULT_INVEST
    l = Evo2AI.DEFAULT_LOAN
    return [
        t.bias, t.w_rounds, t.w_my, t.w_avg, t.w_top, t.w_ev, t.w_std,
        i.bias, i.w_rounds, i.w_my, i.w_avg, i.w_top, i.w_amount,
        l.bias, l.w_rounds, l.w_my, l.w_avg, l.w_top, l.w_amount,
    ]


def _flatten_evo3_defaults() -> list[float]:
    t = Evo3AI.DEFAULT_TREASURE
    i = Evo3AI.DEFAULT_INVEST
    l = Evo3AI.DEFAULT_LOAN
    return [
        t.bias, t.w_rounds, t.w_my, t.w_avg, t.w_top,
        t.w_ev, t.w_std, t.w_mean_delta, t.w_std_delta,
        i.bias, i.w_rounds, i.w_my, i.w_avg, i.w_top, i.w_amount,
        i.w_mean_delta, i.w_std_delta,
        l.bias, l.w_rounds, l.w_my, l.w_avg, l.w_top, l.w_amount,
        l.w_mean_delta, l.w_std_delta,
    ]


def _flatten_evo4_defaults() -> list[float]:
    # The Evo4 genome appends w_opp_max / w_opp_avg to the treasure head,
    # a single color_bias_influence scalar, and a 7-weight internal Evo2
    # treasure predictor. The class defaults bake in zeros for the new
    # weights and re-use Evo2's defaults for the internal predictor.
    t = Evo4AI.DEFAULT_TREASURE
    i = Evo4AI.DEFAULT_INVEST
    l = Evo4AI.DEFAULT_LOAN
    color_bias = Evo4AI.DEFAULT_COLOR_BIAS_INFLUENCE
    internal = Evo4AI.DEFAULT_INTERNAL_EVO2_TREASURE
    return [
        t.bias, t.w_rounds, t.w_my, t.w_avg, t.w_top,
        t.w_ev, t.w_std, t.w_mean_delta, t.w_std_delta,
        t.w_opp_max, t.w_opp_avg,
        i.bias, i.w_rounds, i.w_my, i.w_avg, i.w_top, i.w_amount,
        i.w_mean_delta, i.w_std_delta,
        l.bias, l.w_rounds, l.w_my, l.w_avg, l.w_top, l.w_amount,
        l.w_mean_delta, l.w_std_delta,
        color_bias,
        internal.bias, internal.w_rounds, internal.w_my, internal.w_avg,
        internal.w_top, internal.w_ev, internal.w_std,
    ]


# --- Lookup chains ----------------------------------------------------------
#
# Each function returns a list of candidate filenames (relative to the
# saved_best_weights/ directory) for loading that AI's evolved weights.
# The first existing file wins. The chain order matches the per-AI CLI
# factories in :mod:`megagem.__main__` so opponent loading follows the
# same priority as the live CLI.


def _candidates_evo1(num_players: int) -> list[str]:
    return [
        f"best_weights_evo1_vs_all_{num_players}p.json",
        f"best_weights_evo1_vs_heuristic_{num_players}p.json",
        f"best_weights_evo1_self_{num_players}p.json",
        f"best_weights_evo1_{num_players}p.json",
        # Legacy un-prefixed files predating the unified evolve script.
        f"best_weights_{num_players}p.json",
        "best_weights.json",
    ]


def _candidates_evo2(num_players: int) -> list[str]:
    return [
        f"best_weights_evo2_vs_all_{num_players}p.json",
        f"best_weights_evo2_vs_evo1_{num_players}p.json",
        f"best_weights_evo2_vs_evo3_{num_players}p.json",
        f"best_weights_evo2_vs_evo4_{num_players}p.json",
        f"best_weights_evo2_self_{num_players}p.json",
        # Legacy tags from the pre-unified evolve_evo2 script.
        f"best_weights_evo2_vs_old_evo2_{num_players}p.json",
        f"best_weights_evo2_vs_old_{num_players}p.json",
        f"best_weights_evo2_{num_players}p.json",
        "best_weights_evo2.json",
    ]


def _candidates_evo3(num_players: int) -> list[str]:
    return [
        f"best_weights_evo3_vs_all_{num_players}p.json",
        f"best_weights_evo3_vs_evo1_{num_players}p.json",
        f"best_weights_evo3_vs_evo2_{num_players}p.json",
        f"best_weights_evo3_vs_evo4_{num_players}p.json",
        f"best_weights_evo3_self_{num_players}p.json",
        f"best_weights_evo3_{num_players}p.json",
        "best_weights_evo3.json",
    ]


def _candidates_evo4(num_players: int) -> list[str]:
    return [
        f"best_weights_evo4_vs_all_{num_players}p.json",
        f"best_weights_evo4_vs_evo1_{num_players}p.json",
        f"best_weights_evo4_vs_evo2_{num_players}p.json",
        f"best_weights_evo4_vs_evo3_{num_players}p.json",
        f"best_weights_evo4_self_{num_players}p.json",
        f"best_weights_evo4_{num_players}p.json",
        "best_weights_evo4.json",
    ]


# --- Paste-ready printers ---------------------------------------------------
#
# Format the final winning weights as Python source the user can drop
# into the AI class to lift the new defaults. Each printer matches the
# block layout the corresponding ``from_weights`` classmethod consumes.


def _fmt_block(block: list[float]) -> str:
    return ", ".join(f"{w:+.4f}" for w in block)


def _paste_ready_evo1(weights: list[float]) -> None:
    if not weights:
        print("(no weights — GA produced no result)")
        return
    t = weights[0:6]
    i = weights[6:12]
    l = weights[12:18]
    print()
    print("Evolved weights (paste into HyperAdaptiveSplitAI):")
    print(f"    DEFAULT_TREASURE = _BidModel({_fmt_block(t)})")
    print(f"    DEFAULT_INVEST   = _BidModel({_fmt_block(i)})")
    print(f"    DEFAULT_LOAN     = _BidModel({_fmt_block(l)})")


def _paste_ready_evo2(weights: list[float]) -> None:
    if not weights:
        print("(no weights — GA produced no result)")
        return
    t = weights[0:7]
    i = weights[7:13]
    l = weights[13:19]
    print()
    print("Evolved weights (paste into Evo2AI):")
    print(f"    DEFAULT_TREASURE = _TreasureModel({_fmt_block(t)})")
    print(f"    DEFAULT_INVEST   = _InvestModel({_fmt_block(i)})")
    print(f"    DEFAULT_LOAN     = _LoanModel({_fmt_block(l)})")


def _paste_ready_evo3(weights: list[float]) -> None:
    if not weights:
        print("(no weights — GA produced no result)")
        return
    t = weights[0:9]
    i = weights[9:17]
    l = weights[17:25]
    print()
    print("Evolved weights (paste into Evo3AI):")
    print(f"    DEFAULT_TREASURE = _Evo3TreasureModel({_fmt_block(t)})")
    print(f"    DEFAULT_INVEST   = _Evo3InvestModel({_fmt_block(i)})")
    print(f"    DEFAULT_LOAN     = _Evo3LoanModel({_fmt_block(l)})")


def _paste_ready_evo4(weights: list[float]) -> None:
    if not weights:
        print("(no weights — GA produced no result)")
        return
    t = weights[0:11]
    i = weights[11:19]
    l = weights[19:27]
    color_bias = weights[27]
    internal = weights[28:35]
    print()
    print("Evolved weights (paste into Evo4AI):")
    print(f"    DEFAULT_TREASURE = _Evo4TreasureModel({_fmt_block(t)})")
    print(f"    DEFAULT_INVEST   = _Evo3InvestModel({_fmt_block(i)})")
    print(f"    DEFAULT_LOAN     = _Evo3LoanModel({_fmt_block(l)})")
    print(f"    DEFAULT_COLOR_BIAS_INFLUENCE = {color_bias:+.4f}")
    print(f"    DEFAULT_INTERNAL_EVO2_TREASURE = _Evo2TreasureModel({_fmt_block(internal)})")


# --- DEFAULT_SEED constants -------------------------------------------------
#
# Copied verbatim from the four legacy evolve scripts so the GA's
# starting individual #0 is byte-identical to the pre-unification
# behaviour.


DEFAULT_SEED_EVO1 = [
    # treasure: bias + 5 weights
    0.70, 0.25, 0.35, -0.10, -0.15, -0.05,
    # invest: bias + 5 weights
    0.80, 0.10, 0.20, -0.05, -0.05, 0.00,
    # loan: bias + 5 weights
    0.10, 0.05, -0.40, 0.10, 0.10, -0.05,
]


DEFAULT_SEED_EVO2 = [
    # treasure: bias, w_rounds, w_my, w_avg, w_top, w_ev, w_std
    0.9671062444221764,
    -0.0906995616980441,
    0.07804979550128198,
    0.05375147152736104,
    -0.04247465810129918,
    0.32783828473034604,
    -0.011838494331700117,
    # invest: bias, w_rounds, w_my, w_avg, w_top, w_amount
    1.908464547879478,
    0.4300303741599258,
    -0.1201852409204779,
    -0.28421403664160627,
    0.3149361220138405,
    0.07219353469220569,
    # loan: bias, w_rounds, w_my, w_avg, w_top, w_amount
    -0.4139242208454687,
    -0.31190499765072527,
    0.13966251262722051,
    0.12135141558388368,
    -0.0669196243751372,
    0.36349000133503273,
]


DEFAULT_SEED_EVO3 = [
    # treasure: bias, w_rounds, w_my, w_avg, w_top, w_ev, w_std,
    #           w_mean_delta, w_std_delta
    0.9671062444221764,
    -0.0906995616980441,
    0.07804979550128198,
    0.05375147152736104,
    -0.04247465810129918,
    0.32783828473034604,
    -0.011838494331700117,
    0.0,
    0.0,
    # invest: bias, w_rounds, w_my, w_avg, w_top, w_amount,
    #         w_mean_delta, w_std_delta
    1.908464547879478,
    0.4300303741599258,
    -0.1201852409204779,
    -0.28421403664160627,
    0.3149361220138405,
    0.07219353469220569,
    0.0,
    0.0,
    # loan: bias, w_rounds, w_my, w_avg, w_top, w_amount,
    #       w_mean_delta, w_std_delta
    -0.4139242208454687,
    -0.31190499765072527,
    0.13966251262722051,
    0.12135141558388368,
    -0.0669196243751372,
    0.36349000133503273,
    0.0,
    0.0,
]


DEFAULT_SEED_EVO4 = [
    # treasure: bias, w_rounds, w_my, w_avg, w_top, w_ev, w_std,
    #           w_mean_delta, w_std_delta, w_opp_max, w_opp_avg
    0.9671062444221764,
    -0.0906995616980441,
    0.07804979550128198,
    0.05375147152736104,
    -0.04247465810129918,
    0.32783828473034604,
    -0.011838494331700117,
    0.0,
    0.0,
    0.0,
    0.0,
    # invest: bias, w_rounds, w_my, w_avg, w_top, w_amount,
    #         w_mean_delta, w_std_delta
    1.908464547879478,
    0.4300303741599258,
    -0.1201852409204779,
    -0.28421403664160627,
    0.3149361220138405,
    0.07219353469220569,
    0.0,
    0.0,
    # loan: bias, w_rounds, w_my, w_avg, w_top, w_amount,
    #       w_mean_delta, w_std_delta
    -0.4139242208454687,
    -0.31190499765072527,
    0.13966251262722051,
    0.12135141558388368,
    -0.0669196243751372,
    0.36349000133503273,
    0.0,
    0.0,
    # color_bias_influence — zero so Evo4 starts as Evo3.
    0.0,
    # internal_evo2_treasure: seeded from Evo2.DEFAULT_TREASURE.
    0.9671062444221764,
    -0.0906995616980441,
    0.07804979550128198,
    0.05375147152736104,
    -0.04247465810129918,
    0.32783828473034604,
    -0.011838494331700117,
]


# --- Profile registry -------------------------------------------------------


AI_PROFILES: dict[str, AIProfile] = {
    "evo1": AIProfile(
        key="evo1",
        label="HyperAdaptiveSplitAI",
        ai_class=HyperAdaptiveSplitAI,
        num_weights=HyperAdaptiveSplitAI.NUM_WEIGHTS,
        default_seed=DEFAULT_SEED_EVO1,
        # Tighter mutations than the later bots — the original
        # hyper_adaptive GA was tuned for these magnitudes.
        mutation_sigma=0.15,
        mutation_clip=2.0,
        flatten_defaults=_flatten_evo1_defaults,
        candidate_filenames=_candidates_evo1,
        paste_ready=_paste_ready_evo1,
    ),
    "evo2": AIProfile(
        key="evo2",
        label="Evo2AI",
        ai_class=Evo2AI,
        num_weights=Evo2AI.NUM_WEIGHTS,
        default_seed=DEFAULT_SEED_EVO2,
        mutation_sigma=0.05,
        mutation_clip=5.0,
        flatten_defaults=_flatten_evo2_defaults,
        candidate_filenames=_candidates_evo2,
        paste_ready=_paste_ready_evo2,
    ),
    "evo3": AIProfile(
        key="evo3",
        label="Evo3AI",
        ai_class=Evo3AI,
        num_weights=Evo3AI.NUM_WEIGHTS,
        default_seed=DEFAULT_SEED_EVO3,
        mutation_sigma=0.05,
        mutation_clip=5.0,
        flatten_defaults=_flatten_evo3_defaults,
        candidate_filenames=_candidates_evo3,
        paste_ready=_paste_ready_evo3,
    ),
    "evo4": AIProfile(
        key="evo4",
        label="Evo4AI",
        ai_class=Evo4AI,
        num_weights=Evo4AI.NUM_WEIGHTS,
        default_seed=DEFAULT_SEED_EVO4,
        mutation_sigma=0.05,
        mutation_clip=5.0,
        flatten_defaults=_flatten_evo4_defaults,
        candidate_filenames=_candidates_evo4,
        paste_ready=_paste_ready_evo4,
    ),
}


PROFILE_KEYS = tuple(AI_PROFILES.keys())
