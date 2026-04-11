"""Per-AI profile registry for the unified GA tuner.

Each :class:`AIProfile` carries the constants that differ between the
four evolvable bots — the AI class itself, genome length, and
mutation sigma/clip. Everything else is uniform across profiles:

* Individual #0 of the GA is loaded from ``saved_best_weights/`` via
  the shared lookup chain in :mod:`scripts.evolve.opponents`.
* ``flatten_defaults`` (the fallback when no saved weights file
  exists) is a classmethod on the AI class itself, not a closure
  living in this module.

The four profile keys are ``evo1`` (HyperAdaptiveSplitAI), ``evo2``,
``evo3``, and ``evo4``. The ``evo1`` label is a convenience — the
class itself is still :class:`HyperAdaptiveSplitAI`; we use the
``evo1`` name inside the GA so file paths and CLI choices line up
symmetrically with the later bots.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    knows the genome length and how to spawn the AI. Weight loading
    (including the fallback-to-class-defaults path) is done by
    :func:`scripts.evolve.opponents.load_profile_weights` using the
    shared lookup chain.
    """

    key: str                                          # "evo1" | "evo2" | "evo3" | "evo4"
    label: str                                        # human-readable, e.g. "HyperAdaptiveSplitAI"
    ai_class: type[Player]
    num_weights: int
    mutation_sigma: float
    mutation_clip: float

    def flatten_defaults(self) -> list[float]:
        """Delegate to the AI class's ``flatten_defaults`` classmethod.

        Each :class:`Player` subclass maintains its own hardcoded
        ``DEFAULT_*`` constants plus a ``flatten_defaults`` classmethod
        that packs them into the flat vector ``from_weights`` consumes.
        Keeping the layout next to the class definition is the only
        sensible spot — adding a new feature to (say) Evo4's treasure
        head has to touch ``from_weights`` *and* ``flatten_defaults``
        together, so they want to live in the same file.
        """
        return self.ai_class.flatten_defaults()


# --- Profile registry -------------------------------------------------------


AI_PROFILES: dict[str, AIProfile] = {
    "evo1": AIProfile(
        key="evo1",
        label="HyperAdaptiveSplitAI",
        ai_class=HyperAdaptiveSplitAI,
        num_weights=HyperAdaptiveSplitAI.NUM_WEIGHTS,
        # Tighter mutations than the later bots — the original
        # hyper_adaptive GA was tuned for these magnitudes.
        mutation_sigma=0.15,
        mutation_clip=2.0,
    ),
    "evo2": AIProfile(
        key="evo2",
        label="Evo2AI",
        ai_class=Evo2AI,
        num_weights=Evo2AI.NUM_WEIGHTS,
        mutation_sigma=0.10,
        mutation_clip=5.0,
    ),
    "evo3": AIProfile(
        key="evo3",
        label="Evo3AI",
        ai_class=Evo3AI,
        num_weights=Evo3AI.NUM_WEIGHTS,
        mutation_sigma=0.10,
        mutation_clip=5.0,
    ),
    "evo4": AIProfile(
        key="evo4",
        label="Evo4AI",
        ai_class=Evo4AI,
        num_weights=Evo4AI.NUM_WEIGHTS,
        mutation_sigma=0.10,
        mutation_clip=5.0,
    ),
}


PROFILE_KEYS = tuple(AI_PROFILES.keys())
