"""Shared base class for the evolved AIs.

:class:`BaseEvoAI` hoists the scaffolding that ``Evo2AI``, ``Evo3AI``,
and ``Evo4AI`` previously re-implemented independently:

* ``__init__`` that stores the player name, seeds a private
  ``random.Random`` (currently unused but kept for subclass flexibility),
  and initializes the Evo3/Evo4 opponent-history fields
  (``_opp_history`` / ``_last_default_bid``). The history fields are
  created for every evolved AI — Evo2 never reads them, but the cost
  is one empty list and one ``None`` per instance, which is the
  cheapest way to keep Evo3/Evo4's ``observe_round`` free of init
  branching.
* ``choose_gem_to_reveal`` — the chart-delta-× relative-holdings reveal
  policy originally copied from ``HeuristicAI.choose_gem_to_reveal``
  and duplicated verbatim in all three Evo modules.

Per-generation feature math (``_treasure_value_stats``,
``_weighted_delta_stats``, ``_predict_opponent_treasure_bids``, the
per-head linear models, ``choose_bid``, ``explain_bid``,
``from_weights``, ``flatten_defaults``) still lives in each
``evo{N}.py`` because the feature vectors and weight layouts differ
per generation. This base class only owns what is truly identical.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING

from ..cards import GemCard
from ..value_charts import value_for
from .base import Player

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


class BaseEvoAI(Player):
    """Common base for the evolved AI chain (Evo2 → Evo3 → Evo4).

    Subclasses must still implement :meth:`choose_bid`; the
    :class:`Player` ABC declares it abstract and this base class does
    not try to guess a generic dispatch. Reveal policy
    (:meth:`choose_gem_to_reveal`) and the opponent-history scratch
    state are provided here.
    """

    def __init__(self, name: str, *, seed: int | None = None) -> None:
        super().__init__(name)
        # Private RNG — reserved for subclass use; none of the current
        # Evo AIs consume it, but keeping it here means a future evolved
        # AI can reach for ``self._rng`` without re-adding the import.
        self._rng = random.Random(seed)
        # Opp-delta history used by ``Evo3AI.observe_round`` and
        # ``Evo4AI.observe_round``. Evo2 never touches either field,
        # which costs one empty list + one ``None`` per instance — a
        # trivial price for keeping the subclass ``__init__`` bodies
        # free of repeated bookkeeping.
        self._opp_history: list[tuple[str, float]] = []
        # Scratch cache populated by ``choose_bid`` in Evo3/Evo4 with
        # the "baseline" bid (what the AI would bid with default
        # opponent-delta inputs). ``observe_round`` reads and clears it
        # so a stale value can't leak across rounds.
        self._last_default_bid: int | None = None

    # ------------------------------------------------------------------
    # Reveal policy — direct lift of the formerly-duplicated copies in
    # Evo2AI / Evo3AI / Evo4AI, which in turn copied HeuristicAI.
    # Canonical home for the logic is now here.
    # ------------------------------------------------------------------
    def choose_gem_to_reveal(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
    ) -> GemCard:
        chart = public_state.value_chart
        display = public_state.value_display

        my_holding = my_state.collection_gems
        opp_holding: Counter = Counter()
        for ps in public_state.player_states:
            if ps is my_state:
                continue
            opp_holding.update(ps.collection_gems)

        best_score: tuple[int, int] | None = None
        best_card: GemCard | None = None
        for card in my_state.hand:
            color = card.color
            current = display.get(color, 0)
            delta = value_for(chart, current + 1) - value_for(chart, current)
            relative = my_holding.get(color, 0) - opp_holding.get(color, 0)
            net_benefit = delta * relative
            tiebreaker = -my_holding.get(color, 0)
            score = (net_benefit, tiebreaker)
            if best_score is None or score > best_score:
                best_score = score
                best_card = card

        return best_card if best_card is not None else my_state.hand[0]
