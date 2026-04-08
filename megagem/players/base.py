"""The :class:`Player` ABC — the modular AI seam.

Subclass it to plug in a smarter strategy. The engine clamps any returned
bid to the legal range, so AIs cannot accidentally produce illegal moves —
they will simply be capped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..cards import AuctionCard, GemCard

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


class Player(ABC):
    """Base class for any MegaGem player (human or AI)."""

    name: str
    is_human: bool = False

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def choose_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> int:
        ...

    @abstractmethod
    def choose_gem_to_reveal(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
    ) -> GemCard:
        ...

    # ------------------------------------------------------------------
    # Optional debug-mode hook.
    # ------------------------------------------------------------------
    def explain_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> list[str]:
        """Return the *detail* lines for ``--debug`` rationale output.

        Each AI subclass that wants to surface its reasoning in the CLI's
        ``--debug`` mode overrides this. The framework
        (:mod:`megagem.explain`) handles the standard header
        (``bid=X cap=Y coins=Z ai=Class``) and indents whatever lines
        this method returns underneath. Returning an empty list — the
        default — produces a header-only block.

        Implementations must be observational only: never mutate state,
        never call back into ``choose_bid``. The explainer wraps any
        exception thrown here so a buggy override cannot break gameplay.
        """
        return []
