"""AI rationale recording + pretty-printing for the CLI debug mode.

This module is a thin shell. The *content* of every rationale lives on
the AI itself, in :meth:`megagem.players.Player.explain_bid` (default
empty list, overridden by each AI that wants to surface its reasoning).
This file owns just three pieces:

1. :class:`ExplainingPlayer` — a passthrough decorator that calls
   ``inner.explain_bid(...)`` on every ``choose_bid`` and stashes the
   resulting lines so the CLI can display them after the round.
2. :func:`format_rationale` — pretty-prints the standard header (bid,
   cap, coins, ai class) plus the AI-supplied detail lines.
3. :func:`render_round_rationales` — walks the seating order at the end
   of a round and concatenates each wrapped player's block.

To make a new AI legible in ``--debug`` mode, override ``explain_bid``
on the class. There is no per-class dispatch in this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cards import AuctionCard, GemCard
from .engine import max_legal_bid
from .players import Player

if TYPE_CHECKING:
    from .state import GameState, PlayerState


class ExplainingPlayer(Player):
    """Pass-through decorator that records rationale on each bid call.

    The wrapper *forwards* every decision to the wrapped player unchanged
    — observability only, so it cannot perturb game results. After each
    ``choose_bid`` it stores:

    * ``last_lines``: the list of detail strings returned by
      ``inner.explain_bid(...)`` (or an error placeholder).
    * ``last_meta``: a dict carrying the standard-header values (cap,
      coins, ai class name) so the renderer doesn't have to recompute
      them later.
    """

    def __init__(self, inner: Player) -> None:
        super().__init__(inner.name)
        self.is_human = inner.is_human
        self._inner = inner
        self.last_lines: list[str] | None = None
        self.last_meta: dict | None = None

    def choose_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> int:
        # Build the rationale *before* delegating so the inner AI sees the
        # exact same state we report on. Catch anything thrown by a buggy
        # explainer so it can never break gameplay.
        try:
            self.last_meta = {
                "cap": max_legal_bid(my_state, auction),
                "coins": my_state.coins,
                "ai_class": type(self._inner).__name__,
            }
            self.last_lines = self._inner.explain_bid(
                public_state, my_state, auction
            )
        except Exception as exc:  # noqa: BLE001 — best-effort observability
            self.last_lines = [f"rationale-error: {exc!r}"]
        return self._inner.choose_bid(public_state, my_state, auction)

    def choose_gem_to_reveal(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
    ) -> GemCard:
        return self._inner.choose_gem_to_reveal(public_state, my_state)


# ---------- pretty printer ------------------------------------------------


def format_rationale(
    meta: dict | None,
    lines: list[str] | None,
    bid: int,
) -> str:
    """Render one bid's rationale block.

    Layout::

        bid=N  cap=M  coins=K  ai=ClassName
          line one
          line two
          ...

    With ``meta=None`` (no record yet) or ``lines=None`` (an inner error
    crashed before we could record), falls back to a header-only block
    so the printout still tells the user *which* seat is missing data.
    """
    if meta is None:
        return f"    bid={bid}  (no rationale recorded)"
    header = (
        f"    bid={bid}  cap={meta['cap']}  "
        f"coins={meta['coins']}  ai={meta['ai_class']}"
    )
    if not lines:
        return header
    detail = "\n".join(f"      {line}" for line in lines)
    return f"{header}\n{detail}"


def render_round_rationales(
    state: "GameState", players: list[Player], bids: list[int]
) -> str:
    """Render the per-player rationale block for one round.

    Iterates the seating order, skipping non-``ExplainingPlayer`` seats
    (humans, anything that wasn't wrapped). Returns "" if nothing has a
    rationale to show, so the caller can decide whether to print at all.
    """
    blocks: list[str] = []
    for player, ps, bid in zip(players, state.player_states, bids):
        if not isinstance(player, ExplainingPlayer):
            continue
        blocks.append(f"  {ps.name}:")
        blocks.append(format_rationale(player.last_meta, player.last_lines, bid))
    if not blocks:
        return ""
    return "AI rationale:\n" + "\n".join(blocks)
