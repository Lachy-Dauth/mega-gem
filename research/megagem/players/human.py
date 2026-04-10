"""HumanPlayer — CLI-driven interactive player."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..cards import AuctionCard, GemCard, InvestCard, LoanCard, TreasureCard
from ..engine import max_legal_bid
from .base import Player

if TYPE_CHECKING:
    from ..state import GameState, PlayerState


class HumanPlayer(Player):
    """CLI-driven human player. Lazily imports `render` to avoid cycles."""

    is_human = True

    def __init__(self, name: str = "You", debug: bool = False) -> None:
        super().__init__(name)
        self.debug = debug

    def choose_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> int:
        from .. import render

        print(render.render_board(public_state, debug=self.debug))
        print()
        print(f"Auction card: {auction}")
        if isinstance(auction, TreasureCard):
            gems_for_sale = public_state.revealed_gems[: auction.gems]
            label = ", ".join(str(g) for g in gems_for_sale) if gems_for_sale else "(none)"
            print(f"  For sale: {label}")
        elif isinstance(auction, LoanCard):
            print(f"  Win to receive {auction.amount} coins; pay {auction.amount} back at game end.")
        elif isinstance(auction, InvestCard):
            print(f"  Win to lock your bid + receive an extra {auction.amount} at game end.")
        print()
        print(render.render_hand(my_state))

        cap = max_legal_bid(my_state, auction)
        prompt_extra = ""
        if isinstance(auction, LoanCard) and cap > my_state.coins:
            prompt_extra = f" (you may bid up to {cap} since this is a loan)"
        while True:
            raw = input(f"{self.name}, enter your bid 0-{cap}{prompt_extra}: ").strip()
            try:
                bid = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue
            if bid < 0 or bid > cap:
                print(f"Bid must be between 0 and {cap}.")
                continue
            return bid

    def choose_gem_to_reveal(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
    ) -> GemCard:
        from .. import render

        print()
        print("You won — reveal one gem from your hand to the Value Display.")
        print(render.render_hand(my_state))
        while True:
            raw = input(f"Pick a gem 1-{len(my_state.hand)}: ").strip()
            try:
                idx = int(raw) - 1
            except ValueError:
                print("Please enter an integer.")
                continue
            if 0 <= idx < len(my_state.hand):
                return my_state.hand[idx]
            print("Out of range.")
