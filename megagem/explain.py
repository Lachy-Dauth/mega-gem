"""AI rationale recording + pretty-printing for the CLI debug mode.

`ExplainingPlayer` wraps any `Player` so each `choose_bid` call records
*why* the underlying AI bid what it did — which features it saw, the
reserve it carried, the discount each head produced, the value estimate
on the auction card. The wrapper never changes the bid; it just observes.

`format_rationale` pretty-prints the recorded dict into a human-readable
block that the CLI shows alongside each round's result when `--debug`
is on. The point is to make the evolved AI legible: you can see which
features it's reacting to and which head fired for the current auction.

The rationale builder dispatches on the inner AI's class so the printout
adapts to whatever AI you happen to be playing against — Random shows
just the cap, Heuristic shows the discount and features, EvolvedSplit
shows the per-head discounts, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .cards import AuctionCard, GemCard, InvestCard, LoanCard, TreasureCard
from .engine import max_legal_bid
from .players import (
    AdaptiveHeuristicAI,
    HeuristicAI,
    HyperAdaptiveAI,
    HyperAdaptiveSplitAI,
    Player,
    _compute_discount_features,
    _hyper_compute_discount_features,
    _hyper_treasure_value,
)

if TYPE_CHECKING:
    from .state import GameState, PlayerState


class ExplainingPlayer(Player):
    """Pass-through decorator that records a rationale on each bid call.

    Stores the most recent rationale on ``self.last_rationale`` so the CLI
    can read it after ``play_round`` returns. The wrapper *forwards* every
    decision to the wrapped player unchanged — the rationale is purely
    observational, so it cannot perturb game results.
    """

    def __init__(self, inner: Player) -> None:
        super().__init__(inner.name)
        self.is_human = inner.is_human
        self._inner = inner
        self.last_rationale: dict[str, Any] | None = None

    def choose_bid(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
        auction: AuctionCard,
    ) -> int:
        # Build the rationale *before* delegating so the inner AI sees the
        # exact same state we report on. Catch any errors so a buggy
        # explainer can never break gameplay.
        try:
            self.last_rationale = _build_rationale(
                self._inner, public_state, my_state, auction
            )
        except Exception as exc:  # noqa: BLE001 — best-effort observability
            self.last_rationale = {"error": repr(exc)}
        return self._inner.choose_bid(public_state, my_state, auction)

    def choose_gem_to_reveal(
        self,
        public_state: "GameState",
        my_state: "PlayerState",
    ) -> GemCard:
        return self._inner.choose_gem_to_reveal(public_state, my_state)


# ---------- rationale builders -------------------------------------------


def _auction_kind(auction: AuctionCard) -> str:
    if isinstance(auction, TreasureCard):
        return "treasure"
    if isinstance(auction, LoanCard):
        return "loan"
    if isinstance(auction, InvestCard):
        return "invest"
    return "?"


def _features_dict(features) -> dict[str, float]:
    return {
        "progress": features.progress,
        "my_cash_ratio": features.my_cash_ratio,
        "avg_cash_ratio": features.avg_cash_ratio,
        "top_cash_ratio": features.top_cash_ratio,
        "variance": features.variance,
    }


def _build_rationale(
    ai: Player,
    public_state: "GameState",
    my_state: "PlayerState",
    auction: AuctionCard,
) -> dict[str, Any]:
    cap = max_legal_bid(my_state, auction)
    rat: dict[str, Any] = {
        "ai_class": type(ai).__name__,
        "auction": str(auction),
        "auction_kind": _auction_kind(auction),
        "cap": cap,
        "coins": my_state.coins,
    }

    # HyperAdaptiveSplitAI: per-head discounts. This is the interesting
    # case because it's the GA-evolved one and the heads are what people
    # actually want to inspect.
    if isinstance(ai, HyperAdaptiveSplitAI):
        features = _hyper_compute_discount_features(public_state, my_state)
        reserve = ai._reserve_for_future(public_state)
        rat["features"] = _features_dict(features)
        rat["reserve"] = reserve
        rat["spendable"] = max(0, my_state.coins - reserve)
        rat["treasure_discount"] = ai.treasure_model.discount(features)
        rat["invest_discount"] = ai.invest_model.discount(features)
        rat["loan_discount"] = ai.loan_model.discount(features)
        rat["chosen_head"] = _auction_kind(auction)
        if isinstance(auction, TreasureCard):
            rat["treasure_value_estimate"] = _hyper_treasure_value(
                auction, public_state, my_state
            )
        return rat

    # HyperAdaptiveAI (single-head): show the one shared discount.
    if isinstance(ai, HyperAdaptiveAI):
        features = _hyper_compute_discount_features(public_state, my_state)
        reserve = ai._reserve_for_future(public_state)
        rat["features"] = _features_dict(features)
        rat["reserve"] = reserve
        rat["spendable"] = max(0, my_state.coins - reserve)
        rat["discount"] = ai.discount_rate(features)
        if isinstance(auction, TreasureCard):
            rat["treasure_value_estimate"] = _hyper_treasure_value(
                auction, public_state, my_state
            )
        return rat

    # AdaptiveHeuristicAI: same discount-rate model, classic features.
    if isinstance(ai, AdaptiveHeuristicAI):
        features = _compute_discount_features(public_state, my_state)
        rat["features"] = _features_dict(features)
        rat["discount"] = ai.discount_rate(features)
        return rat

    # Plain HeuristicAI / RandomAI / unknown: just the cap + class name.
    if isinstance(ai, HeuristicAI):
        rat["note"] = "vanilla heuristic — fixed-fraction bid sizing"
    return rat


# ---------- pretty printer ------------------------------------------------


def format_rationale(rat: dict[str, Any] | None, bid: int) -> str:
    """Multi-line, indented block describing one AI's bid decision."""
    if rat is None:
        return f"    bid={bid}  (no rationale recorded)"
    if "error" in rat:
        return f"    bid={bid}  rationale-error: {rat['error']}"

    lines = [
        f"    bid={bid}  cap={rat['cap']}  coins={rat['coins']}  ai={rat['ai_class']}"
    ]

    if "features" in rat:
        f = rat["features"]
        lines.append(
            "      features:  progress={:.2f}  my_cash={:.2f}  avg_cash={:.2f}  top_cash={:.2f}  var={:.2f}".format(
                f["progress"],
                f["my_cash_ratio"],
                f["avg_cash_ratio"],
                f["top_cash_ratio"],
                f["variance"],
            )
        )

    if "reserve" in rat:
        lines.append(
            "      reserve={}  spendable={}".format(
                rat["reserve"], rat.get("spendable", "–")
            )
        )

    chosen = rat.get("chosen_head")
    if chosen is not None:
        marker_for = {
            "treasure": ("◀", "  ", "  "),
            "invest":   ("  ", "◀", "  "),
            "loan":     ("  ", "  ", "◀"),
        }.get(chosen, ("  ", "  ", "  "))
        lines.append(
            "      heads:  treasure={:.2f}{}  invest={:.2f}{}  loan={:.2f}{}".format(
                rat["treasure_discount"], marker_for[0],
                rat["invest_discount"],   marker_for[1],
                rat["loan_discount"],     marker_for[2],
            )
        )
    elif "discount" in rat:
        lines.append("      discount={:.2f}".format(rat["discount"]))

    if "treasure_value_estimate" in rat:
        lines.append(
            "      treasure_value_estimate={}".format(rat["treasure_value_estimate"])
        )

    if "note" in rat:
        lines.append("      note: {}".format(rat["note"]))

    return "\n".join(lines)


def render_round_rationales(
    state: "GameState", players: list[Player], bids: list[int]
) -> str:
    """Render the per-player rationale block for one round.

    Iterates the seating order, skipping non-`ExplainingPlayer` seats
    (humans, anything that wasn't wrapped). Returns "" if nothing has
    a rationale to show, so the caller can decide whether to print at all.
    """
    blocks = []
    for player, ps, bid in zip(players, state.player_states, bids):
        if not isinstance(player, ExplainingPlayer):
            continue
        rat = player.last_rationale
        blocks.append(f"  {ps.name}:")
        blocks.append(format_rationale(rat, bid))
    if not blocks:
        return ""
    return "AI rationale:\n" + "\n".join(blocks)
