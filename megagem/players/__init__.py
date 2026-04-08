"""Player implementations for MegaGem.

The :class:`Player` ABC is the modular AI seam: subclass it to plug in a
smarter strategy. The engine clamps any returned bid to the legal range,
so AIs cannot accidentally produce illegal moves — they will simply be
capped.

Every AI in the zoo lives in its own module inside this package; shared
value-estimation and discount-feature math lives in
:mod:`megagem.players.helpers`. Importing the public names straight from
:mod:`megagem.players` continues to work via the re-exports below.
"""

from .adaptive_heuristic import AdaptiveHeuristicAI
from .base import Player
from .evo2 import Evo2AI
from .evo3 import Evo3AI
from .helpers import (
    _GEMS_PER_COLOR,
    _TOTAL_AUCTIONS,
    _DiscountFeatures,
    _chart_value,
    _compute_discount_features,
    _ev_remaining_auctions,
    _expected_avg_treasure_value,
    _expected_final_display,
    _format_discount_features,
    _hyper_avg_treasure_value,
    _hyper_compute_discount_features,
    _hyper_ev_remaining_auctions,
    _hyper_expected_per_gem_value,
    _hyper_hidden_distribution,
    _hyper_treasure_gem_value,
    _hyper_treasure_value,
    _marginal_gem_value,
    _mission_completion_bonus,
    _mission_progress_bonus,
    _remaining_supply,
    _treasure_value,
)
from .heuristic import HeuristicAI
from .human import HumanPlayer
from .hyper_adaptive import HyperAdaptiveAI
from .hyper_adaptive_split import HyperAdaptiveSplitAI, _BidModel
from .hypergeometric import HypergeometricAI
from .random_ai import RandomAI

__all__ = [
    "AdaptiveHeuristicAI",
    "Evo2AI",
    "Evo3AI",
    "HeuristicAI",
    "HumanPlayer",
    "HyperAdaptiveAI",
    "HyperAdaptiveSplitAI",
    "HypergeometricAI",
    "Player",
    "RandomAI",
    "_BidModel",
]
