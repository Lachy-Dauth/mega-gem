"""SearchAI — determinized Monte Carlo search over bids.

Every earlier bot maps hand-picked features to a bid with a linear head.
SearchAI instead *plays the game forward*:

1. **Sample worlds.** The only hidden information in MegaGem is how the
   unseen gems split between opponents' hands and the gem deck, plus the
   order of the remaining auction cards. Everything else (display,
   collections, coins, missions, the auction-deck *composition*) is
   public. Each world deals the unseen pool at random into hands of the
   right sizes and a deck of the right size, and shuffles the remaining
   auction cards.
2. **Predict the price to win.** In each world every opponent's bid is
   predicted by running a model bot (Evo4 by default) from that seat,
   with the sampled hand. Bids are sealed and simultaneous, so my bid
   can't change theirs: the world collapses to a single *threshold* —
   the cheapest bid that wins, honouring the engine's tie-break.
3. **Roll out both branches.** From the threshold I play the world to
   the end twice — once losing the auction, once winning it at the
   threshold price — with every seat (me included) driven by rollout
   bots. Both branches share the world and RNG seeds, so the comparison
   is paired and low-variance.
4. **Pick the bid.** For a candidate bid ``b``: worlds whose threshold
   is above ``b`` score the lose branch; the rest score the win branch,
   minus the overpayment ``b − threshold``. Scores are a sigmoid of my
   final margin over the best opponent (a smooth win probability), and
   the best average wins.

Gem reveals don't change final values (every hand ends up in the Value
Display), only what opponents learn and when; SearchAI reuses
:class:`HeuristicAI`'s reveal rule.

All randomness comes from ``seed``, so a seeded SearchAI is
reproducible. Like the GA bots, the rollout/opponent model weights are
passed in by the caller (the CLI and the web client load
``saved_best_weights/``); the players package never reads files.
"""

from __future__ import annotations

import copy
import math
import random
from collections import Counter
from typing import TYPE_CHECKING, Callable

from ..cards import AuctionCard, Color, GemCard, TreasureCard
from ..engine import (
    _check_missions,
    _replenish_revealed,
    is_game_over,
    max_legal_bid,
    play_round,
    score_game,
)
from ..state import GameState, PlayerState
from .base import Player
from .evo2 import Evo2AI
from .evo3 import Evo3AI
from .evo4 import Evo4AI
from .heuristic import HeuristicAI
from .hyper_adaptive_split import HyperAdaptiveSplitAI
from .random_ai import RandomAI

if TYPE_CHECKING:
    pass


ModelFactory = Callable[[str, int], Player]

_GEMS_PER_COLOR = 6
_COLORS: tuple[Color, ...] = tuple(Color)
# One shared GemCard per colour: GemCard is a frozen dataclass compared
# by value, so worlds can reuse these instead of allocating.
_GEM: dict[Color, GemCard] = {c: GemCard(c) for c in _COLORS}


def _default_model_factory(weights: list[float] | None) -> ModelFactory:
    if weights is None:
        return lambda name, seed: Evo4AI(name, seed=seed)
    return lambda name, seed: Evo4AI.from_weights(name, weights, seed=seed)


class _Scripted(Player):
    """Seat that plays a fixed first bid, then defers to ``inner``."""

    def __init__(self, inner: Player, first_bid: int, revealer: Player | None) -> None:
        super().__init__(inner.name)
        self._inner = inner
        self._first_bid: int | None = first_bid
        self._revealer: Player | None = revealer

    def choose_bid(self, public_state, my_state, auction) -> int:
        if self._first_bid is not None:
            bid, self._first_bid = self._first_bid, None
            return bid
        return self._inner.choose_bid(public_state, my_state, auction)

    def choose_gem_to_reveal(self, public_state, my_state) -> GemCard:
        if self._revealer is not None:
            revealer, self._revealer = self._revealer, None
            return revealer.choose_gem_to_reveal(public_state, my_state)
        return self._inner.choose_gem_to_reveal(public_state, my_state)

    def observe_round(self, public_state, my_idx, result) -> None:
        self._inner.observe_round(public_state, my_idx, result)


class _Sniper(Player):
    """Rollout policy for my own future turns: ``inner``'s valuation, but
    since every seat in a simulated world is a deterministic model, pay
    only the threshold price whenever ``inner`` would win anyway."""

    def __init__(self, inner: Player, threshold: Callable) -> None:
        super().__init__(inner.name)
        self._inner = inner
        self._threshold = threshold

    def choose_bid(self, public_state, my_state, auction) -> int:
        bid = self._inner.choose_bid(public_state, my_state, auction)
        me = public_state.player_states.index(my_state)
        opp = {
            j: max(0, min(int(p.choose_bid(public_state, ps, auction)), max_legal_bid(ps, auction)))
            for j, (p, ps) in enumerate(zip(public_state.players, public_state.player_states))
            if j != me
        }
        t = self._threshold(public_state, me, opp)
        return t if bid >= t else bid

    def choose_gem_to_reveal(self, public_state, my_state) -> GemCard:
        return self._inner.choose_gem_to_reveal(public_state, my_state)

    def observe_round(self, public_state, my_idx, result) -> None:
        self._inner.observe_round(public_state, my_idx, result)


class _RoundMemo(Player):
    """Caches ``inner``'s bid for the current round.

    The sniper asks every opponent for its bid before the engine does;
    bids are deterministic given the (unchanged) state, so the engine's
    own call can reuse the answer instead of recomputing it.
    """

    def __init__(self, inner: Player) -> None:
        super().__init__(inner.name)
        self._inner = inner
        self._round = -1
        self._bid = 0

    def choose_bid(self, public_state, my_state, auction) -> int:
        if self._round != public_state.round_number:
            self._bid = self._inner.choose_bid(public_state, my_state, auction)
            self._round = public_state.round_number
        return self._bid

    def choose_gem_to_reveal(self, public_state, my_state) -> GemCard:
        return self._inner.choose_gem_to_reveal(public_state, my_state)

    def observe_round(self, public_state, my_idx, result) -> None:
        self._inner.observe_round(public_state, my_idx, result)


def unseen_pool(state: GameState, my_state: PlayerState) -> list[Color]:
    """Colours of every gem I can't see: opponents' hands + the gem deck."""
    seen: Counter = Counter(g.color for g in my_state.hand)
    seen.update(state.value_display)
    seen.update(g.color for g in state.revealed_gems)
    for ps in state.player_states:
        seen.update(ps.collection_gems)
    pool: list[Color] = []
    for c in _COLORS:
        pool.extend([c] * (_GEMS_PER_COLOR - seen.get(c, 0)))
    return pool


class SearchAI(Player):
    """Bid by Monte Carlo rollouts over sampled hidden information."""

    DEFAULT_WORLDS = 24
    # Margin (in coins) that maps to ~73% win probability in the sigmoid.
    DEFAULT_TEMPERATURE = 6.0

    def __init__(
        self,
        name: str,
        *,
        seed: int | None = None,
        worlds: int | None = None,
        temperature: float | None = None,
        model_weights: list[float] | None = None,
        opponent_model: ModelFactory | None = None,
        rollout_model: ModelFactory | None = None,
        infer: bool = False,
        bid_sigma: float = 1.5,
        shadow: bool = False,
        reveal_search: bool = False,
        rollout_snipe: bool = False,
        model_bank: dict[str, ModelFactory] | None = None,
        identify: bool = False,
    ) -> None:
        super().__init__(name)
        self._rng = random.Random(seed)
        # Bookkeeping draws come from their own stream so the search's
        # stream (and therefore explain_bid's replay of it) is unaffected.
        self._aux_rng = random.Random(self._rng.random())
        self.worlds = worlds if worlds is not None else self.DEFAULT_WORLDS
        self.temperature = (
            temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        )
        default = _default_model_factory(model_weights)
        self._opponent_model = opponent_model or default
        self._rollout_model = rollout_model or default
        self._revealer = HeuristicAI(name)
        self.last_search: dict | None = None
        self.infer = infer
        self.bid_sigma = bid_sigma
        # Hand inference bookkeeping (see _record_bid_tables).
        self._bid_records: list[dict] = []
        self._pending_tables: dict[int, dict[tuple, int]] | None = None
        self.shadow = shadow
        self.reveal_search = reveal_search
        self.rollout_snipe = rollout_snipe
        self._shadows: dict[tuple[int, str], Player] = {}
        # Opponent identification: candidate models (first = prior favourite)
        # and each seat's running log-likelihood under every candidate.
        self._bank: dict[str, ModelFactory] = (
            dict(model_bank) if model_bank else {"model": self._opponent_model}
        )
        self.identify = identify and len(self._bank) > 1
        self._type_ll: dict[int, dict[str, float]] = {}
        self._pending_type_preds: dict[int, dict[str, list[int]]] | None = None

    @classmethod
    def standard(
        cls,
        name: str,
        *,
        seed: int | None = None,
        weights: dict[str, list[float] | None] | None = None,
        **overrides,
    ) -> "SearchAI":
        """The configuration the CLI and web client play.

        ``weights`` maps GA profile keys (``evo1``..``evo4``) to weight
        vectors (``None`` → class defaults). The bank holds every bot in
        the zoo so opponents can be identified; Evo4 is both the prior
        favourite and the rollout policy.
        """
        weights = weights or {}

        def evo(cls_, key):
            w = weights.get(key)
            if w is None:
                return lambda n, s: cls_(n, seed=s)
            return lambda n, s: cls_.from_weights(n, w, seed=s)

        bank: dict[str, ModelFactory] = {
            "evo4": evo(Evo4AI, "evo4"),
            "evo3": evo(Evo3AI, "evo3"),
            "evo2": evo(Evo2AI, "evo2"),
            "evolved": evo(HyperAdaptiveSplitAI, "evo1"),
            "heuristic": lambda n, s: HeuristicAI(n, seed=s),
            "random": lambda n, s: RandomAI(n, seed=s),
        }
        options = dict(
            seed=seed,
            model_bank=bank,
            opponent_model=bank["evo4"],
            rollout_model=bank["evo4"],
            identify=True,
            shadow=True,
            infer=True,
            rollout_snipe=True,
        )
        options.update(overrides)
        return cls(name, **options)

    # ------------------------------------------------------------------
    # World construction
    # ------------------------------------------------------------------

    def _deal_world(
        self,
        state: GameState,
        my_idx: int,
        auction: AuctionCard,
        rng: random.Random,
    ) -> tuple[list[list[GemCard]], list[GemCard], list[AuctionCard]]:
        me = state.player_states[my_idx]
        pool = [_GEM[c] for c in unseen_pool(state, me)]
        rng.shuffle(pool)
        hands: list[list[GemCard]] = []
        pos = 0
        for i, ps in enumerate(state.player_states):
            if i == my_idx:
                hands.append(list(me.hand))
                continue
            n = len(ps.hand)
            hands.append(pool[pos:pos + n])
            pos += n
        gem_deck = pool[pos:]
        auctions = list(state.auction_deck)
        rng.shuffle(auctions)
        return hands, gem_deck, auctions

    @staticmethod
    def _build_state(
        state: GameState,
        players: list[Player],
        hands: list[list[GemCard]],
        gem_deck: list[GemCard],
        auctions: list[AuctionCard],
    ) -> GameState:
        """A fresh, fully-owned GameState for one simulated world."""
        pstates = [
            PlayerState(
                name=ps.name,
                is_human=False,
                hand=list(hand),
                coins=ps.coins,
                collection_gems=Counter(ps.collection_gems),
                completed_missions=list(ps.completed_missions),
                loans=list(ps.loans),
                investments=list(ps.investments),
            )
            for ps, hand in zip(state.player_states, hands)
        ]
        return GameState(
            player_states=pstates,
            players=players,
            value_chart=state.value_chart,
            value_display=Counter(state.value_display),
            gem_deck=list(gem_deck),
            revealed_gems=list(state.revealed_gems),
            auction_deck=list(auctions),
            active_missions=list(state.active_missions),
            last_winner_idx=state.last_winner_idx,
            round_number=state.round_number,
        )

    # ------------------------------------------------------------------
    # Hand inference from opponents' bids
    # ------------------------------------------------------------------

    def _record_bid_tables(self, state: GameState, my_idx: int, auction: AuctionCard) -> None:
        """Predict each opponent's bid for every hand they could hold.

        Called before bidding on treasure rounds. ``observe_round`` turns
        the tables into likelihoods once the real bids are public.
        """
        self._pending_tables = None
        if not isinstance(auction, TreasureCard):
            return  # invest/loan bids don't depend on the hand
        pool = Counter(unseen_pool(state, state.player_states[my_idx]))
        # Bots only read other seats' hand *sizes* and the deck *size*, so
        # placeholders stand in for every hidden card.
        hands, deck, auctions = self._placeholder_world(state, my_idx)
        scratch = self._build_state(state, [], hands, deck, auctions)
        tables: dict[int, dict[tuple, int]] = {}
        for j, probe in enumerate(scratch.player_states):
            if j == my_idx or not probe.hand:
                continue
            model = self._seat_model(j)
            cap = max_legal_bid(probe, auction)
            table: dict[tuple, int] = {}
            for counts in _multisets(len(probe.hand), [pool.get(c, 0) for c in _COLORS]):
                probe.hand = [_GEM[c] for c, n in zip(_COLORS, counts) for _ in range(n)]
                raw = model.choose_bid(scratch, probe, auction)
                table[counts] = max(0, min(int(raw), cap))
            tables[j] = table
        self._pending_tables = tables

    @staticmethod
    def _placeholder_world(state: GameState, my_idx: int):
        filler = _GEM[_COLORS[0]]
        hands = [
            list(ps.hand) if i == my_idx else [filler] * len(ps.hand)
            for i, ps in enumerate(state.player_states)
        ]
        return hands, [filler] * len(state.gem_deck), list(state.auction_deck)

    # ------------------------------------------------------------------
    # Shadow opponent models
    # ------------------------------------------------------------------

    def _seat_kind(self, seat: int) -> str:
        """Most likely model for ``seat`` (the bank's first entry until evidence)."""
        default = next(iter(self._bank))
        ll = self._type_ll.get(seat)
        if not self.identify or not ll:
            return default
        return max(self._bank, key=lambda k: (ll.get(k, 0.0), k == default))

    def _seat_model(self, seat: int, kind: str | None = None) -> Player:
        """A fresh copy of the opponent model for ``seat``, history included.

        With ``shadow`` on, each (seat, model) pair has a model bot that
        has watched every real round from that seat, so history-driven
        features (Evo3/Evo4 bid deltas, colour signal) match the real
        opponent's instead of starting from zero every world.
        """
        kind = kind or self._seat_kind(seat)
        if not self.shadow:
            return self._bank[kind](f"o{seat}", 0)
        key = (seat, kind)
        if key not in self._shadows:
            self._shadows[key] = self._bank[kind](f"o{seat}", 0)
        return _clone(self._shadows[key])

    def _shadow_or_new(self, seat: int, kind: str) -> Player:
        return self._shadows.get((seat, kind)) or self._bank[kind](f"o{seat}", 0)

    def _prime_models(self, state: GameState, my_idx: int, auction: AuctionCard) -> None:
        """Per-round bookkeeping before bidding.

        * Identification: predict every seat's bid under every candidate
          model on a few sampled hands; ``observe_round`` scores them.
        * Shadows: let each shadow bid with a guessed hand so its own
          ``observe_round`` gets the baseline it expects.
        """
        kinds = list(self._bank) if self.identify else [self._seat_kind(-1)]
        self._pending_type_preds = None
        if not (self.identify or self.shadow):
            return
        samples = [self._deal_world(state, my_idx, auction, self._aux_rng) for _ in range(4)]
        scratches = [self._build_state(state, [], *smp) for smp in samples]
        preds: dict[int, dict[str, list[int]]] = {}
        for j in range(state.num_players):
            if j == my_idx:
                continue
            preds[j] = {}
            for kind in kinds if self.identify else []:
                if isinstance(self._shadow_or_new(j, kind), RandomAI):
                    # Uniform bids: score analytically, not by sampling.
                    preds[j][kind] = [-1 - max_legal_bid(state.player_states[j], auction)]
                    continue
                out = []
                for sc in scratches:
                    ps = sc.player_states[j]
                    raw = self._seat_model(j, kind).choose_bid(sc, ps, auction)
                    out.append(max(0, min(int(raw), max_legal_bid(ps, auction))))
                preds[j][kind] = out
            if self.shadow:
                for kind in kinds:
                    key = (j, kind)
                    if key not in self._shadows:
                        self._shadows[key] = self._bank[kind](f"o{j}", 0)
                    sc = scratches[0]
                    self._shadows[key].choose_bid(sc, sc.player_states[j], auction)
        if self.identify:
            self._pending_type_preds = preds

    def observe_round(self, public_state, my_idx, result) -> None:
        preds, self._pending_type_preds = self._pending_type_preds, None
        if preds:
            inv = 1.0 / (2.0 * self.bid_sigma ** 2)
            for j, by_kind in preds.items():
                b = result["bids"][j]
                ll = self._type_ll.setdefault(j, {})
                for kind, ps in by_kind.items():
                    if ps[0] < 0:  # uniform model, encoded as -1 - cap
                        lik = 1.0 / -ps[0]
                    else:
                        lik = sum(math.exp(-(b - p) ** 2 * inv) for p in ps) / len(ps)
                    ll[kind] = ll.get(kind, 0.0) + math.log(0.02 + lik)
        for (j, _kind), shadow in self._shadows.items():
            shadow.observe_round(public_state, j, result)
        tables, self._pending_tables = self._pending_tables, None
        # A gem revealed this round was in the winner's hand when they bid,
        # so it belongs to every earlier record of theirs too.
        winner, gem = result["winner_idx"], result.get("revealed_gem")
        if tables:
            for j, table in tables.items():
                self._bid_records.append(
                    {"seat": j, "bid": result["bids"][j], "pred": table, "revealed": []}
                )
        if gem is not None:
            for rec in self._bid_records:
                if rec["seat"] == winner:
                    rec["revealed"].append(gem.color)

    def _world_weight(self, hands: list[list[GemCard]]) -> float:
        w = 1.0
        inv = 1.0 / (2.0 * self.bid_sigma ** 2)
        for rec in self._bid_records:
            counts = Counter(g.color for g in hands[rec["seat"]])
            counts.update(rec["revealed"])
            pred = rec["pred"].get(tuple(counts.get(c, 0) for c in _COLORS))
            if pred is None:
                w *= 0.05
                continue
            d = rec["bid"] - pred
            # Gaussian around the model's bid, plus a floor so a bot that
            # isn't Evo4 (or a human) can't zero out the true world.
            w *= 0.05 + math.exp(-d * d * inv)
        return w

    def _sample_worlds(self, state: GameState, my_idx: int, auction: AuctionCard):
        if not (self.infer and self._bid_records):
            return [self._deal_world(state, my_idx, auction, self._rng) for _ in range(self.worlds)]
        candidates = [
            self._deal_world(state, my_idx, auction, self._rng)
            for _ in range(self.worlds * 8)
        ]
        weights = [self._world_weight(c[0]) for c in candidates]
        return self._rng.choices(candidates, weights=weights, k=self.worlds)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _threshold(self, state: GameState, my_idx: int, opp_bids: dict[int, int]) -> int:
        """Cheapest bid that wins against ``opp_bids`` under the tie rule."""
        top = max(opp_bids.values())
        n = state.num_players
        last = state.last_winner_idx
        if last is None:
            # First auction: ties are a coin flip, so only outbidding is safe.
            return top + 1
        tied = {i for i, b in opp_bids.items() if b == top} | {my_idx}
        for offset in range(1, n + 1):
            idx = (last + offset) % n
            if idx in tied:
                return top if idx == my_idx else top + 1
        return top + 1

    def _rollout(
        self,
        state: GameState,
        my_idx: int,
        hands, gem_deck, auctions,
        auction: AuctionCard,
        my_bid: int,
        seed: int,
        models: list[Player | None],
        opp_bids: dict[int, int],
    ) -> float:
        """Play one branch to the end; return the sigmoid of my final margin.

        Opponents replay exactly the bids the search predicted for this
        round, then continue as (copies of) the same seat models. If the
        rollout re-derived this round's bids from different models, the
        "win at the threshold" branch might not even be a win.
        """
        rng = random.Random(seed)
        players = self._rollout_players(state, my_idx, rng)
        for i, model in enumerate(models):
            if i != my_idx:
                players[i] = _Scripted(_clone(model), opp_bids[i], None)
                if self.rollout_snipe:
                    players[i] = _RoundMemo(players[i])
        players[my_idx] = _Scripted(players[my_idx], my_bid, self._revealer)
        if self.rollout_snipe:
            players[my_idx]._inner = _Sniper(players[my_idx]._inner, self._threshold)
        # Put the current card back on top: play_round pops it again.
        sim = self._build_state(state, players, hands, gem_deck, auctions + [auction])
        sim.round_number -= 1
        return self._play_out(sim, my_idx, rng)

    def _rollout_players(self, state: GameState, my_idx: int, rng: random.Random) -> list[Player]:
        """My seat plays the rollout model; identified opponents play theirs."""
        players: list[Player] = []
        for i in range(state.num_players):
            seed = rng.randrange(2**31)
            if i == my_idx or not self.identify:
                players.append(self._rollout_model(f"p{i}", seed))
            else:
                players.append(self._bank[self._seat_kind(i)](f"p{i}", seed))
        return players

    def _play_out(self, sim: GameState, my_idx: int, rng: random.Random) -> float:
        while not is_game_over(sim):
            play_round(sim, rng=rng)
        scores = [row["total"] for row in score_game(sim)]
        mine = scores[my_idx]
        best_opp = max(s for i, s in enumerate(scores) if i != my_idx)
        return 1.0 / (1.0 + math.exp(-(mine - best_opp) / self.temperature))

    def _search(self, state: GameState, my_state: PlayerState, auction: AuctionCard):
        my_idx = next(i for i, ps in enumerate(state.player_states) if ps is my_state)
        cap = max_legal_bid(my_state, auction)
        samples = []  # (threshold, value_if_lose, value_if_win_at_threshold)
        for hands, gem_deck, auctions in self._sample_worlds(state, my_idx, auction):
            # Predict opponents' bids with model bots seated in this world.
            models = [
                None if i == my_idx else self._seat_model(i)
                for i in range(state.num_players)
            ]
            world = self._build_state(state, models, hands, gem_deck, auctions)
            opp_bids = {}
            for i, ps in enumerate(world.player_states):
                if i == my_idx:
                    continue
                raw = models[i].choose_bid(world, ps, auction)
                opp_bids[i] = max(0, min(int(raw), max_legal_bid(ps, auction)))
            t = self._threshold(state, my_idx, opp_bids)
            seed = self._rng.randrange(2**31)
            branch = (state, my_idx, hands, gem_deck, auctions, auction)
            lose = self._rollout(*branch, 0, seed, models, opp_bids) if t > 0 else None
            win = self._rollout(*branch, t, seed, models, opp_bids) if t <= cap else None
            samples.append((t, lose, win))

        best_bid, best_value = 0, -1.0
        values = []
        for b in range(cap + 1):
            total = 0.0
            for t, lose, win in samples:
                if b >= t:
                    # Overpaying by (b - t) costs that many coins at the end.
                    total += _shift(win, b - t, self.temperature)
                else:
                    total += lose
            v = total / len(samples)
            values.append(v)
            if v > best_value + 1e-12:
                best_bid, best_value = b, v
        self.last_search = {"thresholds": [s[0] for s in samples], "values": values, "bid": best_bid}
        return best_bid

    # ------------------------------------------------------------------
    # Player API
    # ------------------------------------------------------------------

    def choose_bid(self, public_state, my_state, auction) -> int:
        my_idx = next(
            i for i, ps in enumerate(public_state.player_states) if ps is my_state
        )
        if self.infer:
            self._record_bid_tables(public_state, my_idx, auction)
        self._prime_models(public_state, my_idx, auction)
        if max_legal_bid(my_state, auction) == 0:
            return 0
        return self._search(public_state, my_state, auction)

    def choose_gem_to_reveal(self, public_state, my_state) -> GemCard:
        colors = sorted({g.color for g in my_state.hand}, key=lambda c: c.value)
        if not self.reveal_search or len(colors) < 2:
            return self._revealer.choose_gem_to_reveal(public_state, my_state)
        my_idx = next(
            i for i, ps in enumerate(public_state.player_states) if ps is my_state
        )
        totals = dict.fromkeys(colors, 0.0)
        for _ in range(max(4, self.worlds // 2)):
            hands, gem_deck, auctions = self._deal_world(public_state, my_idx, None, self._rng)
            seed = self._rng.randrange(2**31)
            for c in colors:
                totals[c] += self._reveal_rollout(public_state, my_idx, hands, gem_deck, auctions, c, seed)
        best = max(colors, key=lambda c: totals[c])
        return next(g for g in my_state.hand if g.color == best)

    def _reveal_rollout(self, state, my_idx, hands, gem_deck, auctions, color, seed) -> float:
        """Finish the current round revealing ``color``, then roll out."""
        rng = random.Random(seed)
        players = self._rollout_players(state, my_idx, rng)
        sim = self._build_state(state, players, hands, gem_deck, auctions)
        me = sim.player_states[my_idx]
        me.hand.remove(_GEM[color])
        sim.value_display[color] += 1
        _replenish_revealed(sim)
        _check_missions(sim)
        return self._play_out(sim, my_idx, rng)

    def explain_bid(self, public_state, my_state, auction) -> list[str]:
        """Run the same search ``choose_bid`` is about to run, and describe it.

        The RNG is restored afterwards, so the real decision that follows
        is identical to the one explained here.
        """
        if max_legal_bid(my_state, auction) == 0:
            return ["no coins to bid with"]
        rng_state = self._rng.getstate()
        try:
            self._search(public_state, my_state, auction)
        finally:
            self._rng.setstate(rng_state)
        s = self.last_search
        ts = sorted(s["thresholds"])
        values = s["values"]
        lines = [
            f"worlds={len(ts)}  price-to-win: median {ts[len(ts) // 2]}, "
            f"range {ts[0]}..{ts[-1]}",
            f"est. win-prob: bid 0 → {values[0]:.2f}, "
            f"bid {s['bid']} → {values[s['bid']]:.2f} (chosen)",
        ]
        if self.identify:
            kinds = {
                public_state.player_states[j].name: self._seat_kind(j)
                for j in self._type_ll
            }
            if kinds:
                lines.append("opponents look like: " + ", ".join(f"{n}={k}" for n, k in kinds.items()))
        return lines


def _clone(model: Player) -> Player:
    """Cheap copy of a model bot: share its (immutable) weights, copy its
    mutable history containers. ~20x faster than ``copy.deepcopy``."""
    clone = copy.copy(model)
    for attr, value in vars(clone).items():
        if isinstance(value, (list, dict, set, random.Random)):
            setattr(clone, attr, copy.copy(value))
    return clone


def _shift(p: float, coins: int, temperature: float) -> float:
    """Re-score a sigmoid win-probability after losing ``coins`` of margin."""
    if coins == 0:
        return p
    p = min(max(p, 1e-9), 1 - 1e-9)
    logit = math.log(p / (1 - p)) - coins / temperature
    return 1.0 / (1.0 + math.exp(-logit))


def _multisets(size: int, limits: list[int], prefix: tuple = ()):
    """Every colour-count tuple summing to ``size`` with ``counts[i] <= limits[i]``."""
    i = len(prefix)
    if i == len(limits) - 1:
        if size <= limits[i]:
            yield prefix + (size,)
        return
    for n in range(min(size, limits[i]) + 1):
        yield from _multisets(size - n, limits, prefix + (n,))
