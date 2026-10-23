"""NashAI — bid from an equilibrium of each auction, not a model of the bots.

SearchAI is an *exploiter*: it predicts each opponent's exact bid with a
model of a zoo bot and outbids it by one. That is devastating against
deterministic bots and exactly what a Nash player should not rely on.
NashAI instead assumes every opponent is as rational as it is:

1. **Worlds.** As in SearchAI, deal the unseen gem pool into opponents'
   hands + the deck (public information only) a few times.
2. **What the auction is worth to everyone.** In each world, roll the
   game out once per possible winner ``k`` of this auction (the item
   handed to ``k`` for free; every seat then plays the rollout policy).
   That gives every player's payoff if they win and if each rival wins
   instead — the externality matters: losing a treasure to the leader is
   worse than losing it to the player in last.
3. **Solve the auction.** Treat this one auction as a Bayesian game:
   each opponent knows its world (its hand), I only know the
   distribution over worlds. Payoffs are a sigmoid of the final margin
   over the best opponent (a smooth win probability), minus the price
   for the winner. Regret matching+ with linear averaging finds an
   approximate equilibrium in which every player's bid distribution is
   a best response to everyone else's — ties resolved by the real
   tie-break rule.
4. **Play the mixed strategy.** Sample my bid from my equilibrium
   distribution. An opponent who knows exactly how NashAI plays still
   can't predict the bid it will make, only its distribution.

This is an equilibrium of the *stage game* with rollout continuation
values, not of the whole game, but it carries no per-opponent state and
is stochastic by design, which is what makes it hard to exploit (README
§12 has the exploitability measurements).

**Restricted Nash response** (``model_weight=α``, default 0.75): in the
solver each opponent plays what the rollout policy (Evo4) would bid
with probability α, and best-responds otherwise. α = 0 is the pure
equilibrium bot; α > 0 exploits Evo4-like underbidding. In measurement
α = 0.75 was far stronger against the zoo and statistically as hard to
exploit as α = 0: its bids stay mixed, so the exploiter still can't
predict them.

Reveals reuse :class:`HeuristicAI`'s rule (reveals don't change final
values; every hand ends up in the Value Display).
"""

from __future__ import annotations

import math
import random
from typing import Callable

from ..cards import AuctionCard, GemCard, InvestCard, LoanCard, TreasureCard
from ..engine import (
    _apply_invest,
    _apply_loan,
    _apply_treasure,
    _check_missions,
    _replenish_revealed,
    is_game_over,
    max_legal_bid,
    play_round,
    score_game,
)
from ..state import GameState
from .base import Player
from .evo4 import Evo4AI
from .heuristic import HeuristicAI
from .search import build_world_state, deal_world

ModelFactory = Callable[[str, int], Player]


def _sigmoid(x: float) -> float:
    if x < -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class NashAI(Player):
    """Sample bids from an approximate equilibrium of each auction."""

    DEFAULT_WORLDS = 12
    DEFAULT_ITERATIONS = 150
    DEFAULT_TEMPERATURE = 6.0
    # Restricted Nash response weight: each opponent plays the model's bid
    # with this probability. 0.75 was much stronger against the zoo than
    # 0, 0.25 or 0.5, and its exploitability was statistically the same.
    DEFAULT_MODEL_WEIGHT = 0.75
    # Ignore bids above this (never sensible; bounds the solver's cost).
    MAX_BID = 40

    def __init__(
        self,
        name: str,
        *,
        seed: int | None = None,
        worlds: int | None = None,
        iterations: int | None = None,
        temperature: float | None = None,
        opponent_types: int | None = 3,
        model_weight: float | None = None,
        model_weights: list[float] | None = None,
        rollout_model: ModelFactory | None = None,
    ) -> None:
        super().__init__(name)
        self._rng = random.Random(seed)
        self.worlds = worlds if worlds is not None else self.DEFAULT_WORLDS
        self.iterations = iterations if iterations is not None else self.DEFAULT_ITERATIONS
        self.opponent_types = opponent_types
        self.model_weight = (
            model_weight if model_weight is not None else self.DEFAULT_MODEL_WEIGHT
        )
        self.temperature = (
            temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        )
        if rollout_model is None:
            if model_weights is None:
                rollout_model = lambda n, s: Evo4AI(n, seed=s)  # noqa: E731
            else:
                rollout_model = lambda n, s: Evo4AI.from_weights(n, model_weights, seed=s)  # noqa: E731
        self._rollout_model = rollout_model
        self._revealer = HeuristicAI(name)
        self.last_solution: dict | None = None

    # ------------------------------------------------------------------
    # Continuation values
    # ------------------------------------------------------------------

    def _rollout_winner(
        self,
        state: GameState,
        my_idx: int,
        world,
        auction: AuctionCard,
        winner: int,
        seed: int,
        price: int,
    ) -> list[int]:
        """Final scores when ``winner`` takes this auction paying ``price``."""
        hands, gem_deck, auctions = world
        rng = random.Random(seed)
        players = [
            self._rollout_model(f"p{i}", rng.randrange(2**31))
            for i in range(state.num_players)
        ]
        sim = build_world_state(state, players, hands, gem_deck, auctions)
        ws = sim.player_states[winner]
        if isinstance(auction, TreasureCard):
            ws.coins -= price
            _apply_treasure(sim, ws, auction)
        elif isinstance(auction, LoanCard):
            _apply_loan(ws, price, auction)
        elif isinstance(auction, InvestCard):
            _apply_invest(ws, price, auction)
        if ws.hand:
            chooser = self._revealer if winner == my_idx else players[winner]
            gem = chooser.choose_gem_to_reveal(sim, ws)
            if gem not in ws.hand:
                gem = ws.hand[0]
            ws.hand.remove(gem)
            sim.value_display[gem.color] += 1
        _replenish_revealed(sim)
        _check_missions(sim)
        sim.last_winner_idx = winner
        while not is_game_over(sim):
            play_round(sim, rng=rng)
        return [row["total"] for row in score_game(sim)]

    def _model_bids(self, state: GameState, world, auction: AuctionCard) -> list[int]:
        """What the rollout policy would bid from each seat in this world."""
        sim = build_world_state(state, [], *world)
        bids = []
        for i, ps in enumerate(sim.player_states):
            model = self._rollout_model(f"m{i}", 0)
            bids.append(max(0, min(int(model.choose_bid(sim, ps, auction)), max_legal_bid(ps, auction))))
        return bids

    # ------------------------------------------------------------------
    # Equilibrium of one auction
    # ------------------------------------------------------------------

    @staticmethod
    def _tie_ranks(state: GameState) -> list[int]:
        """Lower rank wins a tie: seat order starting left of the last winner."""
        n = state.num_players
        last = state.last_winner_idx
        start = 0 if last is None else (last + 1) % n
        ranks = [0] * n
        for pos in range(n):
            ranks[(start + pos) % n] = pos
        return ranks

    def _solve(
        self,
        scores: list[list[list[int]]],
        caps: list[int],
        ranks: list[int],
        my_idx: int,
        model_bids: list[list[int]] | None = None,
    ) -> list[float]:
        """Regret matching+ on the auction's Bayesian game; my average strategy.

        ``scores[w][k]`` is ``(free, paid, price)``: everyone's final
        scores in world w when k won the auction for free, and when k won
        paying ``price``.
        """
        n = len(caps)
        W = len(scores)
        T = self.temperature
        players = range(n)

        def margins(row):
            return [row[i] - max(row[j] for j in players if j != i) for i in players]

        # What a coin bid costs each player, in final margin. Coins are
        # worth more than 1 mid-game (they buy underpriced items later), so
        # measure it from the two rollouts instead of assuming 1:1 — and
        # never let rollout noise make paying more look free or good.
        slope = []
        for i in players:
            samples = [
                (margins(scores[w][i][1])[i] - margins(scores[w][i][0])[i]) / scores[w][i][2]
                for w in range(W) if scores[w][i][2] > 0
            ]
            slope.append(min(-1.0, sum(samples) / len(samples)) if samples else -1.0)

        # Winning margin at bid b is intercept + slope * b (intercept fitted
        # from both rollouts). Losing: the rival won paying its mid price.
        intercept = [[0.0] * n for _ in range(W)]
        lose_util = [[[0.0] * n for _ in players] for _ in range(W)]
        for w in range(W):
            for k in players:
                free, paid, price = scores[w][k]
                m_free, m_paid = margins(free)[k], margins(paid)[k]
                intercept[w][k] = (m_free + m_paid - slope[k] * price) / 2.0
                for i, m in enumerate(margins(paid)):
                    if i != k:
                        lose_util[w][i][k] = _sigmoid(m / T)

        def util_win(w, i, b):
            return _sigmoid((intercept[w][i] + slope[i] * b) / T)

        # Bid grids: drop bids that are worse than losing in every world.
        grids = []
        for i in players:
            top = min(caps[i], self.MAX_BID)
            worthwhile = 0
            for b in range(top + 1):
                if any(
                    util_win(w, i, b) > min(lose_util[w][i][k] for k in players if k != i) - 1e-9
                    for w in range(W)
                ):
                    worthwhile = b
            if self.model_weight > 0 and model_bids and i != my_idx:
                worthwhile = max(worthwhile, max(min(model_bids[w][i], top) for w in range(W)))
            grids.append(worthwhile + 1)
        win_util = [
            [[util_win(w, i, b) for b in range(grids[i])] for i in players]
            for w in range(W)
        ]

        # Types. I don't know which world is real, so I have one strategy.
        # An opponent knows its own hand but not the rest of the world, so
        # its type is how much it values winning: worlds are bucketed by
        # that value and the opponent plays one strategy per bucket,
        # averaging over the worlds it can't tell apart.
        K = self.opponent_types if self.opponent_types else W
        bucket = {}
        for i in players:
            if i == my_idx:
                continue
            order = sorted(range(W), key=lambda w: intercept[w][i])
            for rank, w in enumerate(order):
                bucket[(i, w)] = rank * K // W

        def uniform(m):
            return [1.0 / m] * m

        strat = {
            (i, b): uniform(grids[i]) for i in players if i != my_idx for b in range(K)
        }
        my_strat = uniform(grids[my_idx])
        regret = {key: [0.0] * grids[key[0]] for key in strat}
        my_regret = [0.0] * grids[my_idx]
        my_avg = [0.0] * grids[my_idx]

        bucket_size = {key: 0 for key in strat}
        for (i, w), b in bucket.items():
            bucket_size[(i, b)] += 1

        G = max(grids)
        for t in range(1, self.iterations + 1):
            my_util = [0.0] * grids[my_idx]
            type_util = {key: [0.0] * grids[key[0]] for key in strat}
            alpha = self.model_weight if model_bids else 0.0
            for w in range(W):
                pmf = [my_strat if i == my_idx else strat[(i, bucket[(i, w)])] for i in players]
                if alpha > 0.0:
                    # Restricted Nash response: with probability alpha each
                    # opponent plays the model's bid for this world; otherwise
                    # it is free to best-respond (the free part's regrets are
                    # computed against these mixed distributions below).
                    for i in players:
                        if i == my_idx:
                            continue
                        mixed = [(1.0 - alpha) * p for p in pmf[i]]
                        mixed[min(model_bids[w][i], grids[i] - 1)] += alpha
                        pmf[i] = mixed
                # lt[i][x] = P(bid_i < x), le[i][x] = P(bid_i <= x).
                lt, le = [], []
                for i in players:
                    acc, below, upto = 0.0, [], []
                    p_i, g_i = pmf[i], grids[i]
                    for x in range(G):
                        below.append(acc)
                        if x < g_i:
                            acc += p_i[x]
                        upto.append(acc)
                    lt.append(below)
                    le.append(upto)

                # beat[f][x] = P(bidding x beats everyone else); f wins a tie
                # against m iff it is ahead of m in the tie-break order.
                beat = []
                for f in players:
                    g_f = grids[f]
                    row = None
                    for m in players:
                        if m == f:
                            continue
                        col = le[m] if ranks[f] < ranks[m] else lt[m]
                        row = col[:g_f] if row is None else [a * b for a, b in zip(row, col)]
                    beat.append(row)
                # Each player's overall chance of winning, for "who wins if I don't".
                p_win = [sum(p * b for p, b in zip(pmf[k], beat[k])) for k in players]

                for f in players:
                    rest = 1.0 - p_win[f]
                    if rest > 1e-12:
                        lose = sum(
                            p_win[k] * lose_util[w][f][k] for k in players if k != f
                        ) / rest
                    else:
                        lose = sum(lose_util[w][f][k] for k in players if k != f) / (n - 1)
                    util = [b * u + (1.0 - b) * lose for b, u in zip(beat[f], win_util[w][f])]
                    if f == my_idx:
                        for x in range(grids[f]):
                            my_util[x] += util[x] / W
                    else:
                        acc = type_util[(f, bucket[(f, w)])]
                        for x in range(grids[f]):
                            acc[x] += util[x]

            for key, util in type_util.items():
                size = bucket_size[key]
                if size:
                    _regret_step(regret[key], strat[key], [u / size for u in util])
            _regret_step(my_regret, my_strat, my_util)
            for x in range(grids[my_idx]):
                my_avg[x] += t * my_strat[x]

        total = sum(my_avg)
        return [p / total for p in my_avg]

    # ------------------------------------------------------------------
    # Player API
    # ------------------------------------------------------------------

    def choose_bid(self, public_state, my_state, auction) -> int:
        if max_legal_bid(my_state, auction) == 0:
            return 0
        my_idx = next(
            i for i, ps in enumerate(public_state.player_states) if ps is my_state
        )
        n = public_state.num_players
        caps = [max_legal_bid(ps, auction) for ps in public_state.player_states]
        # A second, informative price per winner to measure what coins cost.
        probe = [min(c, max(1, c // 2)) for c in caps]
        scores = []
        model_bids = []
        for _ in range(self.worlds):
            world = deal_world(public_state, my_idx, self._rng)
            seed = self._rng.randrange(2**31)
            if self.model_weight > 0:
                model_bids.append(self._model_bids(public_state, world, auction))
            per_winner = []
            for k in range(n):
                free = self._rollout_winner(public_state, my_idx, world, auction, k, seed, 0)
                paid = (
                    self._rollout_winner(public_state, my_idx, world, auction, k, seed, probe[k])
                    if probe[k] > 0 else free
                )
                per_winner.append((free, paid, probe[k]))
            scores.append(per_winner)
        strategy = self._solve(
            scores, caps, self._tie_ranks(public_state), my_idx, model_bids or None
        )
        # Drop dust left over from early iterations before sampling.
        cleaned = [p if p >= 0.01 else 0.0 for p in strategy]
        if not any(cleaned):
            cleaned = strategy
        bid = self._rng.choices(range(len(cleaned)), weights=cleaned)[0]
        self.last_solution = {"strategy": strategy, "bid": bid}
        return bid

    def choose_gem_to_reveal(self, public_state, my_state) -> GemCard:
        return self._revealer.choose_gem_to_reveal(public_state, my_state)

    def explain_bid(self, public_state, my_state, auction) -> list[str]:
        """Re-run the solver on a saved RNG state and show my mixed strategy."""
        if max_legal_bid(my_state, auction) == 0:
            return ["no coins to bid with"]
        rng_state = self._rng.getstate()
        try:
            self.choose_bid(public_state, my_state, auction)
        finally:
            self._rng.setstate(rng_state)
        strategy = self.last_solution["strategy"]
        support = [(b, p) for b, p in enumerate(strategy) if p >= 0.05]
        mix = ", ".join(f"{b}: {p:.0%}" for b, p in support) or "—"
        return [
            f"worlds={self.worlds}  equilibrium mix over bids {{{mix}}}",
            f"sampled bid {self.last_solution['bid']}",
        ]


def _regret_step(regret: list[float], strat: list[float], util: list[float]) -> None:
    """One regret-matching+ update, in place."""
    ev = sum(p * u for p, u in zip(strat, util))
    positive = 0.0
    for x, u in enumerate(util):
        r = regret[x] + u - ev
        regret[x] = r if r > 0.0 else 0.0
        positive += regret[x]
    if positive > 0.0:
        for x in range(len(strat)):
            strat[x] = regret[x] / positive
    else:
        m = len(strat)
        for x in range(m):
            strat[x] = 1.0 / m
