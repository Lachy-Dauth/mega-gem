import random
import unittest

from megagem.cards import LoanCard, TreasureCard
from megagem.engine import is_game_over, max_legal_bid, play_round, score_game, setup_game
from megagem.players import Evo4AI, HeuristicAI, RandomAI
from megagem.players.nash import NashAI


def _small(name, seed, **kw):
    """A cheap NashAI for tests: few worlds and solver iterations."""
    kw.setdefault("worlds", 3)
    kw.setdefault("iterations", 40)
    return NashAI(name, seed=seed, **kw)


def _duel(v0, v1, price=5):
    """One 2-player world: scores when each player wins free / paying ``price``."""
    free0, free1 = [50 + v0 / 2, 50 - v0 / 2], [50 - v1 / 2, 50 + v1 / 2]
    return [
        (free0, [free0[0] - price, free0[1]], price),
        (free1, [free1[0], free1[1] - price], price),
    ]


class SolverTest(unittest.TestCase):
    """The auction solver on games whose answer is known."""

    def test_strategy_is_a_distribution(self):
        bot = NashAI("x", seed=0, iterations=200)
        strat = bot._solve([_duel(12, 6)], caps=[30, 30], ranks=[0, 1], my_idx=0)
        self.assertAlmostEqual(sum(strat), 1.0, places=9)
        self.assertTrue(all(p >= 0 for p in strat))

    def test_duel_bids_up_to_the_margin_swing(self):
        """Two players, margin utility: winning swings the margin by
        v0 + v1, so equilibrium competition drives bids to about that."""
        bot = NashAI("x", seed=0, iterations=400)
        strat = bot._solve([_duel(12, 6)], caps=[30, 30], ranks=[0, 1], my_idx=0)
        mean = sum(b * p for b, p in enumerate(strat))
        self.assertTrue(10 <= mean <= 18, mean)

    def test_worthless_item_gets_no_bid(self):
        bot = NashAI("x", seed=0, iterations=200)
        strat = bot._solve([_duel(0, 0)], caps=[30, 30], ranks=[0, 1], my_idx=0)
        self.assertGreater(strat[0], 0.9)

    def test_paying_more_never_looks_free(self):
        """Rollout noise where the paid branch scores *better* must not make
        a high bid attractive: the coin cost is floored at 1:1."""
        bot = NashAI("x", seed=0, iterations=200)
        free0, free1 = [55, 45], [45, 55]
        noisy = [(free0, [free0[0] + 3, free0[1]], 5), (free1, [free1[0], free1[1] + 3], 5)]
        strat = bot._solve([noisy], caps=[30, 30], ranks=[0, 1], my_idx=0)
        mean = sum(b * p for b, p in enumerate(strat))
        self.assertLess(mean, 25)


class PlayTest(unittest.TestCase):
    def test_bids_are_legal_and_games_finish(self):
        bot = _small("N", 1)
        players = [bot, Evo4AI("a"), HeuristicAI("b"), RandomAI("c", seed=2)]
        state = setup_game(players, chart="E", seed=2)
        rng = random.Random(2)
        while not is_game_over(state):
            ps = state.player_states[0]
            auction = state.auction_deck[-1]
            cap = max_legal_bid(ps, auction)
            result = play_round(state, rng=rng)
            self.assertTrue(0 <= result["bids"][0] <= cap)
        self.assertEqual(len(score_game(state)), 4)

    def test_seeded_bot_is_reproducible(self):
        def run():
            players = [_small("N", 7)] + [Evo4AI(f"E{i}") for i in range(3)]
            state = setup_game(players, chart="C", seed=8)
            rng = random.Random(8)
            return [play_round(state, rng=rng)["bids"] for _ in range(4)]
        self.assertEqual(run(), run())

    def test_bid_ignores_the_real_hidden_cards(self):
        """Only public information + own hand: reshuffling the opponents'
        hands and the deck must not change the decision."""
        players = [HeuristicAI(f"p{i}", seed=i) for i in range(4)]
        state = setup_game(players, chart="A", seed=3)
        rng = random.Random(3)
        for _ in range(3):
            play_round(state, rng=rng)
        me = state.player_states[0]
        first = _small("me", 11).choose_bid(state, me, TreasureCard(1))
        hidden = list(state.gem_deck)
        for ps in state.player_states[1:]:
            hidden += ps.hand
        random.Random(5).shuffle(hidden)
        pos = 0
        for ps in state.player_states[1:]:
            ps.hand = hidden[pos:pos + len(ps.hand)]
            pos += len(ps.hand)
        state.gem_deck = hidden[pos:]
        self.assertEqual(_small("me", 11).choose_bid(state, me, TreasureCard(1)), first)

    def test_never_buys_a_loan_for_almost_its_face_value(self):
        """Regression: valuing bids 1:1 against a free-item rollout once
        made it pay 19 for a 20-coin loan."""
        players = [HeuristicAI(f"p{i}", seed=i) for i in range(4)]
        state = setup_game(players, chart="A", seed=1)
        bot = _small("me", 3, worlds=4, iterations=80)
        me = state.player_states[0]
        bids = [bot.choose_bid(state, me, LoanCard(20)) for _ in range(5)]
        self.assertLess(max(bids), 15, bids)

    def test_explanation_matches_the_real_bid(self):
        bot = _small("N", 4)
        players = [bot] + [Evo4AI(f"E{i}") for i in range(3)]
        state = setup_game(players, chart="B", seed=4)
        auction = state.auction_deck.pop()
        me = state.player_states[0]
        lines = bot.explain_bid(state, me, auction)
        explained = bot.last_solution["bid"]
        self.assertTrue(any("equilibrium mix" in line for line in lines))
        self.assertEqual(bot.choose_bid(state, me, auction), explained)


class FactoryTest(unittest.TestCase):
    def test_cli_and_browser_build_nash(self):
        from megagem.__main__ import AI_FACTORIES as CLI
        from megagem.browser import AI_FACTORIES as WEB, DEFAULT_WEIGHTS_DIR

        self.assertIsInstance(CLI["nash"]("n", seed=1, num_players=4), NashAI)
        bot = WEB["nash"]("n", seed=1, num_players=3, weights_dir=DEFAULT_WEIGHTS_DIR)
        self.assertIsInstance(bot, NashAI)


if __name__ == "__main__":
    unittest.main()
