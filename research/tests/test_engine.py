import random
import unittest

from megagem.cards import LoanCard, TreasureCard
from megagem.engine import (
    _resolve_winner,
    clamp_bid,
    is_game_over,
    max_legal_bid,
    play_round,
    score_game,
    setup_game,
)
from megagem.players import Player, RandomAI
from megagem.state import PlayerState


class FakePlayer(Player):
    """Player that always returns whatever value it's been told."""

    def __init__(self, name: str, bid_value: object):
        super().__init__(name)
        self.bid_value = bid_value

    def choose_bid(self, public_state, my_state, auction):
        return self.bid_value

    def choose_gem_to_reveal(self, public_state, my_state):
        return my_state.hand[0]


class ClampBidTest(unittest.TestCase):
    def test_clamps_above_coins(self):
        ps = PlayerState(name="x", coins=10)
        self.assertEqual(clamp_bid(9999, ps, TreasureCard(1)), 10)

    def test_clamps_below_zero(self):
        ps = PlayerState(name="x", coins=10)
        self.assertEqual(clamp_bid(-5, ps, TreasureCard(1)), 0)

    def test_loan_allows_overdraft(self):
        ps = PlayerState(name="x", coins=3)
        # Loan of 20 → may bid up to coins + 20 = 23
        self.assertEqual(max_legal_bid(ps, LoanCard(20)), 23)
        self.assertEqual(clamp_bid(23, ps, LoanCard(20)), 23)
        self.assertEqual(clamp_bid(24, ps, LoanCard(20)), 23)

    def test_non_int_becomes_zero(self):
        ps = PlayerState(name="x", coins=10)
        self.assertEqual(clamp_bid(None, ps, TreasureCard(1)), 0)
        self.assertEqual(clamp_bid("nope", ps, TreasureCard(1)), 0)


class TieBreakTest(unittest.TestCase):
    def test_first_auction_random(self):
        rng = random.Random(0)
        # Three-way tie, no previous winner — should pick uniformly.
        results = set()
        for _ in range(50):
            results.add(_resolve_winner([5, 5, 5], None, 3, rng))
        self.assertEqual(results, {0, 1, 2})

    def test_left_of_previous_winner(self):
        rng = random.Random(0)
        # Players [0,1,2,3], last winner = 2. "Left" = next in seating order.
        # Walk 3, 0, 1, 2. First tied player encountered is 3.
        winner = _resolve_winner([5, 5, 5, 5], 2, 4, rng)
        self.assertEqual(winner, 3)

    def test_left_skips_non_tied(self):
        rng = random.Random(0)
        # Bids: only players 1 and 3 tie. Last winner = 0.
        # Walk 1 (tied → win).
        winner = _resolve_winner([0, 5, 0, 5], 0, 4, rng)
        self.assertEqual(winner, 1)

    def test_left_wraps_around(self):
        rng = random.Random(0)
        # Bids: only players 0 and 1 tie. Last winner = 2.
        # Walk 3, 0 (tied → win).
        winner = _resolve_winner([5, 5, 0, 0], 2, 4, rng)
        self.assertEqual(winner, 0)


class SmokeTest(unittest.TestCase):
    def test_full_random_games(self):
        for n in (3, 4, 5):
            for seed in range(10):
                players = [RandomAI(f"AI{i}", seed=seed * 17 + i) for i in range(n)]
                state = setup_game(players, chart="A", seed=seed)
                rng = random.Random(seed)
                while not is_game_over(state):
                    play_round(state, rng=rng)
                scores = score_game(state)
                self.assertEqual(len(scores), n)
                # All hands should be empty after scoring.
                for ps in state.player_states:
                    self.assertEqual(ps.hand, [])
                # Auction deck must be drained at game end.
                self.assertFalse(state.auction_deck and state.gem_deck)

    def test_clamping_protects_engine_from_illegal_ai(self):
        players = [
            FakePlayer("Cheater", 10**9),
            FakePlayer("Negative", -100),
            RandomAI("Honest", seed=0),
        ]
        state = setup_game(players, chart="A", seed=1)
        rng = random.Random(2)
        # Should not raise even though FakePlayer returns illegal bids every round.
        rounds = 0
        while not is_game_over(state) and rounds < 30:
            play_round(state, rng=rng)
            rounds += 1
        score_game(state)


if __name__ == "__main__":
    unittest.main()
