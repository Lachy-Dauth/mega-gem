import random
import unittest
from collections import Counter

from megagem.cards import Color, GemCard, InvestCard, LoanCard, TreasureCard
from megagem.engine import is_game_over, play_round, score_game, setup_game
from megagem.players import (
    HeuristicAI,
    RandomAI,
    _expected_final_display,
    _treasure_value,
)
from megagem.state import GameState, PlayerState


def _empty_state(chart: str = "A") -> GameState:
    me = PlayerState(name="Me", coins=20)
    opp = PlayerState(name="Opp", coins=20)
    return GameState(
        player_states=[me, opp],
        players=[],
        value_chart=chart,
    )


class ExpectedDisplayTest(unittest.TestCase):
    def test_my_hand_is_counted_directly(self):
        state = _empty_state()
        me = state.player_states[0]
        me.hand = [GemCard(Color.BLUE), GemCard(Color.BLUE), GemCard(Color.GREEN)]
        # No opponents have any hidden cards (no hand, no deck), nothing in display.
        result = _expected_final_display(state, me)
        self.assertEqual(result[Color.BLUE], 2)
        self.assertEqual(result[Color.GREEN], 1)
        self.assertEqual(result[Color.PINK], 0)

    def test_hidden_cards_estimated_uniformly(self):
        state = _empty_state()
        me = state.player_states[0]
        opp = state.player_states[1]
        # Opponent has 2 hidden hand cards, deck has 8 hidden cards.
        # 4 colors are completely hidden (no display, no collections),
        # so 4 colors split 6 cards each across (2 + 8 = 10) slots.
        # opp_share = 2/10 = 0.2
        opp.hand = [GemCard(Color.BLUE), GemCard(Color.GREEN)]  # contents irrelevant
        # Pretend the deck has 8 cards (we just need len; contents don't matter
        # to _expected_final_display because deck is hidden by definition).
        state.gem_deck = [GemCard(Color.BLUE)] * 8
        result = _expected_final_display(state, me)
        # For each color: hidden_of_color = 6, opp_share = 0.2 → expected ≈ 1.2
        for color in Color:
            self.assertAlmostEqual(result[color], 6 * 0.2, places=4)


class TreasureValueTest(unittest.TestCase):
    def test_one_gem_uses_chart(self):
        state = _empty_state(chart="A")
        me = state.player_states[0]
        # No hidden cards anywhere → expected display is exactly current display.
        # Reveal a Green gem for sale.
        state.revealed_gems = [GemCard(Color.GREEN)]
        # Display already has 3 greens → with another in collection it's still
        # the chart-A index 3 = 12 coins per gem.
        state.value_display[Color.GREEN] = 3
        value = _treasure_value(TreasureCard(1), state, me)
        # No mission contribution and no opponents — value is 12.
        self.assertEqual(value, 12)

    def test_two_gems_sum_marginals(self):
        state = _empty_state(chart="A")
        me = state.player_states[0]
        state.revealed_gems = [GemCard(Color.BLUE), GemCard(Color.PINK)]
        state.value_display[Color.BLUE] = 2  # chart A: 8
        state.value_display[Color.PINK] = 4  # chart A: 16
        value = _treasure_value(TreasureCard(2), state, me)
        self.assertEqual(value, 8 + 16)


class HeuristicBidTest(unittest.TestCase):
    def test_loan_skipped_when_cash_healthy(self):
        ai = HeuristicAI("X", seed=0)
        state = _empty_state()
        me = state.player_states[0]
        me.coins = 25
        bid = ai.choose_bid(state, me, LoanCard(20))
        self.assertEqual(bid, 0)

    def test_invest_with_surplus_bids_at_least_one(self):
        ai = HeuristicAI("X", seed=0)
        state = _empty_state()
        me = state.player_states[0]
        me.coins = 25
        bid = ai.choose_bid(state, me, InvestCard(10))
        self.assertGreaterEqual(bid, 1)

    def test_treasure_bid_capped_by_coins(self):
        ai = HeuristicAI("X", seed=0)
        state = _empty_state(chart="A")
        me = state.player_states[0]
        me.coins = 3
        state.revealed_gems = [GemCard(Color.BLUE), GemCard(Color.PINK)]
        bid = ai.choose_bid(state, me, TreasureCard(1))
        self.assertLessEqual(bid, 3)
        self.assertGreaterEqual(bid, 0)


class HeuristicRevealTest(unittest.TestCase):
    def test_chart_a_reveals_color_we_dominate(self):
        ai = HeuristicAI("X", seed=0)
        state = _empty_state(chart="A")
        me = state.player_states[0]
        opp = state.player_states[1]
        me.collection_gems[Color.BLUE] = 3
        opp.collection_gems[Color.GREEN] = 3
        me.hand = [GemCard(Color.BLUE), GemCard(Color.GREEN)]
        # Chart A is increasing → reveal BLUE (we dominate it).
        revealed = ai.choose_gem_to_reveal(state, me)
        self.assertEqual(revealed.color, Color.BLUE)

    def test_chart_b_reveals_color_opponent_dominates(self):
        ai = HeuristicAI("X", seed=0)
        state = _empty_state(chart="B")
        me = state.player_states[0]
        opp = state.player_states[1]
        me.collection_gems[Color.BLUE] = 3
        opp.collection_gems[Color.GREEN] = 3
        me.hand = [GemCard(Color.BLUE), GemCard(Color.GREEN)]
        # Chart B is decreasing → reveal GREEN to drag opponents' value down.
        revealed = ai.choose_gem_to_reveal(state, me)
        self.assertEqual(revealed.color, Color.GREEN)


class HeadToHeadTest(unittest.TestCase):
    """Heuristic AI should comfortably beat RandomAI over many games."""

    def test_heuristic_wins_majority(self):
        wins = 0
        ties = 0
        games = 60
        for seed in range(games):
            players = [
                HeuristicAI("Heur", seed=seed * 11),
                RandomAI("R1", seed=seed * 11 + 1),
                RandomAI("R2", seed=seed * 11 + 2),
                RandomAI("R3", seed=seed * 11 + 3),
            ]
            state = setup_game(players, chart="A", seed=seed)
            rng = random.Random(seed)
            while not is_game_over(state):
                play_round(state, rng=rng)
            scores = score_game(state)
            heur_score = scores[0]["total"]
            best_random = max(s["total"] for s in scores[1:])
            if heur_score > best_random:
                wins += 1
            elif heur_score == best_random:
                ties += 1
        # Want a clear majority — anything below 60% is a sign the strategy
        # isn't actually adding value over random.
        win_rate = wins / games
        self.assertGreater(
            win_rate,
            0.6,
            f"Heuristic only won {wins}/{games} ({win_rate:.0%}); ties={ties}",
        )


if __name__ == "__main__":
    unittest.main()
