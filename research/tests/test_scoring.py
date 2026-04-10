import unittest
from collections import Counter

from megagem.cards import Color, InvestCard, LoanCard
from megagem.engine import score_game
from megagem.missions import MissionCard, color_counts_at_least
from megagem.players import RandomAI
from megagem.state import GameState, PlayerState
from megagem.value_charts import VALUE_CHARTS, value_for


class ValueChartTest(unittest.TestCase):
    def test_chart_a_matches_rules(self):
        self.assertEqual(VALUE_CHARTS["A"], [0, 4, 8, 12, 16, 20])

    def test_chart_b_matches_rules(self):
        self.assertEqual(VALUE_CHARTS["B"], [20, 16, 12, 8, 4, 0])

    def test_value_for_clamps_high_counts(self):
        self.assertEqual(value_for("A", 6), 20)
        self.assertEqual(value_for("A", 100), 20)

    def test_value_for_zero(self):
        self.assertEqual(value_for("A", 0), 0)
        self.assertEqual(value_for("B", 0), 20)


class ScoringTest(unittest.TestCase):
    def test_full_scoring(self):
        # Player has:
        #   10 starting coins
        #   2 blue gems in collection
        #   1 mission worth 5
        #   1 loan of 20
        #   1 invest (5-coin) with 3 locked
        ps = PlayerState(name="Test")
        ps.coins = 10
        ps.collection_gems[Color.BLUE] = 2
        ps.completed_missions.append(
            MissionCard(
                name="2 blue",
                coins=5,
                requirement=color_counts_at_least({Color.BLUE: 2}),
                category="pendant",
            )
        )
        ps.loans.append(LoanCard(20))
        ps.investments.append((InvestCard(5), 3))

        # Value display: 4 blue gems → chart A says blue is worth 16 each.
        state = GameState(
            player_states=[ps],
            players=[RandomAI("dummy", seed=0)],
            value_chart="A",
        )
        state.value_display[Color.BLUE] = 4

        results = score_game(state)
        self.assertEqual(len(results), 1)
        r = results[0]
        # gems: 2 blue × 16 = 32
        # missions: 5
        # loans: -20
        # investments: 5 + 3 = 8
        # coins: 10
        # total: 10 + 32 + 5 - 20 + 8 = 35
        self.assertEqual(r["gem_value"], 32)
        self.assertEqual(r["mission_value"], 5)
        self.assertEqual(r["loans_total"], 20)
        self.assertEqual(r["invest_returns"], 8)
        self.assertEqual(r["total"], 35)

    def test_score_reveals_remaining_hand(self):
        from megagem.cards import GemCard
        ps = PlayerState(name="Test")
        ps.coins = 0
        ps.hand = [GemCard(Color.PINK), GemCard(Color.PINK)]
        state = GameState(
            player_states=[ps],
            players=[RandomAI("dummy", seed=0)],
            value_chart="A",
        )
        score_game(state)
        # Hand should now be empty and value display should reflect the cards.
        self.assertEqual(ps.hand, [])
        self.assertEqual(state.value_display[Color.PINK], 2)


if __name__ == "__main__":
    unittest.main()
