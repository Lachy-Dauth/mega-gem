import random
import unittest
from collections import Counter

from megagem.cards import Color, GemCard, InvestCard, LoanCard, TreasureCard
from megagem.engine import is_game_over, play_round, score_game, setup_game
from megagem.missions import MissionCard, color_counts_at_least
from megagem.players import (
    HeuristicAI,
    HyperAdaptiveSplitAI,
    RandomAI,
    _BidModel,
    _compute_discount_features,
    _ev_remaining_auctions,
    _expected_final_display,
    _hyper_compute_discount_features,
    _hyper_ev_remaining_auctions,
    _hyper_expected_per_gem_value,
    _hyper_hidden_distribution,
    _hyper_treasure_gem_value,
    _hyper_treasure_value,
    _treasure_value,
)
from megagem.value_charts import value_for
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


# ----------------------------------------------------------------------------
# HyperAdaptiveSplitAI tests
# ----------------------------------------------------------------------------


class HyperAdaptiveSplitBidTest(unittest.TestCase):
    """Per-head bid tests for HyperAdaptiveSplitAI."""

    def _state(self, chart: str = "A") -> tuple[GameState, PlayerState]:
        state = _empty_state(chart=chart)
        me = state.player_states[0]
        # Plausible mid-game state — auction deck non-empty so EV math runs.
        state.auction_deck = [TreasureCard(1)] * 10
        state.gem_deck = [GemCard(Color.BLUE)] * 5
        return state, me

    def test_invest_uses_invest_model_not_treasure_model(self):
        # Treasure model returns 1.0, invest model returns 0.0. If choose_bid
        # were leaking the treasure head into invests, we'd see a large bid;
        # instead the floor (1) kicks in because the head is silent.
        treasure = _BidModel(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # always 1
        invest = _BidModel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)    # always 0
        loan = _BidModel(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ai = HyperAdaptiveSplitAI("S", treasure=treasure, invest=invest, loan=loan)

        state, me = self._state()
        me.coins = 30
        bid = ai.choose_bid(state, me, InvestCard(amount=10))
        self.assertEqual(bid, 1)  # forced token bid, not 30

    def test_loan_zero_when_loan_model_negative(self):
        # Deeply negative bias → discount clamped to 0 → bid 0 even with cap > 0.
        treasure = _BidModel(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        invest = _BidModel(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        loan = _BidModel(-5.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # always 0
        ai = HyperAdaptiveSplitAI("S", treasure=treasure, invest=invest, loan=loan)

        state, me = self._state()
        me.coins = 20
        self.assertEqual(ai.choose_bid(state, me, LoanCard(amount=10)), 0)

    def test_from_weights_round_trip(self):
        weights = [
            # treasure
            0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
            # invest
            0.21, 0.22, 0.23, 0.24, 0.25, 0.26,
            # loan
            0.31, 0.32, 0.33, 0.34, 0.35, 0.36,
        ]
        ai = HyperAdaptiveSplitAI.from_weights("Evo", weights)
        self.assertEqual(ai.treasure_model.bias, 0.11)
        self.assertEqual(ai.treasure_model.w_progress, 0.12)
        self.assertEqual(ai.treasure_model.w_my_cash, 0.13)
        self.assertEqual(ai.treasure_model.w_avg_cash, 0.14)
        self.assertEqual(ai.treasure_model.w_top_cash, 0.15)
        self.assertEqual(ai.treasure_model.w_variance, 0.16)
        self.assertEqual(ai.invest_model.bias, 0.21)
        self.assertEqual(ai.invest_model.w_variance, 0.26)
        self.assertEqual(ai.loan_model.bias, 0.31)
        self.assertEqual(ai.loan_model.w_variance, 0.36)
        # Wrong-length input should raise.
        with self.assertRaises(ValueError):
            HyperAdaptiveSplitAI.from_weights("Bad", [0.0] * 17)


class HyperAdaptiveSplitHeadToHeadTest(unittest.TestCase):
    """Smoke test: defaults shouldn't be broken."""

    def test_split_beats_random_chart_a(self):
        wins = 0
        games = 60
        for seed in range(games):
            players = [
                HyperAdaptiveSplitAI("HS", seed=seed * 23),
                RandomAI("R1", seed=seed * 23 + 1),
                RandomAI("R2", seed=seed * 23 + 2),
                RandomAI("R3", seed=seed * 23 + 3),
            ]
            state = setup_game(players, chart="A", seed=seed)
            rng = random.Random(seed)
            while not is_game_over(state):
                play_round(state, rng=rng)
            scores = score_game(state)
            if scores[0]["total"] > max(s["total"] for s in scores[1:]):
                wins += 1
        win_rate = wins / games
        self.assertGreater(
            win_rate, 0.6, f"HyperAdaptiveSplit only won {win_rate:.0%} on chart A"
        )


if __name__ == "__main__":
    unittest.main()
