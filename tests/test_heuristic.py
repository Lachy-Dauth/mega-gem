import random
import unittest
from collections import Counter

from megagem.cards import Color, GemCard, InvestCard, LoanCard, TreasureCard
from megagem.engine import is_game_over, play_round, score_game, setup_game
from megagem.missions import MissionCard, color_counts_at_least
from megagem.players import (
    AdaptiveHeuristicAI,
    HeuristicAI,
    HyperAdaptiveAI,
    HypergeometricAI,
    RandomAI,
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
# AdaptiveHeuristicAI tests
# ----------------------------------------------------------------------------


class AdaptiveFeaturesTest(unittest.TestCase):
    def test_progress_uses_auction_deck_size(self):
        state = _empty_state()
        me = state.player_states[0]
        # 25 total auctions; 20 left → progress = 0.2
        state.auction_deck = [TreasureCard(1)] * 20
        # Make ev_remaining nonzero so cash ratios are finite.
        state.gem_deck = [GemCard(Color.BLUE)] * 5
        feats = _compute_discount_features(state, me)
        self.assertAlmostEqual(feats.progress, 0.2, places=6)

    def test_cash_ratios_use_ev_denominator(self):
        state = _empty_state(chart="A")
        me = state.player_states[0]
        opp = state.player_states[1]
        me.coins = 30
        opp.coins = 10
        state.auction_deck = [TreasureCard(1)] * 10
        state.gem_deck = [GemCard(Color.BLUE)] * 6
        # Seed the display so chart A produces nonzero per-gem values.
        for color in Color:
            state.value_display[color] = 2
        ev = _ev_remaining_auctions(state, me)
        self.assertGreater(ev, 0)
        feats = _compute_discount_features(state, me)
        self.assertAlmostEqual(feats.my_cash_ratio, 30 / ev, places=6)
        self.assertAlmostEqual(feats.avg_cash_ratio, 10 / ev, places=6)
        self.assertAlmostEqual(feats.top_cash_ratio, 10 / ev, places=6)

    def test_variance_zero_when_no_hidden_cards(self):
        state = _empty_state()
        me = state.player_states[0]
        # Empty hands and empty deck → no hidden cards.
        state.gem_deck = []
        state.auction_deck = [TreasureCard(1)]
        feats = _compute_discount_features(state, me)
        self.assertEqual(feats.variance, 0.0)

    def test_variance_scales_with_hidden_count(self):
        # Same chart, different hidden card counts → variance proportional.
        state_few = _empty_state(chart="A")
        state_few.gem_deck = [GemCard(Color.BLUE)] * 5
        state_few.auction_deck = [TreasureCard(1)]
        state_many = _empty_state(chart="A")
        state_many.gem_deck = [GemCard(Color.BLUE)] * 20
        state_many.auction_deck = [TreasureCard(1)]
        feats_few = _compute_discount_features(state_few, state_few.player_states[0])
        feats_many = _compute_discount_features(state_many, state_many.player_states[0])
        self.assertGreater(feats_many.variance, feats_few.variance)
        # 4× the hidden cards → 4× the variance proxy.
        self.assertAlmostEqual(feats_many.variance, feats_few.variance * 4.0, places=6)


class AdaptiveDiscountRateTest(unittest.TestCase):
    def test_discount_clamped_above_zero(self):
        ai = AdaptiveHeuristicAI("X", seed=0)
        # Force every penalising feature to a huge value.
        feats = _compute_discount_features(_empty_state(), _empty_state().player_states[0])
        feats.progress = 0.0
        feats.my_cash_ratio = 0.0
        feats.avg_cash_ratio = 100.0
        feats.top_cash_ratio = 100.0
        feats.variance = 100.0
        self.assertEqual(ai.discount_rate(feats), 0.0)

    def test_discount_clamped_below_one(self):
        ai = AdaptiveHeuristicAI("X", seed=0)
        feats = _compute_discount_features(_empty_state(), _empty_state().player_states[0])
        feats.progress = 100.0
        feats.my_cash_ratio = 100.0
        feats.avg_cash_ratio = 0.0
        feats.top_cash_ratio = 0.0
        feats.variance = 0.0
        self.assertEqual(ai.discount_rate(feats), 1.0)

    def test_neutral_features_in_unit_interval(self):
        ai = AdaptiveHeuristicAI("X", seed=0)
        feats = _compute_discount_features(_empty_state(), _empty_state().player_states[0])
        feats.progress = 0.5
        feats.my_cash_ratio = 0.5
        feats.avg_cash_ratio = 0.5
        feats.top_cash_ratio = 0.5
        feats.variance = 0.3
        d = ai.discount_rate(feats)
        self.assertGreater(d, 0.0)
        self.assertLess(d, 1.0)


class AdaptiveBidTest(unittest.TestCase):
    def test_treasure_bid_within_cap(self):
        ai = AdaptiveHeuristicAI("X", seed=0)
        state = _empty_state(chart="A")
        me = state.player_states[0]
        me.coins = 5
        state.revealed_gems = [GemCard(Color.BLUE), GemCard(Color.PINK)]
        state.gem_deck = [GemCard(Color.GREEN)] * 4
        state.auction_deck = [TreasureCard(1)] * 10
        bid = ai.choose_bid(state, me, TreasureCard(1))
        self.assertGreaterEqual(bid, 0)
        self.assertLessEqual(bid, 5)

    def test_invest_always_takes_token_bid(self):
        ai = AdaptiveHeuristicAI("X", seed=0)
        state = _empty_state()
        me = state.player_states[0]
        me.coins = 1  # tiny pile, but invest is free money
        state.gem_deck = [GemCard(Color.BLUE)] * 5
        state.auction_deck = [TreasureCard(1)] * 10
        bid = ai.choose_bid(state, me, InvestCard(10))
        self.assertGreaterEqual(bid, 1)

    def test_loan_skipped_when_cash_healthy(self):
        ai = AdaptiveHeuristicAI("X", seed=0)
        state = _empty_state()
        me = state.player_states[0]
        # Make my cash ratio huge so loan policy declines.
        me.coins = 9999
        state.gem_deck = [GemCard(Color.BLUE)] * 5
        state.auction_deck = [TreasureCard(1)] * 10
        bid = ai.choose_bid(state, me, LoanCard(20))
        self.assertEqual(bid, 0)


class AdaptiveHeadToHeadTest(unittest.TestCase):
    """AdaptiveHeuristicAI should comfortably beat RandomAI."""

    def test_adaptive_beats_random_majority(self):
        wins = 0
        ties = 0
        games = 60
        for seed in range(games):
            players = [
                AdaptiveHeuristicAI("Adapt", seed=seed * 13),
                RandomAI("R1", seed=seed * 13 + 1),
                RandomAI("R2", seed=seed * 13 + 2),
                RandomAI("R3", seed=seed * 13 + 3),
            ]
            state = setup_game(players, chart="A", seed=seed)
            rng = random.Random(seed)
            while not is_game_over(state):
                play_round(state, rng=rng)
            scores = score_game(state)
            adapt_score = scores[0]["total"]
            best_random = max(s["total"] for s in scores[1:])
            if adapt_score > best_random:
                wins += 1
            elif adapt_score == best_random:
                ties += 1
        win_rate = wins / games
        self.assertGreater(
            win_rate,
            0.6,
            f"Adaptive only won {wins}/{games} ({win_rate:.0%}); ties={ties}",
        )


# ----------------------------------------------------------------------------
# HypergeometricAI tests
# ----------------------------------------------------------------------------


class HyperHiddenDistributionTest(unittest.TestCase):
    def test_delta_when_no_hidden_of_color(self):
        # All 6 BLUE cards are visible somewhere → distribution is a delta
        # at the known final count regardless of how big the hidden pool is.
        state = _empty_state(chart="A")
        me = state.player_states[0]
        opp = state.player_states[1]
        state.value_display[Color.BLUE] = 4
        me.hand = [GemCard(Color.BLUE), GemCard(Color.BLUE)]
        # Hidden pool exists but cannot contain any BLUE.
        opp.hand = [GemCard(Color.GREEN)] * 3
        state.gem_deck = [GemCard(Color.GREEN)] * 5
        dist = _hyper_hidden_distribution(state, me)
        self.assertEqual(dist[Color.BLUE], {6: 1.0})
        # And per-gem value matches the chart exactly (clamped to index 5).
        per_gem = _hyper_expected_per_gem_value(state, me, "A")
        self.assertEqual(per_gem[Color.BLUE], float(value_for("A", 6)))

    def test_delta_when_hidden_pool_empty(self):
        # No opponent hand cards and no deck → no randomness anywhere.
        state = _empty_state(chart="A")
        me = state.player_states[0]
        state.value_display[Color.PINK] = 2
        me.hand = [GemCard(Color.PINK)]
        # No opp hand, no deck.
        dist = _hyper_hidden_distribution(state, me)
        self.assertEqual(dist[Color.PINK], {3: 1.0})

    def test_hand_built_hypergeometric_matches_exactly(self):
        # Construct a state with hidden_BLUE = 2, hidden_total = 4,
        # opp_hand_total = 2 → P(X=k) = C(2,k)*C(2,2-k)/C(4,2).
        state = _empty_state(chart="A")
        me = state.player_states[0]
        opp = state.player_states[1]
        # Account for 4 BLUE and 4 GREEN in the display.
        state.value_display[Color.BLUE] = 4
        state.value_display[Color.GREEN] = 4
        # Account for the other 3 colors completely (6 of each in my collection).
        # The function only cares about counts, not where they live.
        for color in (Color.PINK, Color.PURPLE, Color.YELLOW):
            me.collection_gems[color] = 6
        # Hidden pool: 2 in opp hand + 2 in deck = 4 total.
        opp.hand = [GemCard(Color.BLUE), GemCard(Color.BLUE)]  # contents irrelevant
        state.gem_deck = [GemCard(Color.BLUE), GemCard(Color.BLUE)]

        dist = _hyper_hidden_distribution(state, me)
        # known_offset for BLUE = display(4) + my_hand(0) = 4
        self.assertAlmostEqual(dist[Color.BLUE][4], 1 / 6, places=12)
        self.assertAlmostEqual(dist[Color.BLUE][5], 4 / 6, places=12)
        self.assertAlmostEqual(dist[Color.BLUE][6], 1 / 6, places=12)
        # Probabilities sum to 1.
        self.assertAlmostEqual(sum(dist[Color.BLUE].values()), 1.0, places=12)
        # GREEN is symmetric to BLUE.
        self.assertAlmostEqual(dist[Color.GREEN][5], 4 / 6, places=12)


class HyperPerGemValueTest(unittest.TestCase):
    def test_chart_e_distribution_strictly_below_point_estimate(self):
        # Setup chosen so E[X_BLUE] = 3 but the distribution is wide.
        # hidden_BLUE = 6, hidden_total = 12, opp_hand_total = 6
        # → P(X) symmetric around 3, P(X=3) = 400/924 ≈ 0.433.
        # chart E = [0, 4, 10, 18, 6, 0]; chart_E[3] = 18 (the peak).
        # E[chart_E(X)] = 10944/924 ≈ 11.844, strictly less than 18.
        state = _empty_state(chart="E")
        me = state.player_states[0]
        opp = state.player_states[1]
        opp.hand = [GemCard(Color.GREEN)] * 6  # contents irrelevant
        state.gem_deck = [GemCard(Color.GREEN)] * 6  # contents irrelevant

        per_gem = _hyper_expected_per_gem_value(state, me, "E")
        # Sanity: well below the misleading point estimate of 18.
        self.assertLess(per_gem[Color.BLUE], 18.0)
        # Exact analytic value, computed by hand above.
        self.assertAlmostEqual(per_gem[Color.BLUE], 10944 / 924, places=9)

    def test_monotonic_chart_a_matches_distribution_weighted_sum(self):
        # Sanity check on chart A using the same setup as the test above:
        # E[chart_A(X)] should equal Σ P(X=k) * chart_A(min(k,5)).
        state = _empty_state(chart="A")
        me = state.player_states[0]
        opp = state.player_states[1]
        opp.hand = [GemCard(Color.GREEN)] * 6
        state.gem_deck = [GemCard(Color.GREEN)] * 6

        dist = _hyper_hidden_distribution(state, me)
        expected_blue = sum(
            p * value_for("A", min(count, 5))
            for count, p in dist[Color.BLUE].items()
        )
        per_gem = _hyper_expected_per_gem_value(state, me, "A")
        self.assertAlmostEqual(per_gem[Color.BLUE], expected_blue, places=9)


class HyperTreasureValueTest(unittest.TestCase):
    def test_joint_mission_completion_for_two_gem_treasure(self):
        # A 2-gem (BLUE, GREEN) treasure should pick up the +5 from a
        # "1 Blue + 1 Green" pendant only when the bundle is evaluated jointly.
        state = _empty_state(chart="A")
        me = state.player_states[0]
        state.revealed_gems = [GemCard(Color.BLUE), GemCard(Color.GREEN)]
        pendant = MissionCard(
            name="Pendant: 1 Blue + 1 Green",
            coins=5,
            requirement=color_counts_at_least({Color.BLUE: 1, Color.GREEN: 1}),
            category="pendant",
        )
        state.active_missions = [pendant]
        with_mission = _hyper_treasure_value(TreasureCard(2), state, me)
        state.active_missions = []
        without_mission = _hyper_treasure_value(TreasureCard(2), state, me)
        self.assertAlmostEqual(with_mission - without_mission, 5.0, places=9)

    def test_two_same_color_treasure_doubles_per_gem_value(self):
        # Two BLUE gems on offer → gem value should be exactly 2 × per_gem[BLUE]
        # (the chart depends on the display, not the collection, so the second
        # gem's expected value is unchanged by the first).
        state = _empty_state(chart="A")
        me = state.player_states[0]
        state.revealed_gems = [GemCard(Color.BLUE), GemCard(Color.BLUE)]
        # Some hidden pool so the distribution isn't degenerate.
        state.player_states[1].hand = [GemCard(Color.GREEN)] * 2
        state.gem_deck = [GemCard(Color.GREEN)] * 4

        per_gem = _hyper_expected_per_gem_value(state, me, "A")
        gem_v = _hyper_treasure_gem_value(
            TreasureCard(2), state, me, per_gem=per_gem
        )
        self.assertAlmostEqual(gem_v, 2 * per_gem[Color.BLUE], places=9)

    def test_no_revealed_gems_returns_zero(self):
        state = _empty_state(chart="A")
        me = state.player_states[0]
        state.revealed_gems = []
        self.assertEqual(_hyper_treasure_value(TreasureCard(1), state, me), 0.0)


class HyperHeadToHeadTest(unittest.TestCase):
    """HypergeometricAI should comfortably beat RandomAI, especially on the
    non-monotonic chart E where the new estimator is most clearly an upgrade
    over the point-estimate version."""

    def _run_against_random(self, chart: str, games: int = 60) -> float:
        wins = 0
        for seed in range(games):
            players = [
                HypergeometricAI("Hyper", seed=seed * 17),
                RandomAI("R1", seed=seed * 17 + 1),
                RandomAI("R2", seed=seed * 17 + 2),
                RandomAI("R3", seed=seed * 17 + 3),
            ]
            state = setup_game(players, chart=chart, seed=seed)
            rng = random.Random(seed)
            while not is_game_over(state):
                play_round(state, rng=rng)
            scores = score_game(state)
            hyper_score = scores[0]["total"]
            best_random = max(s["total"] for s in scores[1:])
            if hyper_score > best_random:
                wins += 1
        return wins / games

    def test_beats_random_chart_a(self):
        win_rate = self._run_against_random("A")
        self.assertGreater(
            win_rate,
            0.6,
            f"Hypergeometric only won {win_rate:.0%} on chart A",
        )

    def test_beats_random_chart_e(self):
        win_rate = self._run_against_random("E")
        self.assertGreater(
            win_rate,
            0.6,
            f"Hypergeometric only won {win_rate:.0%} on chart E",
        )


# ----------------------------------------------------------------------------
# HyperAdaptiveAI tests
# ----------------------------------------------------------------------------


class HyperAdaptiveBidTest(unittest.TestCase):
    def test_treasure_bid_matches_hyper_helpers(self):
        # Strong wiring check: on a chart-E state where the point and hyper
        # estimators disagree, HyperAdaptive's choose_bid must equal the bid
        # we'd compute manually from _hyper_treasure_value and the hyper
        # discount features. This is the only test that pins the integration.
        ai = HyperAdaptiveAI("HA", seed=0)
        state = _empty_state(chart="E")
        me = state.player_states[0]
        opp = state.player_states[1]
        me.coins = 50
        opp.hand = [GemCard(Color.GREEN)] * 6
        state.gem_deck = [GemCard(Color.GREEN)] * 6
        state.revealed_gems = [GemCard(Color.BLUE)]
        state.auction_deck = [TreasureCard(1)] * 10

        bid = ai.choose_bid(state, me, TreasureCard(1))

        # Reproduce the exact arithmetic the class is supposed to do.
        value = _hyper_treasure_value(TreasureCard(1), state, me)
        feats = _hyper_compute_discount_features(state, me)
        discount = ai.discount_rate(feats)
        reserve = ai._reserve_for_future(state)
        spendable = max(0, me.coins - reserve)
        cap = me.coins  # treasure cap = coins
        expected = max(0, min(int(value * discount), spendable, cap))
        self.assertEqual(bid, expected)

    def test_chart_e_bid_differs_from_adaptive(self):
        # Looser smoke test: on the same chart-E state, HyperAdaptive and
        # AdaptiveHeuristicAI should not produce identical bids — that
        # would mean the hyper helpers had no effect.
        ai_hyper = HyperAdaptiveAI("HA", seed=0)
        ai_adapt = AdaptiveHeuristicAI("A", seed=0)
        state = _empty_state(chart="E")
        me = state.player_states[0]
        opp = state.player_states[1]
        me.coins = 50
        opp.hand = [GemCard(Color.GREEN)] * 6
        state.gem_deck = [GemCard(Color.GREEN)] * 6
        state.revealed_gems = [GemCard(Color.BLUE)]
        state.auction_deck = [TreasureCard(1)] * 10
        bid_hyper = ai_hyper.choose_bid(state, me, TreasureCard(1))
        bid_adapt = ai_adapt.choose_bid(state, me, TreasureCard(1))
        self.assertNotEqual(bid_hyper, bid_adapt)

    def test_loan_skipped_when_cash_healthy(self):
        ai = HyperAdaptiveAI("HA", seed=0)
        state = _empty_state()
        me = state.player_states[0]
        me.coins = 9999  # huge cash ratio → loan policy declines
        state.gem_deck = [GemCard(Color.BLUE)] * 5
        state.auction_deck = [TreasureCard(1)] * 10
        self.assertEqual(ai.choose_bid(state, me, LoanCard(20)), 0)

    def test_invest_always_takes_token_bid(self):
        ai = HyperAdaptiveAI("HA", seed=0)
        state = _empty_state()
        me = state.player_states[0]
        me.coins = 1
        state.gem_deck = [GemCard(Color.BLUE)] * 5
        state.auction_deck = [TreasureCard(1)] * 10
        self.assertGreaterEqual(ai.choose_bid(state, me, InvestCard(10)), 1)

    def test_treasure_bid_within_cap(self):
        ai = HyperAdaptiveAI("HA", seed=0)
        state = _empty_state(chart="A")
        me = state.player_states[0]
        me.coins = 5
        state.revealed_gems = [GemCard(Color.BLUE), GemCard(Color.PINK)]
        state.gem_deck = [GemCard(Color.GREEN)] * 4
        state.auction_deck = [TreasureCard(1)] * 10
        bid = ai.choose_bid(state, me, TreasureCard(1))
        self.assertGreaterEqual(bid, 0)
        self.assertLessEqual(bid, 5)


class HyperDiscountFeaturesTest(unittest.TestCase):
    def test_ev_remaining_uses_hyper_avg(self):
        # Same game state — the only difference between point-estimate EV
        # and hyper EV is which avg-per-gem helper is used. On chart E with
        # a wide hidden pool the two should disagree.
        state = _empty_state(chart="E")
        me = state.player_states[0]
        opp = state.player_states[1]
        opp.hand = [GemCard(Color.GREEN)] * 6
        state.gem_deck = [GemCard(Color.GREEN)] * 6
        state.auction_deck = [TreasureCard(1)] * 10

        ev_point = _ev_remaining_auctions(state, me)
        ev_hyper = _hyper_ev_remaining_auctions(state, me)
        self.assertNotAlmostEqual(ev_point, ev_hyper, places=3)
        self.assertGreater(ev_point, ev_hyper)  # point overestimates on E

    def test_features_match_shape_of_point_version(self):
        # Hyper feature set should produce a fully populated DiscountFeatures
        # with the same five fields. We don't pin numerical values here —
        # just shape — because the EV denominator differs by construction.
        state = _empty_state(chart="A")
        me = state.player_states[0]
        state.auction_deck = [TreasureCard(1)] * 20
        state.gem_deck = [GemCard(Color.BLUE)] * 5
        feats = _hyper_compute_discount_features(state, me)
        for attr in (
            "progress",
            "my_cash_ratio",
            "avg_cash_ratio",
            "top_cash_ratio",
            "variance",
        ):
            self.assertTrue(hasattr(feats, attr))


class HyperAdaptiveHeadToHeadTest(unittest.TestCase):
    """HyperAdaptiveAI should clear the same head-to-head bar on every chart."""

    def _run(self, chart: str, games: int = 60) -> float:
        wins = 0
        for seed in range(games):
            players = [
                HyperAdaptiveAI("HA", seed=seed * 19),
                RandomAI("R1", seed=seed * 19 + 1),
                RandomAI("R2", seed=seed * 19 + 2),
                RandomAI("R3", seed=seed * 19 + 3),
            ]
            state = setup_game(players, chart=chart, seed=seed)
            rng = random.Random(seed)
            while not is_game_over(state):
                play_round(state, rng=rng)
            scores = score_game(state)
            if scores[0]["total"] > max(s["total"] for s in scores[1:]):
                wins += 1
        return wins / games

    def test_beats_random_chart_a(self):
        win_rate = self._run("A")
        self.assertGreater(
            win_rate, 0.6, f"HyperAdaptive only won {win_rate:.0%} on chart A"
        )

    def test_beats_random_chart_e(self):
        win_rate = self._run("E")
        self.assertGreater(
            win_rate, 0.6, f"HyperAdaptive only won {win_rate:.0%} on chart E"
        )


if __name__ == "__main__":
    unittest.main()
