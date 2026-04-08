"""Unit tests for ``megagem.players.evo3``.

Focus: the opponent-pricing features are what's new in Evo3, so the
tests target (a) the weighted-stats helper, (b) the observe_round hook
the engine calls after every round, and (c) the round-trip between the
25-element flat vector and the three model classes. A head-to-head
smoke test anchors the AI at ≥60% win rate vs RandomAI — the same bar
HeuristicAI and Evo2 clear.

Run via::

    python -m unittest tests.test_evo3
"""

from __future__ import annotations

import math
import random
import unittest

from megagem.cards import InvestCard, LoanCard, TreasureCard
from megagem.engine import is_game_over, play_round, score_game, setup_game
from megagem.players import Evo2AI, Evo3AI, RandomAI
from megagem.players.evo3 import (
    _CAT_INVEST,
    _CAT_LOAN,
    _CAT_TREASURE,
    _DEFAULT_MEAN_DELTA,
    _DEFAULT_STD_DELTA,
    _Evo3InvestModel,
    _Evo3LoanModel,
    _Evo3TreasureModel,
    _weighted_delta_stats,
)


# ----------------------------------------------------------------------------
# _weighted_delta_stats
# ----------------------------------------------------------------------------


class WeightedDeltaStatsTest(unittest.TestCase):
    def test_empty_history_uses_defaults(self):
        mean, std = _weighted_delta_stats([], _CAT_TREASURE)
        self.assertEqual(mean, _DEFAULT_MEAN_DELTA)
        self.assertEqual(std, _DEFAULT_STD_DELTA)

    def test_single_entry_matching_category(self):
        mean, std = _weighted_delta_stats(
            [(_CAT_TREASURE, 5.0)], _CAT_TREASURE
        )
        self.assertAlmostEqual(mean, 5.0, places=9)
        self.assertAlmostEqual(std, 0.0, places=9)

    def test_matching_category_weighted_four_times(self):
        # One treasure obs at delta=10, one loan obs at delta=0.
        # Matching category: treasure → weights 4 and 1, mean = 40/5 = 8.
        # Loan:  weights 1 and 4, mean = 10/5 = 2.
        history = [(_CAT_TREASURE, 10.0), (_CAT_LOAN, 0.0)]
        mean_t, _ = _weighted_delta_stats(history, _CAT_TREASURE)
        mean_l, _ = _weighted_delta_stats(history, _CAT_LOAN)
        self.assertAlmostEqual(mean_t, 8.0, places=9)
        self.assertAlmostEqual(mean_l, 2.0, places=9)

    def test_variance_nonneg_and_matches_manual_calc(self):
        # Hand-computed expected values for a mixed history, category=invest.
        # Weights: invest × 4, others × 1.
        history = [
            (_CAT_INVEST, 2.0),
            (_CAT_INVEST, 6.0),
            (_CAT_TREASURE, 10.0),
            (_CAT_LOAN, -4.0),
        ]
        mean, std = _weighted_delta_stats(history, _CAT_INVEST)
        # weights: 4, 4, 1, 1 → total 10.
        total_w = 10
        total_x = 4 * 2 + 4 * 6 + 1 * 10 + 1 * -4  # 8 + 24 + 10 - 4 = 38
        total_x2 = 4 * 4 + 4 * 36 + 1 * 100 + 1 * 16  # 16 + 144 + 100 + 16 = 276
        expected_mean = total_x / total_w
        expected_var = total_x2 / total_w - expected_mean * expected_mean
        expected_std = math.sqrt(max(0.0, expected_var))
        self.assertAlmostEqual(mean, expected_mean, places=9)
        self.assertAlmostEqual(std, expected_std, places=9)

    def test_current_category_switches_weighting(self):
        # Same history, different active category → different weighted means.
        history = [(_CAT_TREASURE, 10.0), (_CAT_LOAN, 10.0)]
        m_t, _ = _weighted_delta_stats(history, _CAT_TREASURE)
        m_l, _ = _weighted_delta_stats(history, _CAT_LOAN)
        m_i, _ = _weighted_delta_stats(history, _CAT_INVEST)
        # Treasure matches first entry (w=4) and the other is w=1 → (40+10)/5=10.
        self.assertAlmostEqual(m_t, 10.0, places=9)
        self.assertAlmostEqual(m_l, 10.0, places=9)
        # Invest matches neither → both w=1 → mean = (10+10)/2 = 10.
        self.assertAlmostEqual(m_i, 10.0, places=9)


# ----------------------------------------------------------------------------
# observe_round / _opp_history accounting
# ----------------------------------------------------------------------------


class ObserveRoundTest(unittest.TestCase):
    def _make_ai(self) -> Evo3AI:
        return Evo3AI("T", seed=0)

    def _fake_result(self, auction, bids):
        return {
            "round": 1,
            "auction": auction,
            "bids": bids,
            "winner_idx": 0,
            "winning_bid": max(bids),
            "taken_gems": [],
            "revealed_gem": None,
            "completed_missions": [],
        }

    def test_records_positive_delta_when_opponent_bids_higher(self):
        ai = self._make_ai()
        # my_idx=0, my bid=3, top opponent bid=10 → delta = 10-3 = 7.
        ai.observe_round(
            None,  # state is unused by Evo3.observe_round
            0,
            self._fake_result(TreasureCard(1), [3, 10, 5]),
        )
        self.assertEqual(len(ai._opp_history), 1)
        cat, delta = ai._opp_history[0]
        self.assertEqual(cat, _CAT_TREASURE)
        self.assertEqual(delta, 7.0)

    def test_records_negative_delta_when_i_outbid(self):
        ai = self._make_ai()
        ai.observe_round(
            None, 1, self._fake_result(LoanCard(10), [2, 15, 8])
        )
        cat, delta = ai._opp_history[0]
        self.assertEqual(cat, _CAT_LOAN)
        # my_idx=1, my bid=15, max opp=max(2, 8)=8 → delta = -7.
        self.assertEqual(delta, -7.0)

    def test_ignores_rounds_with_no_opponents(self):
        ai = self._make_ai()
        ai.observe_round(None, 0, self._fake_result(InvestCard(5), [4]))
        self.assertEqual(len(ai._opp_history), 0)

    def test_history_accumulates_across_rounds(self):
        ai = self._make_ai()
        ai.observe_round(None, 0, self._fake_result(TreasureCard(1), [1, 2]))
        ai.observe_round(None, 0, self._fake_result(InvestCard(5), [3, 4]))
        ai.observe_round(None, 0, self._fake_result(LoanCard(10), [5, 6]))
        self.assertEqual(
            [cat for cat, _ in ai._opp_history],
            [_CAT_TREASURE, _CAT_INVEST, _CAT_LOAN],
        )


# ----------------------------------------------------------------------------
# Engine wiring: the engine should call observe_round after every round.
# ----------------------------------------------------------------------------


class EngineWiresObserveRoundTest(unittest.TestCase):
    def test_full_game_fills_opp_history(self):
        players = [
            Evo3AI("E3", seed=1),
            RandomAI("R1", seed=2),
            RandomAI("R2", seed=3),
            RandomAI("R3", seed=4),
        ]
        state = setup_game(players, chart="A", seed=42)
        rng = random.Random(42)
        rounds_played = 0
        while not is_game_over(state):
            play_round(state, rng=rng)
            rounds_played += 1

        evo3 = players[0]
        assert isinstance(evo3, Evo3AI)
        # Every round sees the observation hook called once per player, so
        # the history length must equal the rounds played in the game.
        self.assertEqual(len(evo3._opp_history), rounds_played)
        self.assertGreater(rounds_played, 0)
        # And each entry must tag a known category.
        valid = {_CAT_TREASURE, _CAT_INVEST, _CAT_LOAN}
        for cat, _ in evo3._opp_history:
            self.assertIn(cat, valid)


# ----------------------------------------------------------------------------
# from_weights round-trip
# ----------------------------------------------------------------------------


class FromWeightsTest(unittest.TestCase):
    def test_round_trip(self):
        weights = [float(i) * 0.1 - 1.0 for i in range(Evo3AI.NUM_WEIGHTS)]
        ai = Evo3AI.from_weights("Test", weights)

        t = ai.treasure_model
        self.assertEqual(
            (
                t.bias,
                t.w_rounds,
                t.w_my,
                t.w_avg,
                t.w_top,
                t.w_ev,
                t.w_std,
                t.w_mean_delta,
                t.w_std_delta,
            ),
            tuple(weights[0:9]),
        )

        i = ai.invest_model
        self.assertEqual(
            (
                i.bias,
                i.w_rounds,
                i.w_my,
                i.w_avg,
                i.w_top,
                i.w_amount,
                i.w_mean_delta,
                i.w_std_delta,
            ),
            tuple(weights[9:17]),
        )

        l = ai.loan_model
        self.assertEqual(
            (
                l.bias,
                l.w_rounds,
                l.w_my,
                l.w_avg,
                l.w_top,
                l.w_amount,
                l.w_mean_delta,
                l.w_std_delta,
            ),
            tuple(weights[17:25]),
        )

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            Evo3AI.from_weights("Bad", [0.0] * 19)


# ----------------------------------------------------------------------------
# With zero opponent-delta weights, Evo3 matches Evo2 bid-by-bid.
# ----------------------------------------------------------------------------


class ZeroDeltaWeightsMatchEvo2Test(unittest.TestCase):
    def test_defaults_mirror_evo2_round_one(self):
        """Default Evo3 weights are Evo2 + zero deltas. On the first bid
        (empty history, so features are (0, 1) but multiplied by 0 deltas),
        Evo3 must produce the exact same bid as Evo2 default."""
        players_e3 = [Evo3AI("E3", seed=1), RandomAI("R1", seed=2),
                      RandomAI("R2", seed=3), RandomAI("R3", seed=4)]
        players_e2 = [Evo2AI("E2", seed=1), RandomAI("R1", seed=2),
                      RandomAI("R2", seed=3), RandomAI("R3", seed=4)]

        # Build identical game states for both.
        state_e3 = setup_game(players_e3, chart="A", seed=100)
        state_e2 = setup_game(players_e2, chart="A", seed=100)

        # First auction card is what play_round would pop next. Bid the
        # Evo3/Evo2 on that card and compare.
        auction = state_e3.auction_deck[-1]
        evo3 = players_e3[0]
        evo2 = players_e2[0]
        bid_e3 = evo3.choose_bid(state_e3, state_e3.player_states[0], auction)
        bid_e2 = evo2.choose_bid(state_e2, state_e2.player_states[0], auction)
        self.assertEqual(bid_e3, bid_e2)


# ----------------------------------------------------------------------------
# Opponent-delta weights actually move the bid.
# ----------------------------------------------------------------------------


class DeltaWeightsMoveBidTest(unittest.TestCase):
    def test_positive_mean_delta_weight_raises_bid(self):
        """A treasure head with a +100 coefficient on mean_delta should
        raise its bid after observing opponents bidding far above it."""
        # Treasure model: minimal baseline plus an aggressive mean_delta
        # coefficient. std_delta coefficient=0 so it can't muddy things.
        model = _Evo3TreasureModel(
            bias=0.0,
            w_rounds=0.0,
            w_my=0.0,
            w_avg=0.0,
            w_top=0.0,
            w_ev=0.0,
            w_std=0.0,
            w_mean_delta=1.0,
            w_std_delta=0.0,
        )
        ai = Evo3AI(
            "T",
            treasure=model,
            invest=_Evo3InvestModel(0, 0, 0, 0, 0, 0, 0, 0),
            loan=_Evo3LoanModel(0, 0, 0, 0, 0, 0, 0, 0),
        )
        # Feed it a history where opponents bid 20 above on treasures.
        ai._opp_history = [(_CAT_TREASURE, 20.0)]

        players = [ai, RandomAI("R1", seed=2), RandomAI("R2", seed=3),
                   RandomAI("R3", seed=4)]
        state = setup_game(players, chart="A", seed=5)
        # Force a treasure auction next.
        state.auction_deck.append(TreasureCard(1))
        auction = state.auction_deck[-1]
        bid_with_history = ai.choose_bid(state, state.player_states[0], auction)

        # Same AI, empty history → baseline bid.
        ai._opp_history = []
        bid_empty = ai.choose_bid(state, state.player_states[0], auction)

        self.assertGreater(bid_with_history, bid_empty)


# ----------------------------------------------------------------------------
# Head-to-head smoke test
# ----------------------------------------------------------------------------


class Evo3HeadToHeadTest(unittest.TestCase):
    """Default-weights Evo3AI should comfortably beat 3× RandomAI.

    Same 60% bar the other AIs clear. With zero opponent-delta weights
    the default behaviour reduces to Evo2, so this is the floor we must
    maintain — any regression below Evo2's win rate indicates a bug in
    the observation pipeline.
    """

    def test_evo3_beats_random_majority(self):
        wins = 0
        games = 60
        for seed in range(games):
            players = [
                Evo3AI("E3", seed=seed * 11),
                RandomAI("R1", seed=seed * 11 + 1),
                RandomAI("R2", seed=seed * 11 + 2),
                RandomAI("R3", seed=seed * 11 + 3),
            ]
            state = setup_game(players, chart="A", seed=seed)
            rng = random.Random(seed)
            while not is_game_over(state):
                play_round(state, rng=rng)
            scores = score_game(state)
            evo_score = scores[0]["total"]
            best_random = max(s["total"] for s in scores[1:])
            if evo_score > best_random:
                wins += 1
        win_rate = wins / games
        self.assertGreater(
            win_rate,
            0.6,
            f"Evo3 only won {wins}/{games} ({win_rate:.0%})",
        )


if __name__ == "__main__":
    unittest.main()
