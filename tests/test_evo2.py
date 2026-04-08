"""Unit tests for ``megagem.players_evo2``.

Kept in its own file (rather than appended to ``test_heuristic.py``) so the
new AI's coverage is easy to find and run independently:

    python -m unittest tests.test_evo2

The cases mirror the structure used by the existing AI test classes.
"""

from __future__ import annotations

import math
import random
import unittest
from collections import Counter
from itertools import permutations

from megagem.cards import Color, GemCard, InvestCard, LoanCard, TreasureCard
from megagem.engine import is_game_over, play_round, score_game, setup_game
from megagem.missions import (
    MissionCard,
    at_least_n_distinct_colors,
    color_counts_at_least,
)
from megagem.players import (
    RandomAI,
    _hyper_expected_per_gem_value,
)
from megagem.players_evo2 import (
    Evo2AI,
    _IMPOSSIBLE_DISTANCE,
    _compute_evo2_features,
    _expected_rounds_remaining,
    _InvestModel,
    _LoanModel,
    _min_extra_gems_to_satisfy,
    _mission_probability_delta,
    _p_player_wins_mission,
    _per_color_value_stats,
    _treasure_value_stats,
    _TreasureModel,
)
from megagem.state import GameState, PlayerState


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _empty_state(chart: str = "A", num_players: int = 2) -> GameState:
    states = [PlayerState(name=f"P{i}", coins=20) for i in range(num_players)]
    return GameState(player_states=states, players=[], value_chart=chart)


def _two_blue_pendant() -> MissionCard:
    return MissionCard(
        name="Pendant: 2 Blue",
        coins=5,
        requirement=color_counts_at_least({Color.BLUE: 2}),
        category="pendant",
    )


def _blue_green_pink_crown() -> MissionCard:
    return MissionCard(
        name="Crown: 1 Blue + 1 Green + 1 Pink",
        coins=10,
        requirement=color_counts_at_least(
            {Color.BLUE: 1, Color.GREEN: 1, Color.PINK: 1}
        ),
        category="crown",
    )


def _four_distinct_shield() -> MissionCard:
    return MissionCard(
        name="Shield: 4 different colors",
        coins=10,
        requirement=at_least_n_distinct_colors(4),
        category="shield",
    )


# ----------------------------------------------------------------------------
# _expected_rounds_remaining
# ----------------------------------------------------------------------------


class ExpectedRoundsRemainingTest(unittest.TestCase):
    def _state(self, t1: int, t2: int, nt: int, gems: int) -> GameState:
        s = _empty_state()
        s.auction_deck = (
            [TreasureCard(1)] * t1
            + [TreasureCard(2)] * t2
            + [LoanCard(10)] * nt
        )
        # Only the LENGTH of these lists matters for the helper.
        s.gem_deck = [GemCard(Color.BLUE)] * gems
        s.revealed_gems = []
        return s

    def test_single_treasure_one_round(self):
        s = self._state(t1=1, t2=0, nt=0, gems=10)
        self.assertAlmostEqual(_expected_rounds_remaining(s), 1.0, places=6)

    def test_two_one_gem_treasures_drain_supply_of_one(self):
        # First card is always a 1-gem treasure → drains the supply → 1 round.
        s = self._state(t1=2, t2=0, nt=0, gems=1)
        self.assertAlmostEqual(_expected_rounds_remaining(s), 1.0, places=6)

    def test_no_treasures_runs_to_deck_end(self):
        s = self._state(t1=0, t2=0, nt=2, gems=10)
        self.assertAlmostEqual(_expected_rounds_remaining(s), 2.0, places=6)

    def test_no_auction_deck_zero(self):
        s = self._state(t1=0, t2=0, nt=0, gems=10)
        self.assertEqual(_expected_rounds_remaining(s), 0.0)

    def test_no_gems_zero(self):
        s = self._state(t1=2, t2=0, nt=0, gems=0)
        self.assertEqual(_expected_rounds_remaining(s), 0.0)

    def test_brute_force_cross_check(self):
        """A=4, T1=2, T2=1, NT=1, G=2 — closed form must match enumeration."""
        s = self._state(t1=2, t2=1, nt=1, gems=2)
        closed = _expected_rounds_remaining(s)

        # Each card contributes its gem count when drawn. Game ends after the
        # round in which cumulative gems consumed reaches G or all cards drawn.
        gems_per_card = [1, 1, 2, 0]
        total = 0.0
        count = 0
        for perm in permutations(gems_per_card):
            consumed = 0
            rounds = 0
            for g in perm:
                rounds += 1
                consumed += g
                if consumed >= 2:
                    break
            total += rounds
            count += 1
        brute = total / count
        self.assertAlmostEqual(closed, brute, places=6)
        # Sanity: known value is 50/24.
        self.assertAlmostEqual(closed, 50 / 24, places=6)


# ----------------------------------------------------------------------------
# _per_color_value_stats / _treasure_value_stats
# ----------------------------------------------------------------------------


class PerColorValueStatsTest(unittest.TestCase):
    def test_mean_matches_existing_helper(self):
        s = _empty_state(chart="E")
        me = s.player_states[0]
        # Give the existing helper something nontrivial: opponent has hidden
        # cards, the deck has hidden cards, and a couple of colors already
        # show on the display.
        s.player_states[1].hand = [
            GemCard(Color.BLUE),
            GemCard(Color.BLUE),
            GemCard(Color.GREEN),
        ]
        s.gem_deck = [GemCard(Color.YELLOW)] * 5
        s.value_display[Color.BLUE] = 1
        s.value_display[Color.PINK] = 2

        ours = _per_color_value_stats(s, me, "E")
        theirs = _hyper_expected_per_gem_value(s, me, "E")
        self.assertEqual(set(ours.keys()), set(theirs.keys()))
        for color, mean in theirs.items():
            self.assertAlmostEqual(ours[color][0], mean, places=9)

    def test_variance_zero_when_distribution_degenerate(self):
        s = _empty_state(chart="A")
        me = s.player_states[0]
        # No hidden cards anywhere → every color's distribution is a point
        # mass on the current display, so variance should be exactly zero.
        s.value_display[Color.BLUE] = 2
        s.value_display[Color.GREEN] = 1
        stats = _per_color_value_stats(s, me, "A")
        for _, var in stats.values():
            self.assertEqual(var, 0.0)

    def test_treasure_variance_positive_when_uncertainty_present(self):
        s = _empty_state(chart="A")
        me = s.player_states[0]
        # Force uncertainty: opponent has hidden cards.
        s.player_states[1].hand = [GemCard(Color.BLUE)] * 3
        s.gem_deck = [GemCard(Color.BLUE)] * 3
        s.revealed_gems = [GemCard(Color.BLUE)]
        ev, std = _treasure_value_stats(TreasureCard(1), s, me)
        self.assertGreater(ev, 0.0)
        self.assertGreater(std, 0.0)

    def test_treasure_value_empty_revealed_returns_zero(self):
        s = _empty_state(chart="A")
        me = s.player_states[0]
        s.revealed_gems = []
        ev, std = _treasure_value_stats(TreasureCard(1), s, me)
        self.assertEqual(ev, 0.0)
        self.assertEqual(std, 0.0)


# ----------------------------------------------------------------------------
# _min_extra_gems_to_satisfy
# ----------------------------------------------------------------------------


class MinExtraGemsTest(unittest.TestCase):
    def test_two_blue_distance(self):
        m = _two_blue_pendant()
        self.assertEqual(_min_extra_gems_to_satisfy(Counter(), m), 2)
        self.assertEqual(
            _min_extra_gems_to_satisfy(Counter({Color.BLUE: 1}), m), 1
        )
        self.assertEqual(
            _min_extra_gems_to_satisfy(Counter({Color.BLUE: 2}), m), 0
        )
        self.assertEqual(
            _min_extra_gems_to_satisfy(Counter({Color.BLUE: 5}), m), 0
        )

    def test_blue_green_pink_distance(self):
        m = _blue_green_pink_crown()
        self.assertEqual(_min_extra_gems_to_satisfy(Counter(), m), 3)
        self.assertEqual(
            _min_extra_gems_to_satisfy(Counter({Color.BLUE: 1}), m), 2
        )
        self.assertEqual(
            _min_extra_gems_to_satisfy(
                Counter({Color.BLUE: 1, Color.GREEN: 1}), m
            ),
            1,
        )
        self.assertEqual(
            _min_extra_gems_to_satisfy(
                Counter({Color.BLUE: 1, Color.GREEN: 1, Color.PINK: 1}), m
            ),
            0,
        )

    def test_four_distinct_distance(self):
        m = _four_distinct_shield()
        self.assertEqual(_min_extra_gems_to_satisfy(Counter(), m), 4)
        # Three distinct colors → need one new color → distance 1.
        self.assertEqual(
            _min_extra_gems_to_satisfy(
                Counter(
                    {Color.BLUE: 1, Color.GREEN: 1, Color.PINK: 1}
                ),
                m,
            ),
            1,
        )
        # Pile of one color doesn't help.
        self.assertEqual(
            _min_extra_gems_to_satisfy(Counter({Color.BLUE: 5}), m), 3
        )


# ----------------------------------------------------------------------------
# _p_player_wins_mission
# ----------------------------------------------------------------------------


class PlayerWinsMissionTest(unittest.TestCase):
    def test_already_satisfied_player_gets_one(self):
        s = _empty_state(num_players=3)
        m = _two_blue_pendant()
        s.player_states[1].collection_gems = Counter({Color.BLUE: 2})
        self.assertEqual(_p_player_wins_mission(0, s, m), 0.0)
        self.assertEqual(_p_player_wins_mission(1, s, m), 1.0)
        self.assertEqual(_p_player_wins_mission(2, s, m), 0.0)

    def test_lowest_seat_satisfied_wins_tie(self):
        # Two players already satisfy the mission — engine tie-break is the
        # lowest seat index, so seat 0 should get 1.0 and seat 2 gets 0.0.
        s = _empty_state(num_players=3)
        m = _two_blue_pendant()
        s.player_states[0].collection_gems = Counter({Color.BLUE: 2})
        s.player_states[2].collection_gems = Counter({Color.BLUE: 3})
        self.assertEqual(_p_player_wins_mission(0, s, m), 1.0)
        self.assertEqual(_p_player_wins_mission(2, s, m), 0.0)

    def test_normalized_to_one(self):
        s = _empty_state(num_players=3)
        m = _two_blue_pendant()
        # All players have a chance — sum of probs across players = 1.
        s.player_states[0].coins = 10
        s.player_states[1].coins = 15
        s.player_states[2].coins = 20
        s.player_states[0].collection_gems = Counter({Color.BLUE: 1})
        s.player_states[1].collection_gems = Counter({Color.BLUE: 1})
        s.player_states[2].collection_gems = Counter({Color.BLUE: 1})
        ps = [_p_player_wins_mission(i, s, m) for i in range(3)]
        self.assertAlmostEqual(sum(ps), 1.0, places=9)
        for p in ps:
            self.assertGreater(p, 0.0)

    def test_higher_coins_more_likely(self):
        s = _empty_state(num_players=2)
        m = _two_blue_pendant()
        # Equal distance, different coins.
        s.player_states[0].coins = 5
        s.player_states[1].coins = 30
        s.player_states[0].collection_gems = Counter({Color.BLUE: 1})
        s.player_states[1].collection_gems = Counter({Color.BLUE: 1})
        p0 = _p_player_wins_mission(0, s, m)
        p1 = _p_player_wins_mission(1, s, m)
        self.assertGreater(p1, p0)

    def test_closer_distance_more_likely(self):
        s = _empty_state(num_players=2)
        m = _two_blue_pendant()
        # Equal coins, different distance.
        s.player_states[0].coins = 20
        s.player_states[1].coins = 20
        s.player_states[0].collection_gems = Counter({Color.BLUE: 1})  # d=1
        s.player_states[1].collection_gems = Counter()  # d=2
        p0 = _p_player_wins_mission(0, s, m)
        p1 = _p_player_wins_mission(1, s, m)
        self.assertGreater(p0, p1)

    def test_holding_overrides_dont_mutate_state(self):
        s = _empty_state(num_players=2)
        m = _two_blue_pendant()
        s.player_states[0].collection_gems = Counter({Color.BLUE: 1})
        s.player_states[1].collection_gems = Counter({Color.BLUE: 1})
        # Override that completes seat 0.
        before = dict(s.player_states[0].collection_gems)
        p = _p_player_wins_mission(
            0, s, m, holding_overrides={0: Counter({Color.BLUE: 1})}
        )
        after = dict(s.player_states[0].collection_gems)
        self.assertEqual(p, 1.0)
        self.assertEqual(before, after)


# ----------------------------------------------------------------------------
# _mission_probability_delta
# ----------------------------------------------------------------------------


class MissionProbabilityDeltaTest(unittest.TestCase):
    def test_positive_when_auction_completes_my_mission(self):
        s = _empty_state(num_players=2)
        me = s.player_states[0]
        me.collection_gems = Counter({Color.BLUE: 1})
        s.player_states[1].collection_gems = Counter()
        s.active_missions = [_two_blue_pendant()]
        s.revealed_gems = [GemCard(Color.BLUE)]
        delta = _mission_probability_delta(TreasureCard(1), s, me)
        self.assertGreater(delta, 0.0)

    def test_zero_when_no_active_missions(self):
        s = _empty_state(num_players=2)
        me = s.player_states[0]
        s.active_missions = []
        s.revealed_gems = [GemCard(Color.BLUE)]
        self.assertEqual(
            _mission_probability_delta(TreasureCard(1), s, me), 0.0
        )

    def test_zero_when_no_gems_revealed(self):
        s = _empty_state(num_players=2)
        me = s.player_states[0]
        s.active_missions = [_two_blue_pendant()]
        s.revealed_gems = []
        self.assertEqual(
            _mission_probability_delta(TreasureCard(1), s, me), 0.0
        )

    def test_irrelevant_color_delta_near_zero(self):
        s = _empty_state(num_players=2)
        me = s.player_states[0]
        me.collection_gems = Counter({Color.BLUE: 1})
        s.player_states[1].collection_gems = Counter()
        s.active_missions = [_two_blue_pendant()]
        # Pink doesn't help anyone with this Blue mission.
        s.revealed_gems = [GemCard(Color.PINK)]
        delta = _mission_probability_delta(TreasureCard(1), s, me)
        self.assertAlmostEqual(delta, 0.0, places=6)


# ----------------------------------------------------------------------------
# Treasure EV additivity
# ----------------------------------------------------------------------------


class TreasureValueAdditivityTest(unittest.TestCase):
    def test_mission_delta_flows_into_ev(self):
        from megagem.players import (
            _hyper_expected_per_gem_value,
            _mission_completion_bonus,
            _mission_progress_bonus,
        )

        s = _empty_state(chart="A", num_players=2)
        me = s.player_states[0]
        me.collection_gems = Counter({Color.BLUE: 1})
        s.player_states[1].collection_gems = Counter()
        s.active_missions = [_two_blue_pendant()]
        s.revealed_gems = [GemCard(Color.BLUE)]

        ev, _ = _treasure_value_stats(TreasureCard(1), s, me)

        per_gem = _hyper_expected_per_gem_value(s, me, "A")
        gem_v = per_gem[Color.BLUE]
        extra = Counter({Color.BLUE: 1})
        hard = _mission_completion_bonus(me, s.active_missions, extra)
        soft = _mission_progress_bonus(me, s.active_missions, extra)
        delta = _mission_probability_delta(TreasureCard(1), s, me)

        self.assertAlmostEqual(ev, gem_v + hard + soft + delta, places=6)


# ----------------------------------------------------------------------------
# Evo2AI.from_weights round-trip
# ----------------------------------------------------------------------------


class FromWeightsTest(unittest.TestCase):
    def test_round_trip(self):
        weights = [float(i) * 0.1 - 1.0 for i in range(Evo2AI.NUM_WEIGHTS)]
        ai = Evo2AI.from_weights("Test", weights)

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
            ),
            tuple(weights[0:7]),
        )

        i = ai.invest_model
        self.assertEqual(
            (i.bias, i.w_rounds, i.w_my, i.w_avg, i.w_top, i.w_amount),
            tuple(weights[7:13]),
        )

        l = ai.loan_model
        self.assertEqual(
            (l.bias, l.w_rounds, l.w_my, l.w_avg, l.w_top, l.w_amount),
            tuple(weights[13:19]),
        )

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            Evo2AI.from_weights("Bad", [0.0] * 18)


# ----------------------------------------------------------------------------
# choose_bid sanity: no static reserve gating us off coins
# ----------------------------------------------------------------------------


class NoReserveSanityTest(unittest.TestCase):
    def test_full_pile_available_when_treasure_is_juicy(self):
        # The treasure head outputs the bid in coins directly. Pin the model
        # to a bias far above any plausible cap and assert the AI is willing
        # to commit its full coin pile — no static reserve gates it off.
        s = _empty_state(chart="A", num_players=2)
        me = s.player_states[0]
        me.coins = 25
        s.revealed_gems = [GemCard(Color.BLUE), GemCard(Color.GREEN)]
        s.value_display[Color.BLUE] = 2
        s.value_display[Color.GREEN] = 2
        s.auction_deck = [TreasureCard(2)] * 10
        s.gem_deck = [GemCard(Color.BLUE)] * 5

        ai = Evo2AI(
            "T",
            treasure=_TreasureModel(
                bias=999.0,
                w_rounds=0.0,
                w_my=0.0,
                w_avg=0.0,
                w_top=0.0,
                w_ev=0.0,
                w_std=0.0,
            ),
        )
        bid = ai.choose_bid(s, me, TreasureCard(2))
        # Raw bid saturates at the legal cap (= my coins for a treasure).
        self.assertEqual(bid, 25)

    def test_treasure_bid_matches_linear_model(self):
        # End-to-end check that the bid equals int(linear_model(features))
        # clamped to [0, cap]. No EV-multiplication step in the path.
        s = _empty_state(chart="A", num_players=2)
        me = s.player_states[0]
        me.coins = 25
        s.revealed_gems = [GemCard(Color.BLUE)]
        s.value_display[Color.BLUE] = 1
        s.auction_deck = [TreasureCard(1)] * 5
        s.gem_deck = [GemCard(Color.BLUE)] * 3

        model = _TreasureModel(
            bias=2.0,
            w_rounds=0.0,
            w_my=0.20,
            w_avg=0.0,
            w_top=0.0,
            w_ev=0.50,
            w_std=0.0,
        )
        ai = Evo2AI("T", treasure=model)
        ev, std = _treasure_value_stats(TreasureCard(1), s, me)
        f = _compute_evo2_features(s, me)
        expected_raw = model.bid(f, ev, std)
        expected = max(0, min(int(expected_raw), 25))
        self.assertEqual(ai.choose_bid(s, me, TreasureCard(1)), expected)

    def test_invest_token_bid_when_head_zero(self):
        # Invest cards are free money — Evo2 should still drop a token bid
        # of 1 even when the head outputs zero.
        s = _empty_state(chart="A", num_players=2)
        me = s.player_states[0]
        me.coins = 10
        s.auction_deck = [InvestCard(5)]

        ai = Evo2AI(
            "T",
            invest=_InvestModel(
                bias=0.0,
                w_rounds=0.0,
                w_my=0.0,
                w_avg=0.0,
                w_top=0.0,
                w_amount=0.0,
            ),
        )
        bid = ai.choose_bid(s, me, InvestCard(5))
        self.assertEqual(bid, 1)


# ----------------------------------------------------------------------------
# Head-to-head smoke test
# ----------------------------------------------------------------------------


class Evo2HeadToHeadTest(unittest.TestCase):
    """Default-weights Evo2AI should comfortably beat 3× RandomAI.

    Mirrors the bar set by ``HeadToHeadTest`` for ``HeuristicAI``: any
    AI worth shipping should clear 60% against random over a small but
    statistically informative sample.
    """

    def test_evo2_beats_random_majority(self):
        wins = 0
        games = 60
        for seed in range(games):
            players = [
                Evo2AI("E", seed=seed * 11),
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
            f"Evo2 only won {wins}/{games} ({win_rate:.0%})",
        )


if __name__ == "__main__":
    unittest.main()
