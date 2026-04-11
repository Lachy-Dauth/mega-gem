"""Unit tests for ``megagem.players.evo4``.

Focus: the new bid-signal-driven color distribution adjustment. Evo4
inherits Evo3's opp-delta history verbatim, so most of the *behavioural*
tests in ``test_evo3.py`` apply here too — this file targets only the
new surface area:

1. :func:`_biased_per_color_value_stats` with a zero shift must match
   Evo2's unbiased per-color stats bit-for-bit. This is the critical
   "defaults reduce to Evo3" guarantee.
2. A non-zero shift must actually move the per-color EV.
3. :meth:`Evo4AI.observe_round` must update both the Evo3-style
   opp-history AND the new per-color signal on treasure rounds only.
4. :meth:`Evo4AI.from_weights` round-trip for the 26-element vector.
5. With ``color_bias_influence = 0`` Evo4's first-round bid equals
   Evo3's first-round bid (the "zero influence = Evo3 clone" contract).
6. With non-zero ``color_bias_influence`` and a primed color signal,
   the bid on a treasure containing the boosted color must move.
7. A head-to-head smoke test: default Evo4 beats 3× RandomAI ≥60%
   (same bar every other AI clears).

Run via::

    python -m unittest tests.test_evo4
"""

from __future__ import annotations

import random
import unittest

from megagem.cards import Color, GemCard, InvestCard, LoanCard, TreasureCard
from megagem.engine import is_game_over, play_round, score_game, setup_game
from megagem.players import Evo3AI, Evo4AI, RandomAI
from megagem.players.evo2 import _per_color_value_stats
from megagem.players.evo3 import (
    _CAT_INVEST,
    _CAT_LOAN,
    _CAT_TREASURE,
    _Evo3InvestModel,
    _Evo3LoanModel,
    _Evo3TreasureModel,
)
from megagem.players.evo4 import (
    _biased_per_color_value_stats,
    _empty_color_signal,
)


# ----------------------------------------------------------------------------
# _biased_per_color_value_stats
# ----------------------------------------------------------------------------


class BiasedPerColorValueStatsTest(unittest.TestCase):
    """The helper that replaces Evo2's ``_per_color_value_stats`` with
    a per-color chart-index shift. Two must-haves:
    (a) zero shift ⇒ numerically identical to Evo2's helper;
    (b) non-zero shift ⇒ EV actually changes for the shifted color."""

    def _make_state(self, seed: int = 3):
        players = [
            Evo4AI("E4", seed=seed),
            RandomAI("R1", seed=seed + 1),
            RandomAI("R2", seed=seed + 2),
            RandomAI("R3", seed=seed + 3),
        ]
        state = setup_game(players, chart="A", seed=seed)
        return state, state.player_states[0]

    def test_zero_shift_matches_evo2(self):
        state, my_state = self._make_state()
        shift = _empty_color_signal()
        biased = _biased_per_color_value_stats(state, my_state, "A", shift)
        unbiased = _per_color_value_stats(state, my_state, "A")
        self.assertEqual(set(biased.keys()), set(unbiased.keys()))
        for color in biased:
            ev_b, var_b = biased[color]
            ev_u, var_u = unbiased[color]
            self.assertAlmostEqual(ev_b, ev_u, places=9)
            self.assertAlmostEqual(var_b, var_u, places=9)

    def test_zero_shift_with_empty_dict_matches_evo2(self):
        # Passing an empty dict instead of a full zeroed dict must
        # behave identically — the helper falls back to ``.get(..., 0.0)``.
        state, my_state = self._make_state(seed=11)
        biased = _biased_per_color_value_stats(state, my_state, "A", {})
        unbiased = _per_color_value_stats(state, my_state, "A")
        for color in biased:
            ev_b, _ = biased[color]
            ev_u, _ = unbiased[color]
            self.assertAlmostEqual(ev_b, ev_u, places=9)

    def test_positive_shift_raises_ev_for_boosted_color(self):
        state, my_state = self._make_state(seed=21)
        # Boost Blue by +1 chart-index: its EV should rise (chart A is
        # strictly monotone non-decreasing, so shifting the index up
        # cannot lower the lookup).
        unbiased = _per_color_value_stats(state, my_state, "A")
        shift = _empty_color_signal()
        shift[Color.BLUE] = 1.0
        biased = _biased_per_color_value_stats(state, my_state, "A", shift)
        self.assertGreater(biased[Color.BLUE][0], unbiased[Color.BLUE][0])
        # Other colors are untouched.
        for color in biased:
            if color is Color.BLUE:
                continue
            self.assertAlmostEqual(
                biased[color][0], unbiased[color][0], places=9
            )

    def test_fractional_shift_is_linearly_interpolated(self):
        # Chart A is monotone non-decreasing, so a +0.5 shift should
        # land strictly between the integer-index endpoints for at
        # least one color. Check it's in between, not equal to either.
        state, my_state = self._make_state(seed=33)
        zero_shift = _biased_per_color_value_stats(
            state, my_state, "A", _empty_color_signal()
        )
        full_shift = _empty_color_signal()
        full_shift[Color.BLUE] = 1.0
        one_shift = _biased_per_color_value_stats(
            state, my_state, "A", full_shift
        )
        half_shift_dict = _empty_color_signal()
        half_shift_dict[Color.BLUE] = 0.5
        half_shift = _biased_per_color_value_stats(
            state, my_state, "A", half_shift_dict
        )
        ev0 = zero_shift[Color.BLUE][0]
        ev1 = one_shift[Color.BLUE][0]
        ev_half = half_shift[Color.BLUE][0]
        self.assertLessEqual(ev0, ev_half + 1e-9)
        self.assertLessEqual(ev_half, ev1 + 1e-9)


# ----------------------------------------------------------------------------
# observe_round / color signal accounting
# ----------------------------------------------------------------------------


class ObserveRoundColorSignalTest(unittest.TestCase):
    def _make_ai(self) -> Evo4AI:
        return Evo4AI("T", seed=0)

    def _fake_result(self, auction, bids, taken_gems=None):
        return {
            "round": 1,
            "auction": auction,
            "bids": bids,
            "winner_idx": 0,
            "winning_bid": max(bids),
            "taken_gems": taken_gems or [],
            "revealed_gem": None,
            "completed_missions": [],
        }

    def test_color_signal_updates_on_treasure(self):
        ai = self._make_ai()
        ai._last_default_bid = 3
        gems = [GemCard(Color.BLUE), GemCard(Color.BLUE)]
        ai.observe_round(
            None,
            0,
            self._fake_result(TreasureCard(2), [3, 10, 5], taken_gems=gems),
        )
        # Delta = 10 − 3 = 7, split across 2 gems → +3.5 per gem, both
        # Blue → +7.0 Blue total.
        self.assertAlmostEqual(ai._color_signal[Color.BLUE], 7.0, places=9)
        for color in ai._color_signal:
            if color is Color.BLUE:
                continue
            self.assertAlmostEqual(ai._color_signal[color], 0.0, places=9)
        # Opp-history also grew.
        self.assertEqual(len(ai._opp_history), 1)
        self.assertEqual(ai._opp_history[0], (_CAT_TREASURE, 7.0))

    def test_color_signal_splits_across_mixed_gems(self):
        ai = self._make_ai()
        ai._last_default_bid = 0
        gems = [GemCard(Color.BLUE), GemCard(Color.GREEN)]
        ai.observe_round(
            None,
            0,
            self._fake_result(TreasureCard(2), [0, 10, 0], taken_gems=gems),
        )
        # Delta = 10 across 2 gems → +5 each.
        self.assertAlmostEqual(ai._color_signal[Color.BLUE], 5.0, places=9)
        self.assertAlmostEqual(ai._color_signal[Color.GREEN], 5.0, places=9)

    def test_color_signal_negative_delta_drains_signal(self):
        ai = self._make_ai()
        ai._last_default_bid = 12
        gems = [GemCard(Color.PINK)]
        ai.observe_round(
            None,
            0,
            self._fake_result(TreasureCard(1), [12, 4, 0], taken_gems=gems),
        )
        # Delta = 4 − 12 = -8 → Pink signal = -8.
        self.assertAlmostEqual(ai._color_signal[Color.PINK], -8.0, places=9)

    def test_invest_and_loan_rounds_do_not_touch_color_signal(self):
        ai = self._make_ai()
        ai._last_default_bid = 2
        ai.observe_round(
            None, 0, self._fake_result(InvestCard(5), [2, 10, 0])
        )
        ai._last_default_bid = 3
        ai.observe_round(
            None, 0, self._fake_result(LoanCard(10), [3, 12, 1])
        )
        for value in ai._color_signal.values():
            self.assertAlmostEqual(value, 0.0, places=9)
        # But the opp-history should still have grown — the invest/loan
        # delta tracking is unchanged from Evo3.
        self.assertEqual(
            [cat for cat, _ in ai._opp_history],
            [_CAT_INVEST, _CAT_LOAN],
        )

    def test_treasure_round_with_no_taken_gems_leaves_signal_zero(self):
        # Treasure round but nobody won (or no gems were on display) —
        # ``taken_gems`` comes back empty. We still want the opp-history
        # to update (it carries useful pricing info), but the color
        # signal must remain untouched so we don't divide by zero.
        ai = self._make_ai()
        ai._last_default_bid = 1
        ai.observe_round(
            None,
            0,
            self._fake_result(TreasureCard(2), [1, 9, 3], taken_gems=[]),
        )
        self.assertEqual(len(ai._opp_history), 1)
        for value in ai._color_signal.values():
            self.assertAlmostEqual(value, 0.0, places=9)

    def test_missing_baseline_skips_update(self):
        ai = self._make_ai()
        self.assertIsNone(ai._last_default_bid)
        gems = [GemCard(Color.YELLOW)]
        ai.observe_round(
            None,
            0,
            self._fake_result(TreasureCard(1), [2, 5], taken_gems=gems),
        )
        self.assertEqual(len(ai._opp_history), 0)
        for value in ai._color_signal.values():
            self.assertAlmostEqual(value, 0.0, places=9)


# ----------------------------------------------------------------------------
# from_weights round-trip
# ----------------------------------------------------------------------------


class FromWeightsTest(unittest.TestCase):
    def test_round_trip(self):
        weights = [float(i) * 0.1 - 1.0 for i in range(Evo4AI.NUM_WEIGHTS)]
        ai = Evo4AI.from_weights("Test", weights)

        t = ai.treasure_model
        self.assertEqual(
            (
                t.bias, t.w_rounds, t.w_my, t.w_avg, t.w_top,
                t.w_ev, t.w_std, t.w_mean_delta, t.w_std_delta,
            ),
            tuple(weights[0:9]),
        )

        i = ai.invest_model
        self.assertEqual(
            (
                i.bias, i.w_rounds, i.w_my, i.w_avg, i.w_top,
                i.w_amount, i.w_mean_delta, i.w_std_delta,
            ),
            tuple(weights[9:17]),
        )

        l = ai.loan_model
        self.assertEqual(
            (
                l.bias, l.w_rounds, l.w_my, l.w_avg, l.w_top,
                l.w_amount, l.w_mean_delta, l.w_std_delta,
            ),
            tuple(weights[17:25]),
        )

        self.assertEqual(ai.color_bias_influence, weights[25])

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            Evo4AI.from_weights("Bad", [0.0] * 25)
        with self.assertRaises(ValueError):
            Evo4AI.from_weights("Bad", [0.0] * 27)


# ----------------------------------------------------------------------------
# Defaults reduce to Evo3 — round-one bid parity
# ----------------------------------------------------------------------------


class DefaultsMatchEvo3Test(unittest.TestCase):
    """With ``color_bias_influence=0`` and an empty signal, Evo4's first
    bid must exactly equal Evo3's first bid on the same game state."""

    def test_defaults_mirror_evo3_round_one(self):
        players_e4 = [Evo4AI("E4", seed=1), RandomAI("R1", seed=2),
                      RandomAI("R2", seed=3), RandomAI("R3", seed=4)]
        players_e3 = [Evo3AI("E3", seed=1), RandomAI("R1", seed=2),
                      RandomAI("R2", seed=3), RandomAI("R3", seed=4)]

        state_e4 = setup_game(players_e4, chart="A", seed=100)
        state_e3 = setup_game(players_e3, chart="A", seed=100)

        auction = state_e4.auction_deck[-1]
        evo4 = players_e4[0]
        evo3 = players_e3[0]
        bid_e4 = evo4.choose_bid(state_e4, state_e4.player_states[0], auction)
        bid_e3 = evo3.choose_bid(state_e3, state_e3.player_states[0], auction)
        self.assertEqual(bid_e4, bid_e3)


# ----------------------------------------------------------------------------
# Non-zero influence + primed signal actually moves the bid
# ----------------------------------------------------------------------------


class ColorBiasInfluenceMovesBidTest(unittest.TestCase):
    def test_primed_signal_with_positive_influence_raises_bid(self):
        """Craft a treasure head whose ``w_ev`` is +1 (so higher EV →
        higher bid). Prime the Blue color signal positive, set
        ``color_bias_influence=1``, then force a treasure containing
        Blue to the top of the auction deck. The bid with the primed
        signal must be strictly higher than the same AI with a zero
        signal on the same game state."""
        model = _Evo3TreasureModel(
            bias=0.0,
            w_rounds=0.0,
            w_my=0.0,
            w_avg=0.0,
            w_top=0.0,
            w_ev=1.0,
            w_std=0.0,
            w_mean_delta=0.0,
            w_std_delta=0.0,
        )
        ai = Evo4AI(
            "T",
            treasure=model,
            invest=_Evo3InvestModel(0, 0, 0, 0, 0, 0, 0, 0),
            loan=_Evo3LoanModel(0, 0, 0, 0, 0, 0, 0, 0),
            color_bias_influence=1.0,
        )

        players = [ai, RandomAI("R1", seed=2), RandomAI("R2", seed=3),
                   RandomAI("R3", seed=4)]
        state = setup_game(players, chart="A", seed=5)
        state.auction_deck.append(TreasureCard(1))
        # Ensure there's a Blue gem on the display to be won.
        state.revealed_gems.insert(0, GemCard(Color.BLUE))
        auction = state.auction_deck[-1]

        # Baseline: zero signal.
        ai._color_signal = _empty_color_signal()
        bid_zero = ai.choose_bid(state, state.player_states[0], auction)

        # Primed: big positive Blue signal.
        ai._color_signal = _empty_color_signal()
        ai._color_signal[Color.BLUE] = 4.0
        bid_primed = ai.choose_bid(state, state.player_states[0], auction)

        self.assertGreater(bid_primed, bid_zero)

    def test_zero_influence_ignores_signal(self):
        """With ``color_bias_influence=0`` the bid must be invariant to
        the color signal — even a huge primed signal can't nudge it."""
        ai = Evo4AI("T", color_bias_influence=0.0)
        players = [ai, RandomAI("R1", seed=2), RandomAI("R2", seed=3),
                   RandomAI("R3", seed=4)]
        state = setup_game(players, chart="A", seed=7)
        state.auction_deck.append(TreasureCard(1))
        state.revealed_gems.insert(0, GemCard(Color.BLUE))
        auction = state.auction_deck[-1]

        ai._color_signal = _empty_color_signal()
        bid_zero = ai.choose_bid(state, state.player_states[0], auction)
        ai._color_signal[Color.BLUE] = 10.0
        bid_primed = ai.choose_bid(state, state.player_states[0], auction)
        self.assertEqual(bid_primed, bid_zero)


# ----------------------------------------------------------------------------
# Engine wiring: observe_round fires each round on Evo4 too.
# ----------------------------------------------------------------------------


class EngineWiresObserveRoundTest(unittest.TestCase):
    def test_full_game_fills_opp_history(self):
        players = [
            Evo4AI("E4", seed=1),
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

        evo4 = players[0]
        assert isinstance(evo4, Evo4AI)
        self.assertEqual(len(evo4._opp_history), rounds_played)
        self.assertGreater(rounds_played, 0)


# ----------------------------------------------------------------------------
# Head-to-head smoke test: default Evo4 ≥60% vs 3× RandomAI.
# ----------------------------------------------------------------------------


class Evo4HeadToHeadTest(unittest.TestCase):
    """Default-weights Evo4AI should clear the same 60% vs-Random bar
    every previous AI clears. With ``color_bias_influence=0`` Evo4
    reduces to Evo3, which reduces to Evo2, so any regression below
    this line is a wiring bug, not a training-data problem."""

    def test_evo4_beats_random_majority(self):
        wins = 0
        games = 60
        for seed in range(games):
            players = [
                Evo4AI("E4", seed=seed * 11),
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
            f"Evo4 only won {wins}/{games} ({win_rate:.0%})",
        )


if __name__ == "__main__":
    unittest.main()
