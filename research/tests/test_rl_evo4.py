"""Tests for the ES trainer in :mod:`scripts.rl`.

Follows the repo's stdlib-unittest conventions (see ``test_evo4.py``):
each helper gets a focused unit test before the outer loop is
exercised, and the final integration test mirrors the
``Evo4HeadToHeadTest`` pattern — 60 games vs 3× RandomAI on chart A,
asserting a ≥60% win rate.

Critically, all tests run with ``workers=1`` so they are bit-for-bit
deterministic and don't rely on :mod:`multiprocessing` at all. The
parallel path is exercised by the CLI smoke run, not by the unit
tests.
"""

from __future__ import annotations

import math
import random
import unittest

from megagem.engine import is_game_over, play_round, score_game, setup_game
from megagem.players import Evo4AI, RandomAI
from scripts.evolve.profiles import AI_PROFILES

from scripts.rl import run_es
from scripts.rl.fitness import _game_reward, rank_transform
from scripts.rl.optim import Adam
from scripts.rl.trainer import (
    _clip,
    _deserialize_rng_state,
    _serialize_rng_state,
    sample_epsilons,
)


# ---------------------------------------------------------------------------
# Rank-transform unit tests
# ---------------------------------------------------------------------------


class RankTransformTest(unittest.TestCase):
    """Fitness shaping: the centered rank transform.

    Three invariants and two edge cases:

    * Sums to ~0 (centered).
    * Monotone-preserving (lowest input → lowest output).
    * Ties are broken deterministically by index.
    * Empty input → empty output.
    * Single-element input → ``[0.0]`` (no divide-by-zero).
    """

    def test_sum_is_zero(self):
        values = [3.1, -2.0, 5.5, 0.0, 1.2]
        out = rank_transform(values)
        self.assertAlmostEqual(sum(out), 0.0, places=9)

    def test_monotone_preserving(self):
        values = [-1.0, 0.0, 1.0, 2.0, 3.0]
        out = rank_transform(values)
        # Lowest input gets the lowest output; highest gets the highest.
        for i in range(len(out) - 1):
            self.assertLess(out[i], out[i + 1])
        self.assertAlmostEqual(out[0], -0.5, places=9)
        self.assertAlmostEqual(out[-1], 0.5, places=9)

    def test_ties_broken_deterministically(self):
        # All equal → early indices get lower ranks by the stable sort.
        values = [1.0, 1.0, 1.0, 1.0]
        out = rank_transform(values)
        self.assertEqual(len(out), 4)
        self.assertAlmostEqual(out[0], -0.5, places=9)
        self.assertAlmostEqual(out[-1], 0.5, places=9)
        # Monotone in index order for the degenerate case.
        for i in range(len(out) - 1):
            self.assertLess(out[i], out[i + 1])

    def test_empty(self):
        self.assertEqual(rank_transform([]), [])

    def test_single(self):
        self.assertEqual(rank_transform([42.0]), [0.0])


# ---------------------------------------------------------------------------
# Game-reward unit test
# ---------------------------------------------------------------------------


class GameRewardTest(unittest.TestCase):
    """Normalized score margin must stay in ``[-1, 1]`` and match the
    sign of the raw margin."""

    def test_positive_margin(self):
        scores = [
            {"total": 100},
            {"total": 60},
            {"total": 40},
            {"total": 20},
        ]
        r = _game_reward(scores, my_idx=0)
        # mine=100, best_opp=60 → (100-60)/(100+60) = 0.25
        self.assertAlmostEqual(r, 0.25, places=6)

    def test_negative_margin(self):
        scores = [
            {"total": 30},
            {"total": 70},
            {"total": 20},
            {"total": 10},
        ]
        r = _game_reward(scores, my_idx=0)
        self.assertAlmostEqual(r, (30 - 70) / (30 + 70), places=6)
        self.assertLess(r, 0)

    def test_zero_division_safe(self):
        scores = [{"total": 0}, {"total": 0}]
        r = _game_reward(scores, my_idx=0)
        self.assertEqual(r, 0.0)


# ---------------------------------------------------------------------------
# Adam unit tests
# ---------------------------------------------------------------------------


class AdamStepTest(unittest.TestCase):
    """Adam moves θ in the gradient direction, does not mutate input,
    and round-trips through ``state_dict``."""

    def test_positive_grad_moves_theta_up(self):
        adam = Adam(lr=0.1)
        adam.init(3)
        theta = [0.0, 0.0, 0.0]
        grad = [1.0, 1.0, 1.0]
        new_theta = adam.step(theta, grad)
        for w in new_theta:
            self.assertGreater(w, 0.0)

    def test_negative_grad_moves_theta_down(self):
        adam = Adam(lr=0.1)
        adam.init(3)
        theta = [0.0, 0.0, 0.0]
        grad = [-1.0, -1.0, -1.0]
        new_theta = adam.step(theta, grad)
        for w in new_theta:
            self.assertLess(w, 0.0)

    def test_step_does_not_mutate_input(self):
        adam = Adam(lr=0.1)
        theta = [1.0, 2.0, 3.0]
        grad = [0.1, 0.2, 0.3]
        original = list(theta)
        adam.step(theta, grad)
        self.assertEqual(theta, original)

    def test_state_dict_round_trip(self):
        adam = Adam(lr=0.05, beta1=0.85, beta2=0.995, eps=1e-7)
        adam.init(5)
        theta = [0.1, -0.2, 0.3, -0.4, 0.5]
        grad = [1.0, -1.0, 0.5, -0.5, 0.0]
        theta = adam.step(theta, grad)
        theta = adam.step(theta, grad)  # build up some moment

        restored = Adam.from_state(adam.state_dict())
        self.assertEqual(restored.lr, adam.lr)
        self.assertEqual(restored.beta1, adam.beta1)
        self.assertEqual(restored.beta2, adam.beta2)
        self.assertEqual(restored.eps, adam.eps)
        self.assertEqual(restored.m, adam.m)
        self.assertEqual(restored.v, adam.v)
        self.assertEqual(restored.t, adam.t)

        # After the round-trip both optimizers must produce the same
        # next update given the same gradient.
        next_from_original = adam.step(list(theta), [0.7, 0.0, -0.3, 0.1, 0.2])
        next_from_restored = restored.step(list(theta), [0.7, 0.0, -0.3, 0.1, 0.2])
        self.assertEqual(next_from_original, next_from_restored)


# ---------------------------------------------------------------------------
# Epsilon sampling + theta-clip unit tests
# ---------------------------------------------------------------------------


class SampleEpsilonsTest(unittest.TestCase):
    def test_length_and_mirroring(self):
        rng = random.Random(7)
        eps = sample_epsilons(n_pairs=3, dim=5, rng=rng)
        self.assertEqual(len(eps), 6)  # 2 * n_pairs
        for k in range(3):
            pos = eps[2 * k]
            neg = eps[2 * k + 1]
            self.assertEqual(len(pos), 5)
            self.assertEqual(len(neg), 5)
            for a, b in zip(pos, neg):
                self.assertAlmostEqual(a + b, 0.0, places=12)

    def test_seeded_reproducibility(self):
        a = sample_epsilons(2, 4, random.Random(42))
        b = sample_epsilons(2, 4, random.Random(42))
        self.assertEqual(a, b)


class ClipTest(unittest.TestCase):
    def test_clips_both_ends(self):
        out = _clip([10.0, -10.0, 1.0, -1.0, 0.0], clip=5.0)
        self.assertEqual(out, [5.0, -5.0, 1.0, -1.0, 0.0])

    def test_does_not_mutate_input(self):
        theta = [10.0, -10.0]
        _clip(theta, 5.0)
        self.assertEqual(theta, [10.0, -10.0])


# ---------------------------------------------------------------------------
# RNG state serialization round-trip
# ---------------------------------------------------------------------------


class RngStateSerializationTest(unittest.TestCase):
    def test_round_trip(self):
        rng = random.Random(123)
        for _ in range(10):
            rng.random()
        state = rng.getstate()
        serialized = _serialize_rng_state(state)
        # Serialized form is JSON-friendly: no tuples at any nesting.
        self._assert_no_tuples(serialized)
        deserialized = _deserialize_rng_state(serialized)
        self.assertEqual(deserialized, state)

        # After restoring, the RNG produces the same sequence.
        rng2 = random.Random()
        rng2.setstate(deserialized)
        for _ in range(5):
            self.assertEqual(rng.random(), rng2.random())

    def _assert_no_tuples(self, obj):
        if isinstance(obj, tuple):
            self.fail(f"unexpected tuple in serialized form: {obj!r}")
        if isinstance(obj, list):
            for x in obj:
                self._assert_no_tuples(x)


# ---------------------------------------------------------------------------
# Tiny end-to-end training step
# ---------------------------------------------------------------------------


class TinyTrainingStepTest(unittest.TestCase):
    """One generation of ES, 4-perturbation batch, 1 game per chart.

    Asserts: θ is a 35-long list of finite floats and the held-out
    evaluation curves are populated. This is the smallest possible
    integration test — it proves the full loop wires together without
    proving any learning has happened.
    """

    def test_one_generation_completes(self):
        profile = AI_PROFILES["evo4"]
        result = run_es(
            profile,
            mode_key="vs_random",
            population_size=4,
            generations=1,
            games_per_chart=1,
            sigma=0.1,
            lr=0.03,
            seed=0,
            num_players=4,
            workers=1,
            quiet=True,
        )
        self.assertEqual(len(result.theta), 35)
        for w in result.theta:
            self.assertTrue(math.isfinite(w), f"non-finite weight: {w}")
            self.assertGreaterEqual(w, -5.0)
            self.assertLessEqual(w, 5.0)
        self.assertEqual(len(result.mean_reward_per_gen), 1)
        self.assertEqual(len(result.win_rate_per_gen), 1)
        self.assertEqual(len(result.training_mean_per_gen), 1)
        self.assertEqual(result.generations_completed, 1)
        # Resume-state payload populated.
        self.assertIn("m", result.adam_state)
        self.assertIn("v", result.adam_state)
        self.assertEqual(result.adam_state["t"], 1)
        self.assertIsNotNone(result.rng_state)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class DeterminismTest(unittest.TestCase):
    """Two back-to-back runs with identical args produce identical θ.

    This is the sanity check that catches (a) accidental use of a
    global :mod:`random` singleton, (b) nondeterministic merge order
    in the fitness evaluator, (c) set-iteration noise, and (d) dict
    insertion-order drift. ``workers=1`` sidesteps the
    multiprocessing dimension — the parallel path uses the same
    pre-allocated accumulator pattern, so sequential determinism
    implies parallel determinism by construction.
    """

    def test_identical_runs_produce_identical_theta(self):
        profile = AI_PROFILES["evo4"]
        kwargs = dict(
            mode_key="vs_random",
            population_size=4,
            generations=2,
            games_per_chart=1,
            sigma=0.1,
            lr=0.03,
            seed=7,
            num_players=4,
            workers=1,
            quiet=True,
        )
        a = run_es(profile, **kwargs)
        b = run_es(profile, **kwargs)
        self.assertEqual(a.theta, b.theta)
        self.assertEqual(a.mean_reward_per_gen, b.mean_reward_per_gen)
        self.assertEqual(a.win_rate_per_gen, b.win_rate_per_gen)
        self.assertEqual(a.best_theta, b.best_theta)


# ---------------------------------------------------------------------------
# Head-to-head integration: trained θ clears the 60% vs-Random bar
# ---------------------------------------------------------------------------


class HeadToHeadIntegrationTest(unittest.TestCase):
    """Mirrors :class:`tests.test_evo4.Evo4HeadToHeadTest`.

    Trains Evo4 for 3 tiny generations against RandomAI, then replays
    the canonical 60-game chart-A test from ``test_evo4.py:667``. The
    60% bar is the repo-wide minimum — every AI in the zoo clears
    it at its class defaults, and any tuned-Evo4 that *fails* to
    clear it is a regression, not a training-data problem.

    Default Evo4 already clears the bar (see existing ``test_evo4``),
    so this test is mostly proving the ES loop doesn't corrupt an
    already-good θ. Three generations is overkill for that, but it
    also exercises the multi-gen code path that determines the
    ``best_theta`` selection.
    """

    def test_trained_evo4_beats_random_majority(self):
        profile = AI_PROFILES["evo4"]
        result = run_es(
            profile,
            mode_key="vs_random",
            population_size=8,
            generations=3,
            games_per_chart=2,
            sigma=0.1,
            lr=0.03,
            seed=0,
            num_players=4,
            workers=1,
            quiet=True,
        )
        trained_weights = result.best_theta

        wins = 0
        games = 60
        for seed in range(games):
            players = [
                Evo4AI.from_weights("RL", trained_weights, seed=seed * 11),
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
            0.60,
            f"RL-trained Evo4 only won {wins}/{games} ({win_rate:.0%})",
        )


if __name__ == "__main__":
    unittest.main()
