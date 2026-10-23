import random
import unittest
from collections import Counter

from megagem.browser import DEFAULT_WEIGHTS_DIR, _load_evo_weights
from megagem.cards import Color, GemCard, InvestCard, TreasureCard
from megagem.engine import is_game_over, max_legal_bid, play_round, score_game, setup_game
from megagem.players import Evo4AI, HeuristicAI, RandomAI
from megagem.players.search import SearchAI, _multisets, unseen_pool


def _small(name, seed, **kw):
    """A cheap SearchAI for tests: few worlds, default-weight Evo4 models."""
    return SearchAI(name, seed=seed, worlds=kw.pop("worlds", 4), **kw)


def _mid_game_state(seed=3, rounds=4):
    players = [HeuristicAI(f"p{i}", seed=i) for i in range(4)]
    state = setup_game(players, chart="E", seed=seed)
    rng = random.Random(seed)
    for _ in range(rounds):
        play_round(state, rng=rng)
    return state


class UnseenPoolTest(unittest.TestCase):
    def test_pool_is_opponents_hands_plus_deck(self):
        state = _mid_game_state()
        me = state.player_states[0]
        hidden = Counter(g.color for g in state.gem_deck)
        for ps in state.player_states[1:]:
            hidden.update(g.color for g in ps.hand)
        self.assertEqual(Counter(unseen_pool(state, me)), hidden)

    def test_multisets_respect_limits(self):
        combos = list(_multisets(4, [6, 6, 6, 6, 6]))
        self.assertEqual(len(combos), 70)  # C(8, 4)
        for combo in _multisets(3, [1, 0, 2, 0, 5]):
            self.assertEqual(sum(combo), 3)
            self.assertLessEqual(combo[0], 1)
            self.assertEqual(combo[1], 0)


class ThresholdTest(unittest.TestCase):
    def test_tie_break_follows_seat_after_last_winner(self):
        state = _mid_game_state()
        bot = _small("me", 0)
        # Seat 0 is me. Last winner seat 2 → seat order 3, 0, 1: on a tie
        # with seat 1 I win (I'm closer); on a tie with seat 3 I lose.
        state.last_winner_idx = 2
        self.assertEqual(bot._threshold(state, 0, {1: 5, 2: 3, 3: 1}), 5)
        self.assertEqual(bot._threshold(state, 0, {1: 1, 2: 3, 3: 5}), 6)

    def test_first_round_ties_are_not_trusted(self):
        state = _mid_game_state()
        state.last_winner_idx = None
        bot = _small("me", 0)
        self.assertEqual(bot._threshold(state, 0, {1: 4, 2: 4, 3: 0}), 5)


class NoPeekingTest(unittest.TestCase):
    def test_bid_ignores_the_real_hidden_cards(self):
        """Reshuffling opponents' hands + the deck must not change the bid."""
        state = _mid_game_state()
        auction = TreasureCard(1)
        me = state.player_states[0]
        first = _small("me", 11).choose_bid(state, me, auction)

        hidden = list(state.gem_deck)
        for ps in state.player_states[1:]:
            hidden += ps.hand
        random.Random(99).shuffle(hidden)
        pos = 0
        for ps in state.player_states[1:]:
            ps.hand = hidden[pos:pos + len(ps.hand)]
            pos += len(ps.hand)
        state.gem_deck = hidden[pos:]
        self.assertEqual(_small("me", 11).choose_bid(state, me, auction), first)


class BranchConsistencyTest(unittest.TestCase):
    """The win branch must really win the auction and the lose branch lose it.

    Regression: with shadow models the rollouts used to re-derive this
    round's opponent bids from *different* (history-free) models, so the
    "win at the threshold" branch often wasn't a win at all.
    """

    def test_branches_resolve_as_labelled(self):
        from megagem import engine

        outcomes = []

        class Probe(SearchAI):
            def _play_out(self, sim, my_idx, rng):
                result = engine.play_round(sim, rng=rng)
                bid = result["bids"][my_idx]
                outcomes.append((bid, result["winner_idx"] == my_idx))
                return 0.5

        # Checked-in weights: their history terms are non-zero, so shadow
        # models genuinely differ from fresh ones.
        w = _load_evo_weights(DEFAULT_WEIGHTS_DIR, "evo4", 4)
        for shadow in (False, True):
            bot = Probe("me", seed=3, worlds=6, shadow=shadow, model_weights=w)
            players = [bot] + [Evo4AI.from_weights(f"o{i}", w) for i in range(3)]
            state = setup_game(players, chart="B", seed=6)
            rng = random.Random(6)
            checked = 0
            while not is_game_over(state):
                outcomes.clear()
                me = state.player_states[0]
                auction = state.auction_deck.pop()  # as play_round does
                bot.choose_bid(state, me, auction)
                state.auction_deck.append(auction)
                if bot.last_search is not None and outcomes:
                    cap = max_legal_bid(me, auction)
                    expected = []
                    for t in bot.last_search["thresholds"]:
                        if t > 0:
                            expected.append((0, False))  # lose branch
                        if t <= cap:
                            expected.append((t, True))   # win branch
                    self.assertEqual(outcomes, expected, f"shadow={shadow}")
                    checked += 1
                bot.last_search = None
                play_round(state, rng=rng)
            self.assertGreater(checked, 5)


class InferenceTest(unittest.TestCase):
    def test_records_treasure_bids_and_reveals(self):
        bot = _small("me", 1, infer=True)
        players = [bot] + [Evo4AI(f"o{i}") for i in range(3)]
        state = setup_game(players, chart="A", seed=4)
        rng = random.Random(4)
        treasures = 0
        for _ in range(6):
            result = play_round(state, rng=rng)
            treasures += isinstance(result["auction"], TreasureCard)
        self.assertEqual(len(bot._bid_records), 3 * treasures)
        # A world dealing every opponent its true hand explains their bids
        # (default Evo4 opponents are exactly the default model).
        true_hands = [ps.hand for ps in state.player_states]
        self.assertGreater(bot._world_weight(true_hands), 0.0)

    def test_identification_recognises_bot_types(self):
        bank = {
            "evo4": lambda n, s: Evo4AI(n, seed=s),
            "heuristic": lambda n, s: HeuristicAI(n, seed=s),
            "random": lambda n, s: RandomAI(n, seed=s),
        }
        bot = _small("me", 2, model_bank=bank, identify=True, shadow=True)
        players = [bot, HeuristicAI("h"), Evo4AI("e"), HeuristicAI("h2")]
        state = setup_game(players, chart="A", seed=5)
        rng = random.Random(5)
        while not is_game_over(state):
            play_round(state, rng=rng)
        self.assertEqual(bot._seat_kind(1), "heuristic")
        self.assertEqual(bot._seat_kind(2), "evo4")


class HeadToHeadTest(unittest.TestCase):
    def test_beats_random_on_chart_a(self):
        wins = 0
        games = 10
        for g in range(games):
            players = [_small("S", g)] + [RandomAI(f"R{i}", seed=g * 10 + i) for i in range(3)]
            state = setup_game(players, chart="A", seed=g)
            rng = random.Random(g)
            while not is_game_over(state):
                play_round(state, rng=rng)
            totals = [r["total"] for r in score_game(state)]
            wins += totals[0] == max(totals)
        self.assertGreaterEqual(wins / games, 0.6)

    def test_seeded_bot_is_reproducible(self):
        def run():
            players = [_small("S", 7, reveal_search=True)] + [Evo4AI(f"E{i}") for i in range(3)]
            state = setup_game(players, chart="C", seed=8)
            rng = random.Random(8)
            bids = []
            while not is_game_over(state):
                bids.append(play_round(state, rng=rng)["bids"])
            return bids
        self.assertEqual(run(), run())


if __name__ == "__main__":
    unittest.main()
