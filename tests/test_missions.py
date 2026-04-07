import unittest
from collections import Counter

from megagem.cards import Color
from megagem.missions import (
    at_least_n_distinct_colors,
    color_counts_at_least,
    make_mission_deck,
    n_of_same_any_color,
    n_pairs_any_color,
)


class MissionDeckTest(unittest.TestCase):
    def test_total_count(self):
        deck = make_mission_deck()
        self.assertEqual(len(deck), 30)

    def test_category_breakdown(self):
        deck = make_mission_deck()
        cats = Counter(m.category for m in deck)
        self.assertEqual(cats["shield"], 2)
        self.assertEqual(cats["pendant"], 16)
        self.assertEqual(cats["crown"], 12)

    def test_pendant_specific_pairs(self):
        deck = make_mission_deck()
        pendants = [m for m in deck if m.category == "pendant"]
        # 1 generic 2-of-same + 5 specific 2-of-X + 10 X+Y pairs
        self.assertEqual(len(pendants), 16)

    def test_crown_specific_triples(self):
        deck = make_mission_deck()
        crowns = [m for m in deck if m.category == "crown"]
        # 1 generic 3-of-same + 1 generic 3-different + 10 X+Y+Z triples
        self.assertEqual(len(crowns), 12)


class RequirementTest(unittest.TestCase):
    def test_at_least_n_distinct(self):
        req = at_least_n_distinct_colors(3)
        self.assertFalse(req(Counter({Color.BLUE: 5, Color.GREEN: 1})))
        self.assertTrue(req(Counter({Color.BLUE: 1, Color.GREEN: 1, Color.PINK: 1})))
        self.assertTrue(req(Counter({Color.BLUE: 1, Color.GREEN: 1, Color.PINK: 1, Color.PURPLE: 1})))

    def test_n_pairs(self):
        req = n_pairs_any_color(2)
        self.assertFalse(req(Counter({Color.BLUE: 2, Color.GREEN: 1})))
        self.assertTrue(req(Counter({Color.BLUE: 2, Color.GREEN: 2})))
        self.assertTrue(req(Counter({Color.BLUE: 4, Color.GREEN: 2})))

    def test_n_of_same(self):
        req3 = n_of_same_any_color(3)
        self.assertFalse(req3(Counter({Color.BLUE: 2, Color.GREEN: 2})))
        self.assertTrue(req3(Counter({Color.BLUE: 3})))

    def test_color_counts_at_least(self):
        req = color_counts_at_least({Color.BLUE: 2, Color.GREEN: 1})
        self.assertFalse(req(Counter({Color.BLUE: 1, Color.GREEN: 5})))
        self.assertTrue(req(Counter({Color.BLUE: 2, Color.GREEN: 1})))
        self.assertTrue(req(Counter({Color.BLUE: 3, Color.GREEN: 2, Color.PINK: 1})))


if __name__ == "__main__":
    unittest.main()
