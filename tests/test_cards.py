import unittest
from collections import Counter

from megagem.cards import (
    Color,
    InvestCard,
    LoanCard,
    TreasureCard,
    make_auction_deck,
    make_gem_deck,
)


class GemDeckTest(unittest.TestCase):
    def test_total_count(self):
        deck = make_gem_deck()
        self.assertEqual(len(deck), 30)

    def test_six_per_color(self):
        deck = make_gem_deck()
        counts = Counter(card.color for card in deck)
        for color in Color:
            self.assertEqual(counts[color], 6, f"{color} should have 6 cards")


class AuctionDeckTest(unittest.TestCase):
    def test_total_count(self):
        deck = make_auction_deck()
        self.assertEqual(len(deck), 25)

    def test_breakdown(self):
        deck = make_auction_deck()
        treasure_1 = sum(1 for c in deck if isinstance(c, TreasureCard) and c.gems == 1)
        treasure_2 = sum(1 for c in deck if isinstance(c, TreasureCard) and c.gems == 2)
        loan_10 = sum(1 for c in deck if isinstance(c, LoanCard) and c.amount == 10)
        loan_20 = sum(1 for c in deck if isinstance(c, LoanCard) and c.amount == 20)
        invest_5 = sum(1 for c in deck if isinstance(c, InvestCard) and c.amount == 5)
        invest_10 = sum(1 for c in deck if isinstance(c, InvestCard) and c.amount == 10)

        self.assertEqual(treasure_1, 12)
        self.assertEqual(treasure_2, 5)
        self.assertEqual(loan_10, 2)
        self.assertEqual(loan_20, 2)
        self.assertEqual(invest_5, 2)
        self.assertEqual(invest_10, 2)


if __name__ == "__main__":
    unittest.main()
