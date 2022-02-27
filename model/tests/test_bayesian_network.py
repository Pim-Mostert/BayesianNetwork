from unittest import TestCase

from model.bayesian_network import BayesianNetwork
from common.utilities import Empty


class TestBayesianNetwork(TestCase):
    def setUp(self):
        cfg = Empty()
        cfg.a = 8
        cfg.b = 14

        self.sut = BayesianNetwork(cfg)

    def test_calculate(self):
        result = self.sut.calculate()
        self.assertEqual(result, 22)

    def test_calculate_notequal(self):
        result = self.sut.calculate()
        self.assertNotEqual(result, 21)
