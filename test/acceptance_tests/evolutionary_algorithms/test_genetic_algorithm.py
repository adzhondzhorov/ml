from unittest import TestCase

from app.evolutionary_algorithms.genetic_algorithm import GeneticAlgorithm

from test.acceptance_tests.data import WEATHER_SPORT_PLAY_DATA
from test.acceptance_tests.base_test import assert_learn


class GeneticAlgorithmAcceptanceTests(TestCase):
    def test_ga(self):
        ga = GeneticAlgorithm()
        assert_learn(ga)
