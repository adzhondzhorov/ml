from unittest import TestCase

from app.rule_based_learning.sequential_covering import SequentialCovering

from test.acceptance_tests.data import WEATHER_SPORT_PLAY_DATA
from test.acceptance_tests.base_test import assert_learn

class SequentialCoveringAcceptanceTests(TestCase):
    def test_k_nearest_neighbour(self):
        sc = SequentialCovering(len(WEATHER_SPORT_PLAY_DATA[0]))
        assert_learn(sc)