from unittest import TestCase

from app.ensemble_learning.weighted_majority import WeightedMajority
from app.bayesian_learning.naive_bayes import NaiveBayes
from app.decision_tree.id3 import ID3

from test.acceptance_tests.data import WEATHER_SPORT_PLAY_DATA
from test.acceptance_tests.base_test import assert_learn


class WeightedMajorityAcceptanceTests(TestCase):
    def test_weighted_majority(self):
        wm = WeightedMajority([NaiveBayes(len(WEATHER_SPORT_PLAY_DATA[0]), len(WEATHER_SPORT_PLAY_DATA)),
                               ID3(len(WEATHER_SPORT_PLAY_DATA[0]))])
        assert_learn(wm)