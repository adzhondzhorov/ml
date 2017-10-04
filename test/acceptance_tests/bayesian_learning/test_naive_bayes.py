from unittest import TestCase

from app.bayesian_learning.naive_bayes import NaiveBayes

from test.acceptance_tests.data import WEATHER_SPORT_PLAY_DATA
from test.acceptance_tests.base_test import assert_learn


class NaiveBayesAcceptanceTests(TestCase):
    def test_naive_bayes(self):
        nb = NaiveBayes(len(WEATHER_SPORT_PLAY_DATA[0]), len(WEATHER_SPORT_PLAY_DATA))
        assert_learn(nb)