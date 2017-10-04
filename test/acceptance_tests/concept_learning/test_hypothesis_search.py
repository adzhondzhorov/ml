from unittest import TestCase

from app.concept_learning.hypothesis_search import FindS, CandidateElimination

from test.acceptance_tests.data import WEATHER_SPORT_PLAY_DATA
from test.acceptance_tests.base_test import assert_learn


class HypothesisSearchAcceptanceTests(TestCase):
    def test_find_s(self):
        find_s = FindS(len(WEATHER_SPORT_PLAY_DATA[0]))
        assert_learn(find_s)


    def test_candidate_elimination(self):
        candidate_elimination = CandidateElimination(len(WEATHER_SPORT_PLAY_DATA[0]))
        assert_learn(candidate_elimination)
