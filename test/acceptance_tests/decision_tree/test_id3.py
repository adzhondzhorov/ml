from unittest import TestCase

from app.decision_tree.id3 import ID3

from test.acceptance_tests.data import WEATHER_SPORT_PLAY_DATA
from test.acceptance_tests.base_test import assert_learn


class ID3AcceptanceTests(TestCase):
    def test_id3(self):
        id3 = ID3(len(WEATHER_SPORT_PLAY_DATA[0]))
        assert_learn(id3)