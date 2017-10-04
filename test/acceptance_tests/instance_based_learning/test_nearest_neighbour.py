from unittest import TestCase

from app.instance_based_learning.nearest_neighbours import KNearestNeighbours

from test.acceptance_tests.base_test import assert_learn

class KNearestNeighboursAcceptanceTests(TestCase):
    def test_k_nearest_neighbour(self):
        knn = KNearestNeighbours(1)
        assert_learn(knn)