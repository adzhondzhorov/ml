from unittest import TestCase

from app.neural_networks.base import sgn, sigmoid

EPSILON = 0.01

class TestInstance(TestCase):
    def test_sgn_positive(self):
        assert sgn(100) == 1
    
    def test_sgn_negative(self):
        assert sgn(-0.1) == -1

    def test_sgn_zero(self):
        assert sgn(0) == -1

    def test_sigmoid(self):
        assert abs(sigmoid(-0.71) - 0.329) < EPSILON

    def test_sigmoid_zero(self):
        return sigmoid(0) == 0.5
