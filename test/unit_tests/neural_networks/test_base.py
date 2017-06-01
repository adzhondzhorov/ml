from unittest import TestCase

from app.neural_networks.base import sgn


class TestInstance(TestCase):
    def test_sgn_positive(self):
        assert sgn(100) == 1
    
    def test_sgn_negative(self):
        assert sgn(-0.1) == -1

    def test_sgn_zero(self):
        assert sgn(0) == -1
