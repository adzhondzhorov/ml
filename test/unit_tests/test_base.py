from unittest import TestCase

from app.base import Instance


class TestInstance(TestCase):
    def test_iter(self):
        instance = Instance(["A", "B", True])
        assert len(list(enumerate(instance))) == 2
