from unittest import TestCase

from app.base import Instance, get_attribute_values_map


class TestInstance(TestCase):
    def test_iter(self):
        instance = Instance(["A", "B", True])
        assert len(list(enumerate(instance))) == 2

    def test_get_attribute_values_map(self):
        instances = [Instance(["val1", "val2", True]),
                     Instance(["val2", "val2", True]),
                     Instance(["val3", "val3", False])]
        values_map = get_attribute_values_map(instances)

        assert values_map == {0: {"val1", "val2", "val3"}, 1: {"val2", "val3"}}
