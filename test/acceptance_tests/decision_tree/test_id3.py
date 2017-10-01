from unittest import TestCase

from app.concept_learning.base import Instance
from app.decision_tree.id3 import ID3

instances = (Instance(["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", True]),
             Instance(["Sunny", "Warm", "High", "Strong", "Warm", "Same", True]),
             Instance(["Rainy", "Cold", "High", "Strong", "Warm", "Change", False]),
             Instance(["Rainy", "Cold", "High", "Strong", "Cool", "Change", True]),
             Instance(["Rainy", "Cold", "Normal", "Strong", "Cool", "Change", False]))

class ID3AcceptanceTests(TestCase):
    def test_id3(self):
        id3 = ID3(len(instances[0]))
        id3.train(instances)

        for i in instances:
            assert id3.predict(i) == i[-1]