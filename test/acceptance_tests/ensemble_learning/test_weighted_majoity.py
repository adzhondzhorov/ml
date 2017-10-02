from unittest import TestCase

from app.base import Instance
from app.ensemble_learning.weighted_majority import WeightedMajority
from app.bayesian_learning.naive_bayes import NaiveBayes
from app.decision_tree.id3 import ID3
    
instances = (Instance(["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", True]),
             Instance(["Sunny", "Warm", "High", "Strong", "Warm", "Same", True]),
             Instance(["Rainy", "Warm", "High", "Strong",
                       "Warm", "Change", False]),
             Instance(["Rainy", "Cold", "Normal", "Strong", "Cool", "Change", False]))

class WeightedMajorityAcceptanceTests(TestCase):
    def test_weighted_majority(self):
        wm = WeightedMajority([NaiveBayes(len(instances[0]), len(instances)), ID3(len(instances[0]))])
        wm.train(instances)
        for i in instances:
            assert wm.predict(i) == i[Instance.target_attribute_idx]