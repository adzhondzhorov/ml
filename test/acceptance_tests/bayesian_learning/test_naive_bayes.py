from unittest import TestCase

from app.base import Instance
from app.bayesian_learning.naive_bayes import NaiveBayes
    
instances = (Instance(["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", True]),
             Instance(["Sunny", "Warm", "High", "Strong", "Warm", "Same", True]),
             Instance(["Rainy", "Warm", "High", "Strong",
                       "Warm", "Change", False]),
             Instance(["Rainy", "Cold", "Normal", "Strong", "Cool", "Change", False]))

class NaiveBayesAcceptanceTests(TestCase):
    def test_naive_bayes(self):
        nb = NaiveBayes(len(instances[0]), len(instances))
        nb.train(instances)
        for i in instances:
            assert nb.predict(i) == i[Instance.target_attribute_idx]