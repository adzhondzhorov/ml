from unittest import TestCase

from app.concept_learning.base import ConceptInstance, Hypothesis, All
from app.concept_learning.hypothesis_search import find_s, candidate_elimination

instances = (ConceptInstance(["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", True]),
             ConceptInstance(["Sunny", "Warm", "High", "Strong", "Warm", "Same", True]),
             ConceptInstance(["Rainy", "Cold", "High", "Strong", "Warm", "Change", False]),
             ConceptInstance(["Sunny", "Warm", "High", "Strong", "Cool", "Change", True]))

class HypothesisSearchAcceptanceTests(TestCase):
    def test_find_s(self):
        hypothesis = find_s(instances, len(instances[0]))

        assert hypothesis == Hypothesis(["Sunny", "Warm", All, "Strong", All, All])

    def test_candidate_elimination(self):
        S, G = candidate_elimination(instances, len(instances[0]))

        assert S == {Hypothesis(["Sunny", "Warm", All, "Strong", All, All])}
        assert G == {Hypothesis(["Sunny", All, All, All, All, All]), Hypothesis([All, "Warm", All, All, All, All])}
