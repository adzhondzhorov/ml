from unittest import TestCase

from app.concept_learning.base import ConceptInstance, Hypothesis, All
from app.concept_learning.hypothesis_search import FindS, CandidateElimination

instances = (ConceptInstance(["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", True]),
             ConceptInstance(["Sunny", "Warm", "High", "Strong", "Warm", "Same", True]),
             ConceptInstance(["Rainy", "Cold", "High", "Strong", "Warm", "Change", False]),
             ConceptInstance(["Sunny", "Warm", "High", "Strong", "Cool", "Change", True]))

class HypothesisSearchAcceptanceTests(TestCase):
    def test_find_s(self):
        find_s = FindS(len(instances[0]))
        find_s.train(instances)

        assert find_s.model == Hypothesis(["Sunny", "Warm", All, "Strong", All, All])
        assert find_s.predict(ConceptInstance(["Sunny", "Warm", "Low", "Strong", "Cool", "Change", None]))
        assert find_s.predict(ConceptInstance(["Sunny", "Warm", "High", "Strong", "Warm", "Same", None]))
        assert not find_s.predict(ConceptInstance(["Rainy", "Warm", "High", "Strong", "Cool", "Change", None]))
        assert not find_s.predict(ConceptInstance(["Rainy", "Cold", "High", "Strong", "Warm", "Change", None]))

    def test_candidate_elimination(self):
        candidate_elimination = CandidateElimination(len(instances[0]))
        candidate_elimination.train(instances)
        S, G = candidate_elimination.model

        assert S == {Hypothesis(["Sunny", "Warm", All, "Strong", All, All])}
        assert G == {Hypothesis(["Sunny", All, All, All, All, All]), Hypothesis([All, "Warm", All, All, All, All])}

        assert candidate_elimination.predict(ConceptInstance(["Sunny", "Warm", "Normal", "Strong", "Cool", "Change", None]))
        assert not candidate_elimination.predict(ConceptInstance(["Rainy", "Cold", "Normal", "Light", "Warm", "Same", None]))

        assert candidate_elimination.predict(ConceptInstance(["Sunny", "Warm", "Normal", "Light", "Warm", "Same", None])) is None
        assert candidate_elimination.predict(ConceptInstance(["Sunny", "Cold", "Normal", "Strong", "Warm", "Same", None])) is None
