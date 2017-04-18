from unittest import TestCase
from unittest.mock import patch

from app.concept_learning.base import All, Hypothesis, ConceptInstance, Instance
from app.concept_learning.hypothesis_search import _is_attribute_constraint_satisfied, \
    _generalize_hypothesis_by_attribute, _generalize_hypothesis, _specify_hypothesis_by_attribute, \
    _specify_hypothesis, _is_hypothesis_consistent, _get_attribute_values_map, FindS, CandidateElimination


class HypothesisSearchUnitTests(TestCase):
    @patch("app.concept_learning.hypothesis_search._is_attribute_constraint_satisfied")
    @patch("app.concept_learning.hypothesis_search._generalize_hypothesis_by_attribute")
    def test_find_s_general(self, generalize_hypothesis_mock, is_satisfied_mock):
        instances = [ConceptInstance(["val1", "val2", True]),
                     ConceptInstance(["val1", "val3", True])]

        is_satisfied_mock.side_effect = [False, True] + [False, False]
        generalize_hypothesis_mock.return_value = Hypothesis([None] * 2)
        find_s = FindS(3)
        find_s.train(instances)

        assert generalize_hypothesis_mock.call_count == 3

    @patch("app.concept_learning.hypothesis_search._is_attribute_constraint_satisfied")
    @patch("app.concept_learning.hypothesis_search._generalize_hypothesis_by_attribute")
    def test_find_s_negative_examples(self, generalize_hypothesis_mock, is_satisfied_mock):
        instances = [ConceptInstance(["val1", "val2", True]),
                     ConceptInstance(["val1", "val3", False])]

        is_satisfied_mock.side_effect = [False, True] + [False, False]
        generalize_hypothesis_mock.return_value = Hypothesis([None] * 2)
        find_s = FindS(3)
        find_s.train(instances)

        assert generalize_hypothesis_mock.call_count == 1


    @patch("app.concept_learning.hypothesis_search._is_hypothesis_consistent")
    @patch("app.concept_learning.hypothesis_search._generalize_hypothesis")
    @patch("app.concept_learning.hypothesis_search._specify_hypothesis")
    def test_candidate_elimination(self, specify_hypothesis_mock, generalize_hypothesis_mock, is_consistent_mock):
        instances = [ConceptInstance(["val1", "val2", True]),
                     ConceptInstance(["val2", "val3", False])]

        is_consistent_mock.side_effect = [False, True, False, True]
        min_hypotheses = {Hypothesis(["val1", "val2"])}
        max_hypotheses = {Hypothesis(["val2", All]), Hypothesis([All, "val3"])}
        generalize_hypothesis_mock.return_value = min_hypotheses
        specify_hypothesis_mock.return_value = max_hypotheses

        candidate_elimination = CandidateElimination(3)
        candidate_elimination.train(instances)
        S, G = candidate_elimination.model

        assert S == min_hypotheses
        assert G == max_hypotheses


    @patch("app.concept_learning.hypothesis_search._is_hypothesis_consistent")
    def test_candidate_elimination(self, specify_hypothesis_mock, generalize_hypothesis_mock, is_consistent_mock):
        instances = [ConceptInstance(["val1", "val2", True]),
                     ConceptInstance(["val2", "val3", False])]

        is_consistent_mock.side_effect = [True, True, True, True]

        candidate_elimination = CandidateElimination(3)
        candidate_elimination.train(instances)
        S, G = candidate_elimination.model

        assert S == Hypothesis([None] * 2)
        assert G == Hypothesis([All] * 2)


    @patch("app.concept_learning.hypothesis_search._is_hypothesis_consistent")
    @patch("app.concept_learning.hypothesis_search._generalize_hypothesis")
    @patch("app.concept_learning.hypothesis_search._specify_hypothesis")
    def test_candidate_elimination(self, specify_hypothesis_mock, generalize_hypothesis_mock, is_consistent_mock):
        instances = [ConceptInstance(["val1", "val2", True]),
                     ConceptInstance(["val2", "val3", False])]

        is_consistent_mock.side_effect = [False, False, False, False]
        min_hypotheses = {Hypothesis(["val1", "val2"])}
        max_hypotheses = {Hypothesis(["val2", All]), Hypothesis([All, "val3"])}
        generalize_hypothesis_mock.return_value = min_hypotheses
        specify_hypothesis_mock.return_value = max_hypotheses

        candidate_elimination = CandidateElimination(3)
        candidate_elimination.train(instances)
        S, G = candidate_elimination.model

        assert S == set()
        assert G == set()

    def test_is_attribute_constraint_satisfied(self):
        instance = ConceptInstance(["val1", "val2", True])
        assert _is_attribute_constraint_satisfied(instance, "val1", 0)
        assert not _is_attribute_constraint_satisfied(instance, "val1", 1)
        assert not _is_attribute_constraint_satisfied(instance, None, 0)
        assert _is_attribute_constraint_satisfied(instance, All, 0)
        assert not _is_attribute_constraint_satisfied(instance, "val1", 2)
        assert not _is_attribute_constraint_satisfied(instance, "val1", 100)
        assert not _is_attribute_constraint_satisfied(instance, "val1", -1)

    def test_generalize_hypothesis_by_attribute(self):
        hypothesis = Hypothesis(["val1", None, All])
        assert _generalize_hypothesis_by_attribute(hypothesis, "val1", 0) == hypothesis
        assert _generalize_hypothesis_by_attribute(hypothesis, "val1", 1) == \
               Hypothesis(["val1", "val1", All])
        assert _generalize_hypothesis_by_attribute(hypothesis, "val1", 2) == hypothesis
        assert _generalize_hypothesis_by_attribute(hypothesis, "val2", 0) == \
               Hypothesis([All, None, All])
        assert _generalize_hypothesis_by_attribute(hypothesis, "val2", 100) == hypothesis
        assert _generalize_hypothesis_by_attribute(hypothesis, "val2", -1) == hypothesis

    def test_specify_hypothesis_by_attribute(self):
        hypothesis = Hypothesis(["val1", None, All])
        assert _specify_hypothesis_by_attribute(hypothesis, "val1", 0) == hypothesis
        assert _specify_hypothesis_by_attribute(hypothesis, "val1", 1) == hypothesis
        assert _specify_hypothesis_by_attribute(hypothesis, "val1", 2) == \
               Hypothesis(["val1", None, "val1"])
        assert _specify_hypothesis_by_attribute(hypothesis, "val2", 0) == \
               Hypothesis([None, None, All])
        assert _specify_hypothesis_by_attribute(hypothesis, "val2", 100) == hypothesis
        assert _specify_hypothesis_by_attribute(hypothesis, "val2", -1) == hypothesis

    def test_generalize_hypothesis(self):
        hypothesis = Hypothesis(["val1", None, "val3", All])
        instance = ConceptInstance(["val2", "val2", "val3", "val4", True])
        hypotheses = _generalize_hypothesis(hypothesis, instance)
        assert hypotheses == {Hypothesis([All, "val2", "val3", All])}

    def test_specify_hypothesis(self):
        hypothesis = Hypothesis(["val1", All, All])
        instance = ConceptInstance(["val1", "val2", "val3", True])
        attribute_values_map = {0: {"val1", "val2"},
                                1: {"val1", "val2"},
                                2: {"val1", "val2", "val3"}}
        hypotheses = _specify_hypothesis(hypothesis, instance, attribute_values_map)
        assert hypotheses == {Hypothesis(["val1", "val1", All]),
                              Hypothesis(["val1", All, "val1"]),
                              Hypothesis(["val1", All, "val2"])}

    def test_specify_hypothesis_empty(self):
        hypothesis = Hypothesis(["val1", "val2"])
        instance = ConceptInstance(["val1", "val2", True])
        attribute_values_map = {0: {"val1", "val2"},
                                1: {"val2"}}
        hypotheses = _specify_hypothesis(hypothesis, instance, attribute_values_map)
        assert hypotheses == set()

    @patch("app.concept_learning.hypothesis_search._is_attribute_constraint_satisfied")
    def test_is_hypothesis_consistent_positive_true(self, is_satisfied_mock):
        is_satisfied_mock.side_effect = [True, True, True]
        hypothesis = Hypothesis(["val1", None, "val3"])
        instance = ConceptInstance(["val1", "val2", "val3", True])
        res = _is_hypothesis_consistent(hypothesis, instance)
        assert res

    @patch("app.concept_learning.hypothesis_search._is_attribute_constraint_satisfied")
    def test_is_hypothesis_consistent_positive_false(self, is_satisfied_mock):
        is_satisfied_mock.side_effect = [True, False, True]
        hypothesis = Hypothesis(["val1", None, "val3"])
        instance = ConceptInstance(["val1", "val2", "val3", True])
        res = _is_hypothesis_consistent(hypothesis, instance)
        assert not res

    @patch("app.concept_learning.hypothesis_search._is_attribute_constraint_satisfied")
    def test_is_hypothesis_consistent_negative_false(self, is_satisfied_mock):
        is_satisfied_mock.side_effect = [True, True, True]
        hypothesis = Hypothesis(["val1", None, "val3"])
        instance = ConceptInstance(["val1", "val2", "val3", False])
        res = _is_hypothesis_consistent(hypothesis, instance)
        assert not res

    @patch("app.concept_learning.hypothesis_search._is_attribute_constraint_satisfied")
    def test_is_hypothesis_consistent_negative_true(self, is_satisfied_mock):
        is_satisfied_mock.side_effect = [True, False, True]
        hypothesis = Hypothesis(["val1", None, "val3"])
        instance = ConceptInstance(["val1", "val2", "val3", False])
        res = _is_hypothesis_consistent(hypothesis, instance)
        assert res

    def test_get_attribute_values_map(self):
        instances = [Instance(["val1", "val2", True]),
                     Instance(["val2", "val2", True]),
                     Instance(["val3", "val3", False])]
        values_map = _get_attribute_values_map(instances)

        assert values_map == {0: {"val1", "val2", "val3"}, 1: {"val2", "val3"}}
