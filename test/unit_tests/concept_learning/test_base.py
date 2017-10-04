from unittest import TestCase

from app.concept_learning.base import All, Hypothesis


class TestHypothesis(TestCase):
    def test_compare(self):
        h1 = Hypothesis([None, None])
        h2 = Hypothesis(["val1", None])
        h3 = Hypothesis(["val1", "val2"])
        h4 = Hypothesis(["val1", All])
        h5 = Hypothesis([All, All])
        h6 = Hypothesis(["val2", All])
        assert h1 < h2 < h3 < h4 < h5
        assert not h3 < h6
        assert not h6 < h3
        assert not h6 < h4
        assert not h4 < h6
                
    def test_compare_attribute_constraints_less(self):
        assert Hypothesis._compare_attribute_constraints(None, "val1", Hypothesis.LESS)
        assert not Hypothesis._compare_attribute_constraints("val1", None, Hypothesis.LESS)
        assert Hypothesis._compare_attribute_constraints(None, None, Hypothesis.LESS)
        assert Hypothesis._compare_attribute_constraints(All, All, Hypothesis.LESS)
        assert Hypothesis._compare_attribute_constraints("val1", "val1", Hypothesis.LESS)
        assert Hypothesis._compare_attribute_constraints("val1", All, Hypothesis.LESS)
        assert not Hypothesis._compare_attribute_constraints(All, "val1", Hypothesis.LESS)
        assert not Hypothesis._compare_attribute_constraints("val1", "val2", Hypothesis.LESS)

    def test_compare_attribute_greater(self):
        assert not Hypothesis._compare_attribute_constraints(None, "val1", Hypothesis.GREATER)
        assert Hypothesis._compare_attribute_constraints("val1", None, Hypothesis.GREATER)
        assert Hypothesis._compare_attribute_constraints(None, None, Hypothesis.GREATER)
        assert Hypothesis._compare_attribute_constraints(All, All, Hypothesis.GREATER)
        assert Hypothesis._compare_attribute_constraints("val1", "val1", Hypothesis.GREATER)
        assert not Hypothesis._compare_attribute_constraints("val1", All, Hypothesis.GREATER)
        assert Hypothesis._compare_attribute_constraints(All, "val1", Hypothesis.GREATER)
        assert not Hypothesis._compare_attribute_constraints("val1", "val2", Hypothesis.GREATER)
