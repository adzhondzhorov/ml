from unittest import TestCase

from app.base import Instance
from app.neural_networks.perceptron import Perceptron, LearningRule

instances = (Instance([0, 1, -1]),
             Instance([1, 0, -1]),
             Instance([0, 0, -1]),
             Instance([1, 1, 1]),)


class PerceptronAcceptanceTests(TestCase):
    error = 0.01

    def test_perceptron_rule(self):
        self._test(LearningRule.perceptron_rule)
        
    def test_delta_rule(self):
        self._test(LearningRule.delta_rule)

    def test_stochastic_delta_rule(self):
        self._test(LearningRule.stochastic_delta_rule)

    def _test(self, learning_rule):
        perceptron = Perceptron(len(instances[0]), learning_rule=learning_rule)
        perceptron.train(instances)

        for i in instances:
            assert perceptron.predict(i) == i[Instance.target_attribute_idx]

