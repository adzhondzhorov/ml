from unittest import TestCase

from app.base import Instance
from app.neural_networks.perceptron import Perceptron, LearningRule


class PerceptronUnitTests(TestCase):
    def test_predict_zero(self):
        perceptron = Perceptron(3, learning_rule=LearningRule.perceptron_rule)
        perceptron.model = [0.5, -1, 1]        
        instance = Instance([2, 1.5, -10])
        assert perceptron.predict(instance) == -1

    def test_predict_positive(self):
        perceptron = Perceptron(3, learning_rule=LearningRule.perceptron_rule)
        perceptron.model = [0.5, -1, 2]        
        instance = Instance([2, 1.5, 10])
        assert perceptron.predict(instance) == 1

    def test_predict_negative(self):
        perceptron = Perceptron(3, learning_rule=LearningRule.perceptron_rule)
        perceptron.model = [0.5, -1, 2]        
        instance = Instance([3, 1, -10])
        assert perceptron.predict(instance) == -1

    def test_predict_unbounded(self):
        perceptron = Perceptron(3, learning_rule=LearningRule.perceptron_rule)
        perceptron.model = [0.5, -1, 2]        
        instance = Instance([2, 1.5, -10])
        assert perceptron.predict_unbounded(instance) == 1.5

    def test_predict_unbounded_negative(self):
        perceptron = Perceptron(3, learning_rule=LearningRule.perceptron_rule)
        perceptron.model = [0.5, -1, 2]        
        instance = Instance([3, 1, 10])
        assert  perceptron.predict_unbounded(instance) == -0.5
