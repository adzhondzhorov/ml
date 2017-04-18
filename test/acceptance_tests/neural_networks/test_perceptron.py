from unittest import TestCase

from app.base import Instance
from app.neural_networks.perceptron import Perceptron

instances = (Instance([0, 1, 0]),
             Instance([1, 0, 0]),
             Instance([0, 0, 1]),
             Instance([1, 1, 1]),)


class PerceptronAcceptanceTests(TestCase):
    def test_perceptron_rule(self):
        perceptron = Perceptron(len(instances[0]))
        perceptron.train(instances)

        error = 0.1
        for i in instances:
            assert abs(perceptron.predict(i) - i[len(instances[0])]) < error
