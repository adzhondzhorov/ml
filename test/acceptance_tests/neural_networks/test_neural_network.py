from unittest import TestCase

from app.base import Instance
from app.neural_networks.neural_network import NeuralNetwork

instances = (Instance([1, 0, 0, 0, [1, 0, 0, 0]]),
             Instance([0, 1, 0, 0, [0, 1, 0, 0]]),
             Instance([0, 0, 1, 0, [0, 0, 1, 0]]),
             Instance([0, 0, 0, 1, [0, 0, 0, 1]]))

class NeuralNetworkAcceptanceTests(TestCase):
    error = 0.1

    def test_neural_network(self):
        nn = NeuralNetwork([4, 3, 4])
        nn.train(instances)

        for i in instances:
            assert (abs(nn.predict(i) - i[Instance.target_attribute_idx]) < NeuralNetworkAcceptanceTests.error).all()

