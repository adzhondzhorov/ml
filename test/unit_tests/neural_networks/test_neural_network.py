from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from app.base import Instance
from app.neural_networks.neural_network import NeuralNetwork


class NeuralNetworkUnitTests(TestCase):
    error = 0.01

    def test_predict(self):
        nn = NeuralNetwork([2, 2, 1])
        nn.model = [np.matrix([[0.5, 0.4],
                               [0.1, 0.2],
                               [0.2, 0.3]]),
                    np.matrix([[0.2],
                               [0.6],
                               [0.4]])]
        instance = Instance([0, 0, 1])
        res = nn.predict(instance)
        assert res.shape == (1, 1)
        assert abs(res.item(0, 0) - 0.6927) < NeuralNetworkUnitTests.error

    def test_feedforward(self):
        nn = NeuralNetwork([2, 2, 1])
        nn.model = [np.matrix([[0.5, 0.4],
                               [0.1, 0.2],
                               [0.2, 0.3]]),
                    np.matrix([[0.2],
                               [0.6],
                               [0.4]])]
        instance = Instance([0, 0, 1])
        res = nn._feedforward(instance)
        assert len(res) == 3
        assert res[1].shape == (1, 2)
        assert res[2].shape == (1, 1)
        assert abs(res[1].item(0, 0) - 0.6224) < NeuralNetworkUnitTests.error
        assert abs(res[1].item(0, 1) - 0.5986) < NeuralNetworkUnitTests.error
        assert abs(res[2].item(0, 0) - 0.6927) < NeuralNetworkUnitTests.error


    def test_pad_ones_single_row(self):
        res = NeuralNetwork._pad_ones(np.matrix([2, 3]))
        assert (res == np.matrix([1, 2, 3])).all()

    
    def test_pad_ones_single_col(self):
        res = NeuralNetwork._pad_ones(np.matrix([[2], [3]]))
        assert (res == np.matrix([[1, 2], [1 , 3]])).all()
            
        
    def test_pad_ones_full_matrix(self):
        res = NeuralNetwork._pad_ones(np.matrix([[2, 3, 4],[3, 4, 5]]))
        assert (res == np.matrix([[1, 2, 3, 4], [1, 3, 4, 5]])).all()

    def test_get_error(self):
        nn = NeuralNetwork([2, 2, 1])
        nn._feedforward = Mock()
        nn._feedforward.side_effect = [np.matrix([1, 0]),
                                       np.matrix([4, 3]), 
                                       np.matrix([3,3])]

        instances = [Instance([110, 20, [1, 0]]),
                     Instance([5, 60, [2, 2]]),
                     Instance([5, 60, [4, 3]])]
        res = nn._get_error(instances)
        assert res == 3
