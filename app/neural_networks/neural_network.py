import random
from typing import Iterable
from functools import reduce

import numpy as np

from app.base import Instance, LearningAlgorithm
from app.neural_networks.base import sigmoid

TRESHHOLD = 0.01


class NeuralNetwork(LearningAlgorithm):
    def __init__(self, layers: Iterable[int], ni:float=1):
        self.model = [np.matrix(np.random.rand(layers[i-1] + 1, layers[i]))/20 for i in range(1, len(layers))]
        self.layers = layers
        self.ni = ni

    def train(self, instances: Iterable[Instance]):
        while True:
            for x in instances:
                t = x[Instance.target_attribute_idx]
                outputs = self._feedforward(x)
                o = outputs[-1]
                
                delta = reduce(np.multiply, [o, 1 - o, t - o])
                for i in range(1, len(self.model) + 1):
                    sum_delta = delta * self.model[-i][1:].T
                    
                    output = outputs[-1 - i]
                    output_with_ones = NeuralNetwork._pad_ones(output)
                    delta_w = self.ni * output_with_ones.T * delta
                    self.model[-i] += delta_w
                    
                    delta = reduce(np.multiply, [output, 1 - output, sum_delta])
            
            if self._get_error(instances) < TRESHHOLD:   
                return

    def _get_error(self, instances: Iterable[Instance]) -> float:
        error = 0
        for x in instances:
            t = x[Instance.target_attribute_idx]
            outputs = self._feedforward(x)
            o = outputs[-1]
            error += np.power(t - o, 2).sum() / 2
        return error

    def _feedforward(self, x: Instance) -> Iterable[np.matrix]:
        outputs = [np.matrix(x[:Instance.target_attribute_idx])]
        for weight_matrix in self.model:
            prod = outputs[-1]
            prod = NeuralNetwork._pad_ones(prod)
            output = sigmoid(prod * weight_matrix)
            outputs.append(output)

        return outputs

    def predict(self, x: Instance) -> np.matrix:
        outputs = self._feedforward(x)
        return outputs[-1]
    
    @staticmethod
    def _pad_ones(m: np.matrix) -> np.matrix:
        ones = np.ones(m.shape[0])
        return np.insert(m, 0, ones, axis=1)
    
