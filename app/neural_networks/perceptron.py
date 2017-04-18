import random
from typing import Iterable

from app.base import LearningAlgorithm, Instance
from app.neural_networks.base import sgn


class Perceptron(LearningAlgorithm):
    def __init__(self, len_attributes: int, ni: float=0.1):
        self.model = [random.random() for i in range(len_attributes)]
        self.len_attributes = len_attributes
        self.ni = ni

    def train(self, instances: Iterable[Instance]):
        change = True
        while change:
            change = False
            for x in instances:
                o = self.predict(x)
                t = x[self.len_attributes - 1]
                diff = t - o
                if diff:
                    change = True
                x_new = [1] + x

                for i in range(len(self.model)):
                    delta_w_i = self.ni * diff * x_new[i]
                    self.model[i] += delta_w_i

    def predict(self, x: Instance) -> int:
        x_new = [1] + x
        vec_prod = sum(x_new[i] * self.model[i] for i in range(len(self.model)))
        return sgn(vec_prod)


