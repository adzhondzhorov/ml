import random
from typing import Iterable
from collections import defaultdict
from enum import Enum 

from app.base import Instance, LearningAlgorithm
from app.neural_networks.base import sgn

LearningRule = Enum("LearningRule", "perceptron_rule delta_rule stochastic_delta_rule")

EPSILON = 0.01


class Perceptron(LearningAlgorithm):
    def __init__(self, len_attributes: int, ni: float=0.1, learning_rule: LearningRule=LearningRule.perceptron_rule):
        self.model = [random.random() for i in range(len_attributes)]
        self.len_attributes = len_attributes
        self.ni = ni
        self.learning_rule = learning_rule

    def train(self, instances: Iterable[Instance]):
        if self.learning_rule == LearningRule.perceptron_rule:
            self.train_perceptron_rule(instances)
        elif self.learning_rule == LearningRule.delta_rule:
            self.train_delta_rule(instances)
        elif self.learning_rule == LearningRule.stochastic_delta_rule:
            self.train_stochastic_delta_rule(instances)

    def train_perceptron_rule(self, instances: Iterable[Instance]):
        change = True
        while change:
            change = False
            delta_w = defaultdict(lambda: 0)
            for x in instances:
                o = self.predict(x)
                t = x[Instance.target_attribute_idx]
                diff = t - o
                if diff:
                    change = True
                x_new = [1] + x
                for i in range(len(self.model)):
                    delta_w[i] += self.ni * diff * x_new[i]
                    
            for i in range(len(self.model)):
                self.model[i] += delta_w[i]
    
    def train_delta_rule(self, instances: Iterable[Instance]):
        while True:
            old_model = list(self.model)
            delta_w = defaultdict(lambda: 0)
            for x in instances:
                o = self.predict_unbounded(x)
                t = x[Instance.target_attribute_idx]
                diff = t - o
                x_new = [1] + x
                for i in range(len(self.model)):
                    delta_w[i] += self.ni * diff * x_new[i]
            
            for i in range(len(self.model)):
                self.model[i] += delta_w[i]
            if all(abs(w[0]-w[1]) < EPSILON for w in zip(self.model, old_model)):
                return        
    
    def train_stochastic_delta_rule(self, instances: Iterable[Instance]):
        while True:
            old_model = list(self.model)
            for x in instances:
                o = self.predict_unbounded(x)
                t = x[Instance.target_attribute_idx]
                diff = t - o
                x_new = [1] + x
                for i in range(len(self.model)):
                    delta_w = self.ni * diff * x_new[i]
                    self.model[i] += delta_w

            if all(abs(w[0]-w[1]) < EPSILON for w in zip(self.model, old_model)):
                return        

    def predict(self, x: Instance) -> int:
        return sgn(self.predict_unbounded(x))

    def predict_unbounded(self, x: Instance) -> int:
        x_new = [1] + x
        vec_prod = sum(x_new[i] * self.model[i]
                       for i in range(len(self.model)))
        return vec_prod


instances = (Instance([0, 1, -1]),
             Instance([1, 0, -1]),
             Instance([0, 0, -1]),
             Instance([1, 1, 1]),)

perceptron = Perceptron(len(instances[0]), learning_rule=LearningRule.stochastic_delta_rule)
perceptron.train(instances)
