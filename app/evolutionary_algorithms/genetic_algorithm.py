import math
import random
from typing import Iterable

from app.base import Instance, LearningAlgorithm, get_attribute_values_map



class GeneticAlgorithm(LearningAlgorithm):
    def __init__(self, len_attributes: int, population_size: int=10):
        self.model = None
        self.len_attributes = len_attributes
        self.population_size = population_size

    def train(self, instances: Iterable[Instance]):
        attribute_values_map = get_attribute_values_map(instances)
        target_values = set([instance[Instance.target_attribute_idx] for instance in instances])
            
        population = self.generate_population(target_values, attribute_values_map)


    def predict(self, x: Instance) -> object:
        pass

    def generate_population(self, target_values, attribute_values_map):
        bit_sequence_length = 0
        for attribute_idx in range(self.len_attributes - 1):
            bit_sequence_length += len(attribute_values_map[attribute_idx])

        bit_sequence_length += math.ceil(math.log(len(target_values), 2))
        population = list()
        for i in range(self.population_size):
           population.append("".join([random.choice("01") for i in range(bit_sequence_length)]))

        return population
        
    def crossover(self):
        pass
    
    def get_fitness(self, instace: Instance):
        pass
