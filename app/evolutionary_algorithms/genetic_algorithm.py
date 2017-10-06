import math
import random
from typing import Iterable, Dict, Tuple

import numpy as np

from app.base import Instance, LearningAlgorithm, get_attribute_values_map


class Model(object):
    def __init__(self, index_map:  Iterable[Tuple[int, object]], hypotesis: str):
        self.index_map = index_map
        self.hypotesis = hypotesis

class GeneticAlgorithm(LearningAlgorithm):
    def __init__(self, population_size: int=10, crossover_fraction: float=0.6, mutation_rate: float=0.1, threshold: float=0.9):
        self.model = None
        self.population_size = population_size
        self.crossover_fraction = crossover_fraction
        self.mutation_rate = mutation_rate
        self.threshold = threshold

    def train(self, instances: Iterable[Instance]):
        index_map = self._build_attribute_values_index_map(instances)
        self.model = Model(index_map, None)
        
        crossover_size = math.ceil(self.crossover_fraction * self.population_size)
        crossover_size = crossover_size - 1 if crossover_size % 2 else crossover_size
        mutation_size = math.ceil(self.mutation_rate  * self.population_size)

        population = self._generate_population(len(index_map))
        population_fitness = [self._get_fitness(i, instances) for i in population]
        while max(population_fitness) < self.threshold:
            selection_probability = [
                fitness / sum(population_fitness) for fitness in population_fitness]
            new_population = list()
            for i in range(self.population_size - crossover_size):
                new_population.append(np.random.choice(
                    population, p=selection_probability))

            for i in range(int(crossover_size / 2)):
                parent1 = np.random.choice(population, p=selection_probability)
                parent2 = np.random.choice(population, p=selection_probability)
                child1, child2 = self._crossover(parent1, parent2)
                new_population.append(child1)
                new_population.append(child2)

            population = new_population

            for i in range(mutation_size):
                individual = np.random.choice(population)
                new_population.remove(individual)
                new_individual = self._mutate(individual)
                new_population.append(new_individual)

            population_fitness = [self._get_fitness(i, instances) for i in population]

        self.model.hypotesis = max(population, key=lambda p: population_fitness[population.index(p)])

    def _build_attribute_values_index_map(self, instances: Iterable[Instance]) -> Iterable[Tuple[int, object]]:
        attribute_values_map = get_attribute_values_map(instances)
        index_map = list()
        for idx, values in attribute_values_map.items():
            for value in values:
                index_map.append((idx, value))
        return index_map

    def predict(self, x: Instance) -> object:
        return self._get_target_value(self.model.hypotesis, x)

    def _generate_population(self, individual_len: int):
        population = list()
        for i in range(self.population_size):
           population.append("".join([random.choice("01")
                             for i in range(individual_len)]))

        return population

    def _crossover(self, parent1:str, parent2: str) -> Tuple[str, str]:
        split = np.random.choice(range(1, len(parent1) - 1))
        return parent1[:split] + parent2[split:], parent2[:split] + parent1[split:]

    def _mutate(self, individual:str) -> str:
        idx = np.random.choice(range(len(individual)))
        mutation = "1" if individual[idx] == "0" else "0"

        return individual[:idx] + mutation + individual[idx + 1:]

    def _get_fitness(self, individual: str, instaces: Iterable[Instance]) -> float:
        correct_instances = [i for i in instaces
                             if self._get_target_value(individual, i) == i[Instance.target_attribute_idx]]
        return len(correct_instances) / len(instaces)

    def _get_target_value(self, individual, instance) -> bool:
        for idx, value in enumerate(instance):
            bit = self.model.index_map.index((idx, value)) 
            if individual[bit] == "0":
                return False
        return True
