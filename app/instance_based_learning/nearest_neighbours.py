from typing import Iterable

from collections import defaultdict

from app.base import Instance, LearningAlgorithm, get_most_common_value
from app.instance_based_learning.base import get_distance

class KNearestNeighbours(LearningAlgorithm):
    def __init__(self, k: int=4):
        self.k = k
        self.model = None

    def train(self, instances: Iterable[Instance]):
        self.model = instances

    def predict(self, x: Instance) -> object:
        nearest_neighbours = self._get_nearest_neighbours(x)
        return get_most_common_value([neighbour[Instance.target_attribute_idx] for neighbour in nearest_neighbours])

    def _get_nearest_neighbours(self, x: Instance):
        sorted_neighbours = sorted([instance for instance in self.model], key=lambda i: get_distance(i, x),
                                   reverse=True)
        return sorted_neighbours[:self.k]
