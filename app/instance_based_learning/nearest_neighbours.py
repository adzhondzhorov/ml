from typing import Iterable

from collections import defaultdict

from app.base import Instance, LearningAlgorithm, get_most_common_value


class KNearestNeighbours(LearningAlgorithm):
    def __init__(self, k: int=4):
        self.k = k
        self.model = None

    def train(self, instances: Iterable[Instance]):
        self.model = instances

    def predict(self, x: Instance) -> object:
        weighted_value_votes = defaultdict(lambda: 0)
        for learner in self.learners:
            weighted_value_votes[learner.predict(x)] += self.model[learner]
        return max(weighted_value_votes)
    
    @staticmethod
    def _get_distance(a: Instance, b: Instance):
        pass
