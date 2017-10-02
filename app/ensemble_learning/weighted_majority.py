from typing import Iterable

from collections import defaultdict

from app.base import Instance, LearningAlgorithm, get_most_common_value


class WeightedMajority(LearningAlgorithm):
    def __init__(self, learners: Iterable[LearningAlgorithm], beta: float=0.8):
        self.learners = learners
        self.beta = beta
        self.model = {learner: 1 for learner in learners}

    def train(self, instances: Iterable[Instance]):
        for learner in self.learners:
            learner.train(instances)

        values = set(instance[Instance.target_attribute_idx] for instance in instances)
        for instance in instances:
            votes = dict()
            for learner in self.learners:                
                votes[learner] = learner.predict(instance)
            prediction = get_most_common_value(list(votes.values()))
            for learner in self.learners:
                if votes[learner] != prediction: 
                    self.model[learner] *= beta

    def predict(self, x: Instance) -> object:
        weighted_value_votes = defaultdict(lambda: 0)
        for learner in self.learners:
            weighted_value_votes[learner.predict(x)] += self.model[learner]
        return max(weighted_value_votes)
