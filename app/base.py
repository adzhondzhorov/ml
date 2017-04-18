from typing import Iterable


class Instance(list):
    target_attribute_idx = -1

    def __iter__(self):
        return (i for i in self[:self.target_attribute_idx])


class LearningAlgorithm(object):
    def train(self, instances: Iterable[Instance]):
        pass

    def predict(self, instance: Instance) -> object:
        pass