from typing import Iterable, Dict
from collections import defaultdict

class Instance(list):
    target_attribute_idx = -1

    def __iter__(self):
        return (i for i in self[:self.target_attribute_idx])


class LearningAlgorithm(object):
    def train(self, instances: Iterable[Instance]):
        pass

    def predict(self, instance: Instance) -> object:
        pass

def get_attribute_values_map(instances: Iterable[Instance]) -> Dict[int, object]:
    values_map = defaultdict(set)
    for instance in instances:
        for idx, attribute in enumerate(instance):
            values_map[idx].add(attribute)
    return values_map

def get_most_common_value(values: Iterable[object]) -> object:
    return max(set(values), key=values.count)