from typing import Iterable, Dict, Callable
from collections import defaultdict
from itertools import groupby
from math import log


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


def entropy(items: Iterable, get_relative_value: Callable) -> float:
    entropy = 0
    if items:
        key_groups = groupby(items, get_relative_value)
        for key, group in key_groups:
            p = len(list(group)) / len(items)
            if p != 0 or p != 1:
                entropy += -p*log(p, 2)
    return entropy

def gini(items: Iterable, get_relative_value: Callable) -> float:
    gini = 1
    if items:
        key_groups = groupby(items, get_relative_value)
        for key, group in key_groups:
            p = len(list(group)) / len(items)
            if p != 0 or p != 1:
                gini += -p*p
    return gini
