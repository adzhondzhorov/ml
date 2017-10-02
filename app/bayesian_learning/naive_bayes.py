from typing import Iterable, Dict
from collections import defaultdict

from app.base import Instance, LearningAlgorithm, get_attribute_values_map
from app.bayesian_learning.base import get_bayesian_estimate

class Counts(object):
    def __init__(self, all_count: int, value_counts: Dict[object, int], attribute_for_value_counts: Dict[int, Dict[object, Dict[object, int]]]):
        self.all_count = all_count
        self.value_counts = value_counts
        self.attribute_for_value_counts = attribute_for_value_counts

class Model(object):
    def __init__(self, values: Iterable[object], counts: Counts):
        self.values = values
        self.counts = counts

class NaiveBayes(LearningAlgorithm):
    def __init__(self, len_attributes: int, equivalent_sample_size: int=0):
        self.model = None
        self.len_attributes = len_attributes
        self.equivalent_sample_size = equivalent_sample_size

    def train(self, instances: Iterable[Instance]):
        values = set(instance[Instance.target_attribute_idx] for instance in instances)
        counts = Counts(len(instances), self._get_value_counts(instances, values), self._get_attribute_for_value_counts(instances, values))
        self.model = Model(values, counts)

    def predict(self, x: Instance) -> object:
        return max(self.model.values, key=lambda v: self._get_attributes_for_value_probability_product(x, v) * self._get_value_probability(v))

    def _get_value_counts(self, instances: Iterable[Instance], values: Iterable[object]):
        return {value: NaiveBayes._get_count(instances, Instance.target_attribute_idx, value) for value in values}

    def _get_attribute_for_value_counts(self, instances: Iterable[Instance], values: Iterable[object]):
        attribute_for_value_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        attribute_values_map = get_attribute_values_map(instances)
        for attribute_idx in range(self.len_attributes - 1):
            attribute_for_value_counts[attribute_idx] = defaultdict(lambda: defaultdict(lambda: 0))
            for attribute_value in attribute_values_map[attribute_idx]:
                attribute_for_value_counts[attribute_idx][attribute_value] = defaultdict(lambda: 0)
                for value in values:
                    value_instances = [instance for instance in instances if instance[Instance.target_attribute_idx] == value]
                    attribute_for_value_counts[attribute_idx][attribute_value][value] = NaiveBayes._get_count(value_instances, attribute_idx, attribute_value)
        
        return attribute_for_value_counts

    def _get_value_probability(self, value: object):
        return self.model.counts.value_counts[value] / self.model.counts.all_count

    def _get_attributes_for_value_probability_product(self, instance: Instance, value: object):
        probability_product = 1
        for idx, attribute in enumerate(instance):
            prior_estimate = 1 / len(self.model.counts.attribute_for_value_counts[idx])
            probability_estimate = get_bayesian_estimate(self.model.counts.attribute_for_value_counts[idx][attribute][value],
                                                         self.model.counts.value_counts[value],
                                                         self.equivalent_sample_size,
                                                         prior_estimate)
            probability_product *= probability_estimate
        return probability_product


    @staticmethod
    def _get_count(instances: Iterable[Instance], attribute_idx: int, attribute_value: object):
        return len([instance for instance in instances if instance[attribute_idx] == attribute_value])
