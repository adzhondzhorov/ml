from typing import Iterable

from collections import defaultdict

from app.base import Instance, LearningAlgorithm, get_most_common_value


class SequentialCovering(LearningAlgorithm):
    def __init__(self, len_attributes), performance_theshold: float=0.6):
        self.len_attributes = len_attributes
        self.performance_theshold = performance_theshold

    def train(self, instances: Iterable[Instance]):
        learned_rules = list()
        most_general_rule = [None] * self.len_attributes
        learned_rules.append(most_general_rule)
        examples = list(instances)
        rule = _learn_one_rule(examples)
        while _performance(rule) > self.performance_theshold:
            learned_rules.append(rule)
            examples = [e for e in examples if _rule_predict(rule) != e[Instance.target_attribute_idx]]
            rule = _learn_one_rule(examples)
        

    def predict(self, x: Instance) -> object:
        pass
    
    def _rule_predict(rule: Iterable[object]):
        pass
    def _performance(rule: Iterable[object]):
        pass
        
    def _remove_duplicates(rules: Iterable[object]):
        pass
    
    def _remove_inconsistent(rules: Iterable[object]):
        pass
    
    def _remove_more_general(rules: Iterable[object]):
        pass

    def _learn_one_rule(examples: Iterable[Instance]):
        passs