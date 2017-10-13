from typing import Iterable

from collections import defaultdict

from app.base import Instance, LearningAlgorithm, get_most_common_value
from app.rule_based_learning.base import get_performance
from app.rule_based_learning.one_rule_beam_search import OneRuleBeamSearch

class SequentialCovering(LearningAlgorithm):
    def __init__(self, len_attributes: int, performance_theshold: float=-0.1):
        self.len_attributes = len_attributes
        self.performance_theshold = performance_theshold
        self.one_rule_learner =  OneRuleBeamSearch(len_attributes)

    def train(self, instances: Iterable[Instance]):
        learned_rules = list()

        examples = list(instances)
        one_rule_learner = OneRuleBeamSearch(self.len_attributes)
        rule = one_rule_learner.get_rule(examples)
        while get_performance(rule, examples) > self.performance_theshold:
            learned_rules.append(rule)

            examples = [e for e in examples if rule.predict(e) != e[Instance.target_attribute_idx]]
            if not examples:
                break
            one_rule_learner = OneRuleBeamSearch(self.len_attributes)
            rule = one_rule_learner.get_rule(examples)

        sorted_rules = sorted(learned_rules, key=lambda r: get_performance(r, instances), reverse=True)
        self.model = sorted_rules
        

    def predict(self, x: Instance) -> object:
        for rule in self.model:
            prediction = rule.predict(x)
            if prediction is not None:
                return prediction
            
    