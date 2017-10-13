from typing import Iterable, Set, Dict

from collections import defaultdict

from app.base import Instance, LearningAlgorithm, get_attribute_values_map, get_most_common_value
from app.rule_based_learning.base import Rule, get_performance

class OneRuleBeamSearch(LearningAlgorithm):
    def __init__(self, len_attributes: int, performance_theshold: float=0.6, beam_size: int=4):
        self.len_attributes = len_attributes
        self.beam_size = beam_size
        self.performance_theshold = performance_theshold

    def train(self, instances: Iterable[Instance]):
        attribute_values_map = get_attribute_values_map(instances)
        most_general_rule = Rule([None] * self.len_attributes)
        # using a set will remove duplicates when added
        candidate_rules = set([most_general_rule])
        best_rule = most_general_rule
        while candidate_rules:
            new_candidate_rules = set()
            for rule in candidate_rules:
                new_candidate_rules.update(self._specify(rule, attribute_values_map))
            new_candidate_rules = self._remove_more_general(new_candidate_rules)
            for new_candidate_rule in new_candidate_rules:
                if get_performance(new_candidate_rule, instances) > get_performance(best_rule, instances):
                    best_rule = new_candidate_rule
            sorted_new_candidate_rules = sorted(new_candidate_rules, key=lambda r: get_performance(r, instances), reverse=True)
            candidate_rules = set(sorted_new_candidate_rules[:self.beam_size])
        
        best_rule.prediction = get_most_common_value([i[Instance.target_attribute_idx] for i in instances if best_rule.can_predict(i)])
        self.model = best_rule

    def predict(self, x: Instance) -> object:
        return self.model.predict(x)
    
    def get_rule(self, instances: Iterable[Instance]) -> Rule:
        self.train(instances)
        return self.model
    
    def _specify(self, rule: Rule, attribute_values_map: Dict[int, object]) -> Set[Rule]:
        specified_rules = set()
        for attribute_idx in attribute_values_map:
            for value in attribute_values_map[attribute_idx]:
                # don't generate inconsistent rules
                if rule[attribute_idx] is None:
                    new_rule = Rule(rule)
                    new_rule[attribute_idx] = value
                    specified_rules.add(new_rule)

        return specified_rules
        
    def _remove_more_general(self, rules: Iterable[Rule]) -> Set[Rule]:
        more_general_rules = set()
        for rule1 in rules:
            for rule2 in rules:
                if rule1 != rule2 and OneRuleBeamSearch._is_more_general(rule1, rule2):
                    more_general_rules.add(rule1)
        return rules.difference(more_general_rules)
    
    @staticmethod
    def _is_more_general(general_rule: Rule, specific_rule: Rule):
        for val1, val2 in zip(general_rule, specific_rule):
            if val1 != val2 and val1 is not None:
                return False
        return True
