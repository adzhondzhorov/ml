from typing import Iterable, Set, Tuple
from itertools import chain, product, permutations
from collections import defaultdict

from app.base import Instance, LearningAlgorithm, get_most_common_value
from app.rule_based_learning.base import Predicate, Literal, Rule


class FOIL(LearningAlgorithm):
    def __init__(self, len_attributes: int, target_predicate: Predicate):
        self.len_attributes = len_attributes
        self.target_predicate = target_predicate

    def train(self, instances: Iterable[Instance]):
        learned_rules = list()
        predicates = self._get_predicates(instances)
        constants = self._get_constants(instances)
        pos, neg = self.get_pos_neg_examples(predicates, constants)

        while pos:
            new_neg = list(neg)
            rule = Rule(list())
            rule.prediction = True
            while new_neg:
                literals = self._get_literals(rule, predicates)
                best_literal = max(literals, key=lambda l: FOIL._foil_gain(rule, l, instances))
                rule.append(*best_literal)
                new_neg = [n for n in new_neg if not rule.predict(n)]
            pos = [e for e in pos if not rule.predict(e)]
        self.model = sorted_rules

    def predict(self, x: Instance) -> object:
        for rule in self.model:
            prediction = rule.predict(x)
            if prediction is not None:
                return prediction

    def get_pos_neg_examples(self, predicates: Iterable[Predicate], constants: Iterable[str]) -> Tuple[Iterable[Instance], Iterable[Instance]]:
        pos = list()
        neg = list()
        target_predicate_attributes = product(*([constants] * self.target_predicate.num_arguments))
        for attr in target_predicate_attributes:
            instance = Instance([self.target_predicate, *attr])
            if instance in instances:
                pos.append(instance)
            else:
                neg.append(instance)
        return pos, neg
        
    @staticmethod
    def _foil_gain(rule: Rule, literal: Literal, examples: Iterable[Instance], pos: Iterable[Instance], neg: Iterable[Instance]) -> float:
        for literal in literals

    def _get_literals(self, rule: Rule, predicates: Iterable[Predicate]) -> Set[Literal]:
        literals = set()
        variables = set(range(self.target_predicate.num_arguments))
        for term in rule:
            if isinstance(term, int):
                variables.add(term)
        for predicate in predicates:
            parameters_permutations = permutations(variables, predicate.num_arguments)
            for parameters in parameters_permutations:
                literals.add(Literal(predicate, parameters))

        return literals

    def _get_predicates(self, examples: Iterable[Instance]) -> Set[str]:
        return set([e[0] for e in examples])
    
    def _get_constants(self, examples: Iterable[Instance]) -> Set[str]:
        return set(chain.from_iterable([e[1:] for e in examples]))

female = Predicate("female", 1)
male = Predicate("male", 1)
father = Predicate("father", 2)
mother = Predicate("mother", 2)
daughter = Predicate("daughter", 2)
instances = [Instance([female, "Sharon"]),
             Instance([male, "Bob"]),
             Instance([mother, "Louise", "Sharon"]),
             Instance([daughter, "Sharon", "Louise"]),
             Instance([daughter, "Sharon", "Bob"]),
             Instance([father, "Bob", "Sharon"]),
             Instance([mother, "Nora", "Bob"]),
             Instance([father, "Viktor", "Bob"])]

foil = FOIL(len(instances[0]), target_predicate=daughter)
foil.train(instances)