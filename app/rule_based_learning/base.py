from typing import Iterable
from math import log

from app.base import Instance

class Rule(list):
    def __init__(self, l):
        super().__init__(l)
        self.prediction = None

    def can_predict(self, x: Instance):
        for idx, value in enumerate(self):
            if value != x[idx] and value is not None:
                return False
        return True

    def predict(self, x: Instance):
        return self.prediction if self.can_predict(x) else None
    
    def __hash__(self):
        return tuple(self).__hash__()


def get_performance(rule: Rule, instances: Iterable[Instance]):
    match_examples = [i for i in instances if rule.can_predict(i)]
    if len(match_examples) == 0:
        return 0 
    p = len([e for e in match_examples if e[Instance.target_attribute_idx]]) /len(match_examples)

    return -entropy(p)


def entropy(p: float) -> float:
    if p == 0 or p == 1:
        return 0
    return -p * log(p, 2) - (1-p) * log(1-p, 2) 
