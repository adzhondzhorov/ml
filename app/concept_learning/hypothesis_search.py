from typing import Dict, Iterable

from app.base import LearningAlgorithm, Instance, get_attribute_values_map
from app.concept_learning.base import Hypothesis, All


class FindS(LearningAlgorithm):
    def __init__(self, len_attributes: int):
        self.model = None
        self.len_attributes = len_attributes

    def train(self, instances: Iterable[Instance]):
        # 1. Initialize h to the most specific hypothesis in H
        h = Hypothesis([None] * (self.len_attributes - 1))
        # 2. For each positive training instance x
        for x in instances:
            if x[Instance.target_attribute_idx]:
                # For each attribute constraint a, in h
                for idx, a in enumerate(h):
                    # If the constraint a, is satisfied by x
                    # Then do nothing
                    if not _is_attribute_constraint_satisfied(x, a, idx):
                        # Else replace a, in h by the next more general constraint that is satisfied by x
                        h = _generalize_hypothesis_by_attribute(h, x[idx], idx)
        # 3. Output hypothesis h
        self.model = h

    def predict(self, instance: Instance) -> bool:
        # classified as positive if the instance satisfies the hypothesis; otherwise - negative
        for idx, a in enumerate(self.model):
            if not _is_attribute_constraint_satisfied(instance, a, idx):
                return False
        return True


class CandidateElimination(LearningAlgorithm):
    def __init__(self, len_attributes: int):
        self.model = None
        self.len_attributes = len_attributes

    def train(self, instances: Iterable[Instance]):
        # Initialize G to the set of maximally general hypotheses in H
        G = {Hypothesis([All] * (self.len_attributes - 1))}
        # Initialize S to the set of maximally specific hypotheses in H
        S = {Hypothesis([None] * (self.len_attributes - 1))}

        attribute_values_map = get_attribute_values_map(instances)

        # For each training example d, do
        for d in instances:
            # If d is a positive example
            if d[Instance.target_attribute_idx]:
                # Remove from G any hypothesis inconsistent with d
                G = {g for g in G if _is_hypothesis_consistent(g, d)}
                S_new = set(S)
                # For each hypothesis s in S that is not consistent with d
                for s in S:
                    if not _is_hypothesis_consistent(s, d):
                        # Remove s from S
                        S_new.remove(s)
                        # Add to S all minimal generalizations h of s such that
                        gen_hypotheses = _generalize_hypothesis(s, d)
                        for gen_h in gen_hypotheses:
                            # h is consistent with d, and some member of G is more general than h
                            if _is_hypothesis_consistent(gen_h, d) and any(g > gen_h for g in G):
                                S_new.add(gen_h)
                    # Remove from S any hypothesis that is more general than another hypothesis in S
                    S = {s_new for s_new in S_new if any(s_new > s_ref for s_ref in S_new)}
            # If d is a negative example
            else:
                # Remove from S any hypothesis inconsistent with d
                S = {s for s in S if _is_hypothesis_consistent(s, d)}
                G_new = set(G)
                # For each hypothesis g in G that is not consistent with d
                for g in G:
                    if not _is_hypothesis_consistent(g, d):
                        # Remove g from G
                        G_new.remove(g)
                        # Add to G all minimal specializations h of g such that
                        spec_hypotheses = _specify_hypothesis(g, d, attribute_values_map)
                        for h in spec_hypotheses:
                            # h is consistent with d, and some member of S is more specific than h
                            if _is_hypothesis_consistent(h, d) and any(s < h for s in S):
                                G_new.add(h)
                    # Remove from G any hypothesis that is less general than another hypothesis in G
                    G = {g_new for g_new in G_new if any(g_new < g_ref for g_ref in G_new)}

        self.model = S, G

    def predict(self, instance: Instance)-> bool:
        S, G = self.model

        # classified as positive if the instance satisfies every member of S
        s_prediction = True
        for s in S:
            for idx, a in enumerate(s):
                if not _is_attribute_constraint_satisfied(instance, a, idx):
                    s_prediction = False
                    break
            if not s_prediction:
                break

        if s_prediction:
            return True

        # classified as negative if the instance satisfies none of the members of G
        g_prediction = False
        for g in G:
            local_prediction = True
            for idx, a in enumerate(g):
                if not _is_attribute_constraint_satisfied(instance, a, idx):
                    local_prediction = False
                    break
            if local_prediction:
                g_prediction = True
                break

        if not g_prediction:
            return False

        return None


def _is_attribute_constraint_satisfied(x: Instance, a: object, idx: int):
    if 0 <= idx < len(x)-1:
        return a is not None and (a is All or a == x[idx])
    else:
        return False


def _is_hypothesis_consistent(h: Hypothesis, d: Instance):
    for idx, a in enumerate(h):
        if not _is_attribute_constraint_satisfied(d, a, idx):
            return not d[Instance.target_attribute_idx]
    return d[Instance.target_attribute_idx]


def _generalize_hypothesis_by_attribute(h: Hypothesis, a: object, idx: int):
    generalized_h = Hypothesis(h)
    if 0 <= idx < len(h):
        if h[idx] is None:
            generalized_h[idx] = a
        elif h[idx] != a:
            generalized_h[idx] = All

    return generalized_h


def _generalize_hypothesis(h: Hypothesis, d: Instance):
    generalized_h = Hypothesis(h)
    for idx, a in enumerate(d):
        if h[idx] is None:
            generalized_h[idx] = a
        elif h[idx] != a:
            generalized_h[idx] = All
    return {generalized_h}


def _specify_hypothesis(h: Hypothesis, d: Instance, attribute_values_map: Dict[int, object]):
    specified_hypotheses = set()
    for idx, a in enumerate(d):
        if h[idx] == All:
            for a_val in attribute_values_map[idx]:
                new_h = Hypothesis(h)
                if a_val != a:
                    new_h[idx] = a_val
                    specified_hypotheses.add(new_h)

    return specified_hypotheses


def _specify_hypothesis_by_attribute(h: Hypothesis, a: object, idx: int):
    specified_h = Hypothesis(h)
    if 0 <= idx < len(h):
        if h[idx] is All:
            specified_h[idx] = a
        elif h[idx] != a:
            specified_h[idx] = None

    return specified_h
