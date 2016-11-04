from base import All


class Hypothesis(List):
    def __lt__(self, other):
        for self_attribute, other_attribute in zip(self, other): 
            if not _lt_attribute(self_attribute, other_attribute):
                return False
	return True
     
    def __gt__(self, other):
       for self_attribute, other_attribute in zip(self, other): 
           if _lt_attribute(self_attribute, other_attribute):
               return False
       return True
    
    def _lt_attribute(attribute1, attribute2):
        return attribute1 is None or attribute1 == attribute2 or attribute2 is All


def find_s(data, len_attributes):
    hypothesis = [None] * (len_attributes - 1)
    for instance in data:
        if is_positive(instance):
            for idx, attribute_value in enumerate(instance[:-1]): 
                if not is_attribute_satisfied(attribute_value, idx, hypothesis):
                    generalize_hypothesis_by_attribute(attribute_value, idx, hypothesis)
    return hypothesis


def is_positive(instance):
    return instance[-1] == "Yes"


def is_attribute_satisfied(attribute_value, idx, hypothesis):
    return attribute_value == hypothesis[idx] or hypothesis[idx] is All 


def generalize_hypothesis_by_attribute(attribute_value, idx, hypothesis):
    if hypothesis[idx] is None:
        hypothesis[idx] = attribute_value
    elif hypothesis[idx] != attribute_value:
        hypothesis[idx] = All
i

def is_instance_satisfied(instance, hypothesis):
    for idx, attribute_value in enumerate(instance[:-1]): 
        if not is_attribute_satisfied(attribute_value, idx, hypothesis):
            return False
    return True


def generalize_hypothesis(instance, hypothesis):
    for idx, attribute_value in enumerate(instance[:-1]): 
        yield generalize_hypothesis_by_attribute(attribute_value, idx, hypothesis)


def candidate_elimination(data, len_attributes):
    G = [[All] * (len_attributes - 1)]
    S = [[None] * (len_attributes - 1)]
    
    for instance in data:
        if is_positive(instance):
	    G = [g for g in G if is_instance_satisfied(instance)]
        else:
	    S = S + generalize_hypothesis(instance, hypothesis)
	    
    

