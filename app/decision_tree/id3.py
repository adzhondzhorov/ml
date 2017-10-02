import random
from typing import Iterable
from enum import Enum 

from app.base import Instance, LearningAlgorithm, get_attribute_values_map, get_most_common_value
from app.decision_tree.base import entropy


class Node(object):
    def __init__(self):
        self.children = list()

    def add_child(self, child: "Node"):
        self.children.append(child)
    	

class AttributeNode(Node):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def get_child(self, value:object):
        for child in self.children:
            if child.value == value:
                return child
        return None
    

class ValueNode(Node):
    def __init__(self, value: object):
        super().__init__()
        self.value = value
    
    def is_leaf(self):
        return not self.children

class ID3(LearningAlgorithm):
    def __init__(self, len_attributes: int):
        self.model = Node()
        self.len_attributes = len_attributes

    def train(self, instances: Iterable[Instance]):
        attribute_values_map = get_attribute_values_map(instances)
          
        self._build(self.model, list(range(self.len_attributes - 1)), instances, attribute_values_map)

    def _build(self, node: Node, attributes: Iterable[int], instances: Iterable[Instance], attribute_values_map):
        target_values = [instance[Instance.target_attribute_idx] for instance in instances]
        if(all(target_values)):
            node.add_child(ValueNode(True))
        elif(not any(target_values)):
	        node.add_child(ValueNode(False))
        elif not attributes:
            node.add_child(ValueNode(get_most_common_value(target_values)))
        else:
            attribute_idx = _get_best_attribute(instances, attributes, attribute_values_map)
            attributes.remove(attribute_idx)
            attribute_node = AttributeNode(attribute_idx)
            node.add_child(attribute_node)
            values = attribute_values_map[attribute_idx]
            for value in values:
                value_node = ValueNode(value)
                attribute_node.add_child(value_node)
                value_instances = [instance for instance in instances if instance[attribute_idx] == value]
                self._build(value_node, attributes, value_instances, attribute_values_map)
    
    def predict(self, x: Instance) -> object:
        return ID3._traverse(self.model, x)

    @staticmethod
    def _traverse(node: Node, x: Instance) -> object:
        if type(node) is ValueNode and node.is_leaf():
            return node.value
        elif type(node) is AttributeNode:
            node = node.get_child(x[node.index])
            return ID3._traverse(node, x)
        else:
            return ID3._traverse(node.children[0], x)
            
            
def _get_best_attribute(instances, attributes, attribute_values_map):    
    return max(attributes, key=lambda attr: _information_gain(instances, attr, attribute_values_map[attr]))


def _information_gain(instances, attribute, values):
    all_entropy = _get_entropy(instances)
    weighted_sum = 0
    for value in values:
        value_instances = [i for i in instances if i[attribute] == value]
        value_entropy = _get_entropy(value_instances)
        weighted_sum += value_entropy * len(value_instances)

    return all_entropy - weighted_sum / len(instances)


def _get_entropy(instances):
    if len(instances) == 0:
        return 0
    positive_probability = len([i for i in instances if i[Instance.target_attribute_idx]]) / float(len(instances))
    return entropy(positive_probability)
