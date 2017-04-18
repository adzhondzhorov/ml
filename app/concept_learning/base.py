from app.base import Instance


class AllType(object):
    def __str__(self):
        return "All"

    def __repr__(self):
        return self.__str__()

All = AllType() 


class ConceptInstance(Instance):
    @property
    def is_positive(self):
        return self[self.target_attribute_idx]


class Hypothesis(list):
    LESS = True
    GREATER = False

    def __lt__(self, other):
        return self._compare(other, self.LESS)

    def __gt__(self, other):
        return self._compare(other, self.GREATER)

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __hash__(self):
        return hash(tuple(self))

    def _compare(self, other, operator):
        for self_attribute, other_attribute in zip(self, other):
            if not self._compare_attribute_constraints(self_attribute, other_attribute, operator):
                return False
        return True

    @classmethod
    def _compare_attribute_constraints(cls, attribute1, attribute2, operator):
        c = lambda a1, a2: a1 is All or a2 is None or a1 == a2
        if operator == cls.LESS:
            return c(attribute2, attribute1)
        elif operator == cls.GREATER:
            return c(attribute1, attribute2)
