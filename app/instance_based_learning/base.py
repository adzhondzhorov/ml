import math

from app.base import Instance

def get_distance(a: Instance, b: Instance):
        return math.sqrt(sum([ai == bi for ai, bi in zip(a, b)]))
