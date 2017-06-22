from math import log


def entropy(p: float) -> float:
    if p == 0 or p == 1:
        return 0
    return -p * log(p, 2) - (1-p) * log(1-p, 2) 
