from typing import List

import numpy as np


def sgn(i: float) -> int:
    return 1 if i > 0 else -1


def sigmoid(i: float) -> float:
    return 1 /(1 + np.power(np.e, -i))
