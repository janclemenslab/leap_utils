import numpy as np
from leap_utils.utils import max2d


def test_max2d():
    X = np.zeros((10, 10))
    X[4, 5] = 1
    loc, amp = max2d(X)
    assert amp == 1
    assert loc == (4, 5)

    X = np.zeros((10, 10))
    loc, amp = max2d(X)
    assert amp == 0
    assert loc == (0, 0)
