import numpy as np
from leap_utils.utils import flatten, unflatten


def flatten_unflatten(X0):
    X = X0.reshape((2, 3, X0.shape[-1]))

    assert np.all(flatten(X) == X0)
    print(X0.T)
    print(flatten(X).T)
    print(flatten(X)[3, :])

    assert np.all(unflatten(X0, 3) == X)
    print(X)
    print(unflatten(X0, 3))
    print(unflatten(X0, 3)[1, 0, :])

    # test conversion round-trips
    assert np.all(unflatten(flatten(X), 3) == X)
    assert np.all(flatten(unflatten(X0, 3)) == X0)


def test_1D():
    X0 = np.arange(1, 7).T
    X0 = X0[..., np.newaxis]
    flatten_unflatten(X0)


def test_2D():
    X0 = np.arange(1, 7).T
    X0 = X0[..., np.newaxis]
    X0 = np.concatenate((X0, X0), axis=1)
    X0[:, 1] *= 10
    flatten_unflatten(X0)


test_1D()
test_2D()
