import numpy as np
from sys import platform


def it(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Yield slices from a numpy array over an arbitrary axis.

    Args:
        x(np.ndarray): numpy array
        axis(int): axis over which to iterate (defaults to last axis)
    Returns:
        generator that yields slice along axis

    Example:
        This code will return 10 4x4 matrices ('planes')
        stack_of_planes = np.zeros((10, 4, 4))
        for plane in it(stack_of_planes, axis=0):
            print(plane)

    """
    for idx in range(x.shape[axis]):
        yield x.take(idx, axis=axis)


def flatten(X: np.ndarray) -> (np.ndarray):
    """Convert arrays from [dim1, dim2, ...] to [dim2*dim2, ...].

    Arguments:
        X: np.ndarray [dim1, dim2, ...]
    Returns:
        X: np.ndarray [dim1 * dim2, ...]
    """
    return X.reshape((X.shape[0]*X.shape[1], *X.shape[2:]), order='A')


def unflatten(X: np.ndarray, dim2_len: int = 2) -> (np.ndarray):
    """Convert from [dim1*dim2, ...] to [dim1, dim_len...].

    Arguments:
        X: np.ndarray [dim1_len*dim2_len, ...]
        dim2_len: int=2
    Returns:
        X: np.ndarray [dim1_len, dim2_len, ...]
    """
    return X.reshape((int(X.shape[0]/dim2_len), dim2_len, *X.shape[1:]), order='A')


def smooth(x, N):
    """Smooth signal using box filter of length N samples."""
    return np.convolve(x, np.ones((N,)) / N, mode='full')[(N - 1):]


# - to utils
def max2d(X: np.ndarray) -> (tuple, float):
    """Get position and value of array maximum."""
    row, col = np.unravel_index(X.argmax(), X.shape)
    return (row, col), X[row, col]


def islinux():
    return platform == "linux" or platform == "linux2"


def ismac():
    return platform == "darwin"


def iswin():
    platform == "win32"
