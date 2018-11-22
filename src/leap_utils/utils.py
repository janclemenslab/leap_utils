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


def islinux():
    return platform == "linux" or platform == "linux2"


def ismac():
    return platform == "darwin"


def iswin():
    platform == "win32"
