from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np


def color_confmaps(confmaps, cmap='gist_rainbow') -> (np.array, np.array):
    """Color code different layers in a confidence maps.

    Usage:
        confmaps_merge, colors = color_confmaps(confmaps, cmap='Set3')
        plt.imshow(confmaps_merge)

    Args:
        confmaps
        cmap - str, cmap object, nparray
    Returns:
        colored_maps
        colors
    """
    if isinstance(cmap, str):  # name of colormap
        cm = plt.get_cmap(cmap)
        cols = np.array(cm(np.linspace(0, 1, confmaps.shape[-1])))
    elif isinstance(cmap, Colormap):  # colormap object
        cm = cmap
        cols = np.array(cm(np.linspace(0, 1, confmaps.shape[-1])))
    elif isinstance(cmap, np.ndarray):  # col vals
        cols = cmap

    colors = cols[..., :3]  # remove alpha channel form colors
    color_confmaps = np.zeros((*confmaps.shape[:2], 3))

    for mp in range(confmaps.shape[-1]):
        color_confmaps += confmaps[..., mp:mp+1] * colors[mp:mp+1, :]
    return color_confmaps, colors


def confmaps(confmaps, cmap='gist_rainbow'):
    confmaps_merge, colors = color_confmaps(confmaps, cmap)
    plt.imshow(confmaps_merge)


def joint_distributions(positions, type):
    # check seaborn gallery:
    # https://seaborn.pydata.org/examples/multiple_joint_kde.html
    # https://seaborn.pydata.org/examples/cubehelix_palette.html
    pass
