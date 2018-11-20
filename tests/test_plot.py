import skimage.filters as sk
import matplotlib.pyplot as plt
import numpy as np

from leap_utils.plot import color_confmaps


def test_colormap_confmaps():
    confmaps = np.zeros((60, 60, 6))
    for mp in range(confmaps.shape[-1]):
        confmaps[np.random.randint(5, 55), np.random.randint(5, 55), mp] = 1
        confmaps[..., mp] = sk.gaussian(confmaps[..., mp], 2)
        confmaps[..., mp] = confmaps[..., mp] / np.max(confmaps[..., mp])

    cm = plt.get_cmap('gist_rainbow')
    cols = np.array(cm(np.linspace(0, 1, confmaps.shape[-1])))
    print(type(cols))
    cmap_strs = ['tab10', 'Set3', 'Set1', 'viridis', 'jet',  'hsv', 'inferno',
                 plt.get_cmap('gist_rainbow'), 'gist_rainbow',
                 cols]
    plt.ion()
    plt.subplot(3, 4, 1)
    plt.imshow(np.sum(confmaps, axis=2))
    plt.title('raw')

    for idx, cmap_str in enumerate(cmap_strs):
        confmaps_merge, colors = color_confmaps(confmaps, cmap=cmap_str)
        plt.subplot(3, 4, idx+2)
        plt.imshow(confmaps_merge)
        plt.title(cmap_str)
    plt.pause(2)
