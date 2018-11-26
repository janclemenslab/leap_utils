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


def vplay(frames: np.array, idx: np.array = None, moviemode: bool = False):
    """Plots boxes, either in a movie (moviemode = True) or frame by frame (moviemode = False)

    Input: list of frames (output from export_boxes)

        TODO: description of function and input.

    """
    import cv2

    if idx is None:
        idx = range(len(frames))

    if len(idx) == len(frames)/2:
        ridx = np.zeros(len(frames), dtype=int)
        ridx[::2], ridx[1::2] = idx, idx
        idx = ridx

    if moviemode:
        ii = 0
        while True:
            frame = frames[ii, ...]
            cv2.putText(frame, str(idx[ii]), (12, 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 250), lineType=4)
            cv2.imshow('movie', frame)
            ii += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            if ii >= len(frames)-1:
                ii = 0

    else:
        ii = 0
        while True:
            frame = frames[ii, ...]
            cv2.putText(frame, str(idx[ii]), (12, 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 250), lineType=4)
            cv2.imshow('movie', frame)
            wkey = cv2.waitKey(0)

            if wkey & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            elif wkey & 0xFF == ord('d'):
                ii += 1
            elif wkey & 0xFF == ord('a'):
                ii -= 1

            if ii > len(frames)-1:
                ii = 0
            elif ii < 0:
                ii = len(frames)-1
