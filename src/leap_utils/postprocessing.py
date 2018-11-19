import numpy as np


def max2d(X: np.array) -> (tuple, float):
    """Get position and value of array maximum."""
    row, col = np.unravel_index(X.argmax(), X.shape)
    return (row, col), X[row, col]


def max_simple(confmap: np.array) -> np.array:
    """Detect maxima in each layer of the confmap.

    Arguments:
        confmap = []
    """
    if confmap.ndim != 3:
        raise ValueError(f'input should be 3D (width x height x layers) but is {confmap.shape}.')

    nb_parts = confmap.shape[-1]
    peak_loc = np.zeros((nb_parts, 2))
    peak_amp = np.zeros((nb_parts, 1))
    for part_idx in range(nb_parts):
        peak_loc[part_idx, ...], peak_amp[part_idx, ...] = max2d(confmap[..., part_idx])
    return peak_loc, peak_amp


def process_confmaps_simple(confmaps: np.array) -> (np.array, np.array):
    """Simply take the max."""
    nb_maps, nb_parts = confmaps.shape[0], confmaps.shape[-1]
    positions = np.zeros((nb_maps, nb_parts, 2))
    confidence = np.zeros((nb_maps, nb_parts, 1))
    for nb, confmap in enumerate(confmaps):
        positions[nb, ...], confidence[nb, ...] = max_simple(confmap)
    return positions, confidence


def process_confmaps_bayesian(confmaps: np.array, prior_information) -> (np.array, np.array):
    """Take all local maxime (using skimage), choose maximum based in prio info and confidence."""
    positions = None
    confidence = None
    return positions, confidence
