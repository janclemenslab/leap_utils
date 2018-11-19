import numpy as np


def max2d(X: np.array) -> (tuple, float):
    """Get position and value of array maximum."""
    row, col = np.unravel_index(X.argmax(), X.shape)
    return (row, col), X[row, col]


def max_simple(confmap: np.array) -> (np.array, np.array):
    """Detect maxima in each layer of the confmap.

    prob is val at max pos, normalized by sum over all values in respective layer.
    Arguments:
        confmap = [width, height, layers]
    Returns:
        locs - [layers, 2]
        probs - [layers, 1]

    """
    if confmap.ndim != 3:
        raise ValueError(f'input should be 3D (width, height, layers) but is {confmap.shape}.')

    nb_parts = confmap.shape[-1]
    peak_loc = np.zeros((nb_parts, 2))
    peak_prb = np.zeros((nb_parts, 1))

    for part_idx in range(nb_parts):
        peak_loc[part_idx, ...], peak_amp = max2d(confmap[..., part_idx])
        peak_prb[part_idx, ...] = peak_amp / np.sum(confmap[..., part_idx])

    return peak_loc, peak_prb


def process_confmaps_simple(confmaps: np.array) -> (np.array, np.array):
    """Simply take the max.

    Arguments:
        confmaps = [nbmaps x width x height x layers]
    Returns:
        positions = [nbmaps, layers, 2]
        confidence =  [nbmaps, layers, 1]

    """
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
