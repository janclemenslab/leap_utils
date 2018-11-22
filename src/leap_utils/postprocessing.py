import numpy as np
import skimage.filters
import skimage.feature

from .utils import it, max2d


# TODO:
# - return max values
# - "overload" max2d via num_peaks argument?
# - reorganize and to utils
def max2d_multi(mask: np.ndarray, num_peaks: int, smooth: float = None,
                   exclude_border: bool = True, min_distance: int = 4) -> (np.ndarray, np.ndarray):
    """Detect one or multiple peaks in each channel of an image."""
    maxima = np.ndarray((2, num_peaks, mask.shape[-1]))
    for idx, plane in enumerate(it(mask, axis=-1)):
        if smooth:
            plane = skimage.filters.gaussian(plane, smooth)
        tmp = skimage.feature.peak_local_max(plane, num_peaks=num_peaks, exclude_border=exclude_border, min_distance=min_distance)
        print(tmp)
        maxima[..., idx] = tmp
    return maxima


def max_simple(confmap: np.ndarray) -> (np.ndarray, np.ndarray):
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


def process_confmaps_simple(confmaps: np.ndarray) -> (np.ndarray, np.ndarray):
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


def process_confmaps_bayesian(confmaps: np.ndarray, prior_information) -> (np.ndarray, np.ndarray):
    """Take all local maxime (using skimage), choose maximum based in prio info and confidence."""
    positions = None
    confidence = None
    return positions, confidence
