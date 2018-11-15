import numpy as np


def process_confmaps_simple(confmaps: np.array) -> (np.array, np.array):
    """Simply take the max."""
    positions = None
    confidence = None
    return positions, confidence


def process_confmaps_bayesian(confmaps: np.array, prior_information) -> (np.array, np.array):
    """Take all local maxime (using skimage), choose maximum based in prio info and confidence."""
    positions = None
    confidence = None
    return positions, confidence
