from typing import Sequence, Union, List
import numpy as np
from skimage.transform import rotate as sk_rotate


def crop_frame(frame: np.array, center: np.uintp, box_size: np.uintp, mode: str='clip') -> np.array:
    """Crop frame.

    frame: np.array, center: np.uintp, box_size: np.uintp, mode: str='clip'
    """
    box_hw_left = np.ceil(box_size/2)
    box_hw_right = np.floor(box_size/2)
    x_px = np.arange(center[0]-box_hw_left[0], center[0]+box_hw_right[0]).astype(np.intp)
    y_px = np.arange(center[1]-box_hw_left[1], center[1]+box_hw_right[1]).astype(np.intp)
    return frame.take(y_px, mode=mode, axis=1).take(x_px, mode=mode, axis=0)


def export_boxes(frames: Sequence, box_centers: np.array, box_size: List[int],
                 box_angles: np.array=None) -> (np.array, np.array, np.array):
    """Export boxes.

    Args:
        frames: list of frames (python list - frames themselves are np.arrays)
        box_size: [width, height]
        box_centers: [nframes in vid, flyid, 2]
        box_angles: [nframes in vid, flyid, 1], if not None, will rotate flies
    Returns:
         boxes
         fly_id: fly id for each box
         fly_frame: frame number for each box
    """
    nb_frames = len(frames)
    nb_channels = frames[0].shape[-1]
    nb_flies = box_centers.shape[1]
    nb_boxes = nb_frames*nb_flies
    # check input:

    # make this a dict?
    boxes = np.zeros((nb_boxes, *box_size, nb_channels), dtype=np.uint8)
    fly_id = -np.ones((nb_boxes,), dtype=np.intp)
    fly_frame = -np.ones((nb_boxes,), dtype=np.intp)

    box_idx = -1
    for frame_number, frame in enumerate(frames):
        for fly_number in range(nb_flies):
            box_idx += 1
            fly_id[box_idx] = fly_number
            fly_frame[box_idx] = frame_number
            if box_angles is not None:
                box = crop_frame(frame, box_centers[frame_number, fly_number, :], 1.5*np.array(box_size))  # crop larger box to get padding for rotation
                box = sk_rotate(box, box_angles[frame_number, fly_number, :],
                                resize=False, mode='edge', preserve_range=True)
                box = crop_frame(box, np.array(box.shape)/2, box_size)    # trim rotated box to the right size
            else:
                box = crop_frame(frame, box_centers[frame_number, fly_number, :], np.array(box_size))
            boxes[box_idx, ...] = box

    return boxes, fly_id, fly_frame


def normalize_matlab_boxes(X, permute=(0, 3, 2, 1)):
    """Normalizes shape and scale/dtype of input data.

    This is only required if using boxes saved through matlab since matlab
    messes up the order of dimensions.
    """

    # Add singleton dim for single images
    if X.ndim == 3:
        X = X[None, ...]

    # Adjust dimensions
    X = np.transpose(X, permute)

    # Normalize
    if X.dtype == "uint8":
        X = X.astype("float32") / 255

    return X


def normalize_boxes(X):
    """ Normalizes scale/dtype of input data."""

    # Add singleton dim for single images
    if X.ndim == 3:
        X = X[None, ...]

    # Normalize
    if X.dtype == "uint8":
        X = X.astype("float32") / 255

    return X

def get_angles(heads: np.array, tails: np.array) -> np.array:
    """ Gets angles (to rotate in order to get fly looking up) from
    head-tail axis for all the heads and tails coordinates given.

    Arguments:
        heads: [nframes, fly_id, coordinates(x,y)]
        tails: [nframes, fly_id, coordinates(x,y)]
    Returns:
        fly_angles (in degrees): [nframes, fly_id, 1]

    """
    nframes, nflies = heads.shape[0:2]
    fly_angles = np.zeros((nframes, nflies, 1))
    for ifly in range(nflies):
        fly_angles[:,ifly,0] = 90 + np.arctan2(heads[:,ifly,0]-tails[:,ifly,0],heads[:,ifly,1]-tails[:,ifly,1]) * 180 / np.pi

    return fly_angles
