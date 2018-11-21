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

    nfly = heads.shape[1]
    heads = nframes2nboxes(heads)
    tails = nframes2nboxes(tails)

    fly_angles = np.zeros((heads.shape[0],1))
    fly_angles[:,0] = 90 + np.arctan2(heads[:,0]-tails[:,0],heads[:,1]-tails[:,1]) * 180 / np.pi

    return nboxes2nframes(fly_angles,nfly)


def nframes2nboxes(X: np.array) -> (np.array):
    """ Converts np.arrays of shape [nframes,nfly...] into [nboxes, ...]


    Arguments:
        X: np.array [nframes, nfly, ...]
    Returns:
        Y: np.array [nboxes, ...]
    """

    Y = X.reshape((X.shape[0]*X.shape[1],*X.shape[2:]),order='F')

    return Y


def nboxes2nframes(Y: np.array, nfly: int=2) -> (np.array):
    """ Converts np.arrays of shape [nboxes, ...] into [nframes,nfly...]


    Arguments:
        Y: np.array [nboxes, ...]
        nfly: int=2
    Returns:
        X: np.array [nframes, nfly, ...]
    """

    X = Y.reshape((int(Y.shape[0]/nfly),nfly,*Y.shape[1:]),order='F')

    return X


def detect_bad_boxes_byAngle(pred_positions: np.array, epsilon: float=10, head_idx: int=0, tail_idx: int=11, nfly: int=2) -> (np.array, np.array):
    """ Calculates Head-Tail axis angle to vertical, and
    selects cases that fall out of the threshold epsilon (in degrees).

    Assumes that data of shape [nboxes, ...] is organized for nflies,
    for example, if nfly = 2:
        X = [ (box1_fly1, box1_fly2, box2_fly1, box2_fly2, ...), ...]

    Arguments:
        pred_positions: [nboxes, layers, 2]
        head_idx: int=0
        tail_idx: int=11
        nfly: int=2
    Returns:
        fly_angles (in degrees): [nframes, fly_id, 1]
        bad_boxes_byAngle: [nboxes, 1], where 1 = bad box, 0 = good box
    """

    # Initialize variables
    nboxes = pred_positions.shape[0]
    fly_angles = np.zeros((nboxes,1))
    bad_boxes_byAngle = np.zeros((nboxes,1))

    # Reshape coordinates
    heads = nboxes2nframes(pred_positions[:,head_idx,:],nfly)
    tails = nboxes2nframes(pred_positions[:,tail_idx,:],nfly)

    # Get new angles
    fly_angles = get_angles(heads,tails)

    # Select bad cases according to epsilon
    bad_boxes_byAngle = abs(fly_angles) > epsilon

    return nboxes2nframes(fly_angles,nfly), bad_boxes_byAngle
