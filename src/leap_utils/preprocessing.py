from typing import Sequence, Union, List
import numpy as np
from skimage.transform import rotate as sk_rotate
import deepdish as dd
import scipy.signal


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


def angles(heads: np.array, tails: np.array) -> np.array:
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

    # Reshape coordinates (CURRENTLY NOT WORKING, BUT THIS WOULD BE SHORTER/ MORE ELEGANT)
    # heads = nboxes2nframes(pred_positions[:,head_idx,:],nfly)
    # tails = nboxes2nframes(pred_positions[:,tail_idx,:],nfly)

    # Reshape coordinates
    heads = np.zeros((int(pred_positions.shape[0]/nfly),nfly,2))
    tails = np.zeros(heads.shape)
    heads[:,0,:], heads[:,1,:] = pred_positions[::2,head_idx,:], pred_positions[1::2,head_idx,:]
    tails[:,0,:], tails[:,1,:] = pred_positions[::2,tail_idx,:], pred_positions[1::2,tail_idx,:]

    # Get new angles
    fly_angles = angles(heads,tails)

    # Select bad cases according to epsilon
    bad_boxes_byAngle = abs(fly_angles) > epsilon

    return fly_angles, bad_boxes_byAngle

def smooth(x, N):
    """Smooth signal using box filter of length N samples."""
    return np.convolve(x, np.ones((N,)) / N, mode='full')[(N - 1):]


def fix_orientations(lines0, chamber_number=0):
    """Fix the head-tail orientation of flies based on speed and changes in orientation."""
    nflies = lines0.shape[2]
    vel = np.zeros((lines0.shape[0], nflies))
    ori = np.zeros((lines0.shape[0], 2, nflies))
    lines_fixed = lines0.copy()
    chamber_number = 0

    for fly in range(nflies):
        # get fly lines and smooth
        lines = lines0[:, chamber_number, fly, :, :].astype(np.float64)  # time x [head,tail] x [x,y]
        for h in range(2):
            for p in range(2):
                lines[:, h, p] = smooth(lines[:, h, p], 10)

        # get fly movement and smooth
        dpos = np.gradient(lines[:, 0, :], axis=0)  # change of head position over time - `np.gradient` is less noisy than `np.diff`
        for p in range(2):
            dpos[:, p] = smooth(dpos[:, p], 10)

        # get fly orientation
        ori[:, :, fly] = np.diff(lines, axis=1)[:, 0, :]  # orientation of fly: head pos - tail pos, `[:,0,:]` cuts off trailing dim
        ori_norm = np.linalg.norm(ori[:, :, fly], axis=1)  # "length" of the fly

        # dpos_norm = np.linalg.norm(dpos, axis=1)
        # alignment (dot product) between movement and orientation
        dpos_ori = np.einsum('ij,ij->i', ori[:, :, fly], dpos)  # "element-wise" dot product between orientation and velocity vectors
        vel[:, fly] = dpos_ori / ori_norm  # normalize by fly length (norm of ori (head->tail) vector)

        # 1. clean up velocity - only consider epochs were movement is fast and over a prolonged time
        orichange = np.diff(np.unwrap(np.arctan2(ori[:, 0, fly], ori[:, 1, fly])))  # change in orientation for change point detection
        velsmooththres = smooth(vel[:, fly], 20)  # smooth velocity
        velsmooththres[np.abs(velsmooththres) < 0.4] = 0  # threshold - keep only "fast" events to be more robust
        velsmooththres = scipy.signal.medfilt(velsmooththres, 5)  # median filter to get rid of weird, fast spikes in vel

        # 2. detect the range of points during which velocity changes in sign
        idx, = np.where(velsmooththres != 0)  # indices where vel is high
        switchpoint = np.gradient(np.sign(velsmooththres[idx]))  # changes in the sign of the thres vel
        switchtimes = idx[np.where(switchpoint != 0)]  # indices where changes on vel sign occurr
        switchpoint = switchpoint[switchpoint != 0]    # sign of the change in vel

        # 3. detect actual change point with that range
        changepoints = []  # fill this with
        for cnt in range(0, len(switchtimes[:-1]), 2):
            # define change points as maxima in orientation changes between switchs in direction of motion
            changepoints.append(switchtimes[cnt] + np.argmax(np.abs(orichange[switchtimes[cnt] - 1:switchtimes[cnt + 1]])))
            # mark change points for interpolation
            velsmooththres[changepoints[-1]-1] = -switchpoint[cnt]
            velsmooththres[changepoints[-1]] = switchpoint[cnt+1]

        # 4. fill values using change points - `-1` means we need to swap head and tail
        idx, = np.where(velsmooththres != 0)  # update `idx` to include newly marked change points
        f = scipy.interpolate.interp1d(idx, velsmooththres[idx], kind="nearest", fill_value="extrapolate")
        idx_new, = np.where(velsmooththres == 0)
        ynew = f(range(velsmooththres.shape[0]))

        # 4. swap head and tail
        lines_fixed[ynew < 0, chamber_number, fly, :, :] = lines_fixed[ynew < 0, chamber_number, fly, ::-1, :]
    return lines_fixed
