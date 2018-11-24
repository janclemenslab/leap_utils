import os
import defopt
import logging
import deepdish as dd
import numpy as np
from videoreader import VideoReader
from leap_utils.preprocessing import export_boxes, angles, normalize_boxes, detect_bad_boxes_by_angle, fix_orientations
from leap_utils.postprocessing import process_confmaps_simple
from leap_utils.predict import predict_confmaps, load_network
from leap_utils.utils import iswin, ismac, flatten, unflatten

# Paths
if iswin():
    root = 'Z:/#Common/'
elif ismac():
    root = '/Volumes/ukme04/#Common/'
else:
    root = '/scratch/clemens10/'
#dataPath = root+'chainingmic/dat'
dataPath = root+'chainingmic/dat.processed'
resPath = root+'chainingmic/res'
networkPath = root+'chainingmic/dat/best_model.h5'


def main(expID: str, *, frame_start: int = 0, frame_stop: int = None, frame_step: int = 1, batch_size: int = 100):
    # Fix tracks
    # Paths
    trackPath = f"{dataPath}/{expID}/{expID}_tracks.h5"
    videoPath = f"{dataPath}/{expID}/{expID}.mp4"
    trackfixedPath = f"{resPath}/{expID}//{expID}_tracks_fixed.h5"
    savingPath = f"{resPath}/{expID}"
    posePath = f"{savingPath}/{expID}_poses.h5"

    try:
        os.mkdir(os.path.dirname(trackfixedPath))
    except FileExistsError as e:
        logging.debug(e)

    # Do not fix if they are already fixed
    if not os.path.exists(trackfixedPath):
        logging.info(f"   fixing tracks.")
        fix_tracks(trackPath, trackfixedPath)
    else:
        logging.info(f"   fixed tracks exist")

    # Load video
    logging.info(f"   opening video {videoPath}.")
    vr = VideoReader(videoPath)
    # Load track
    logging.info(f"   loading tracks from {trackfixedPath}.")
    try:
        data = dd.io.load(trackfixedPath)
        centers = data['centers'][:]    # nframe, channel, fly id, coordinates
        tracks = data['lines']
        chbb = data['chambers_bounding_box'][:]
        heads = tracks[:, 0, :, 0, ::-1]   # nframe, fly id, coordinates
        tails = tracks[:, 0, :, 1, ::-1]   # nframe, fly id, coordinates
        heads = heads + chbb[1][0][:]   # nframe, fly id, coordinates
        tails = tails + chbb[1][0][:]   # nframe, fly id, coordinates
        box_angles = angles(heads, tails)
        box_centers = centers[:, 0, :, :]   # nframe, fly id, coordinates
        box_centers = box_centers + chbb[1][0][:]
        nb_flies = box_centers.shape[1]
        logging.info(f"   nflies: {nb_flies}.")
    except OSError as e:
        logging.error(f'   could not load tracks.')
        logging.debug(e)

    # Specifications for boxes
    if frame_stop is None:
        frame_stop = data['frame_count']
        logging.info(f'   Setting frame_stop: {frame_stop}.')
    frame_numbers = range(frame_start, frame_stop, frame_step)
    batch_idx = list(range(0, len(frame_numbers), batch_size))
    batch_idx.append(frame_stop-frame_start-1)        # ADRIAN TESTING THINGS, SHOULD BE REMOVED IF COMMITED ####################################
    logging.info(f"   frame range: {frame_start}:{frame_stop}:{frame_step}.")
    logging.info(f"   processing in {len(batch_idx)-1} batches of size {batch_size}.")

    box_size = [120, 120]
    nb_frames = len(frame_numbers)
    nb_parts = 12
    nb_boxes = nb_frames * nb_flies
    logging.info(f"   loading network from {networkPath}.")
    network = load_network(networkPath, image_size=box_size)

    # maybe make these h5py.Datasets and save directly
    # with h5py.File( 'w') as f:
        # positions = f.create_dataset("positions", (nb_boxes, nb_parts, 2), chunk=True)
    positions = np.zeros((nb_boxes, nb_parts, 2), dtype=np.uint16)
    confidence = np.zeros((nb_boxes, nb_parts, 1), dtype=np.float16)
    bad_boxes = np.zeros((nb_boxes, 1), dtype=np.bool)
    fixed_angles = np.zeros((nb_boxes, 1), dtype=np.float16)
    fly_id = np.zeros((nb_boxes,), dtype=np.uintp)
    fly_frame = np.zeros((nb_boxes,), dtype=np.uintp)

    # TODO: load network once and pass around
    for batch_num in range(len(batch_idx)-1):
        logging.info(f"PROCESSING BATCH {batch_num}.")
        batch_frame_numbers = list(range(frame_numbers[batch_idx[batch_num]], frame_numbers[batch_idx[batch_num+1]]))
        batch_box_numbers = list(range(batch_idx[batch_num]*nb_flies, batch_idx[batch_num+1]*nb_flies))
        logging.info(f"   loading frames.")
        frames = [frame[:, :, :1] for frame in vr[batch_frame_numbers]]  # keep only one color channel
        positions[batch_box_numbers, ...], confidence[batch_box_numbers, ...], _, bad_boxes[batch_box_numbers, ...], fly_id[batch_box_numbers], fly_frame[batch_box_numbers], _, fixed_angles[batch_box_numbers, ...] = process_batch(network, frames, box_centers[batch_frame_numbers, ...], box_angles[batch_frame_numbers, ...], box_size)
    # Saving data
    logging.info(f"   saving poses to {posePath}.")
    posedata = {'positions': positions, 'confidence': confidence, 'expID': expID, 'fixed_angles': fixed_angles,
                'frame_numbers': frame_numbers, 'fly_id': fly_id, 'fly_frame': fly_frame, 'bad_boxes': bad_boxes}
    dd.io.save(posePath, posedata)


def process_batch(network, frames, box_centers, box_angles, box_size):
    logging.info(f"   exporting boxes.")
    boxes, fly_id, fly_frame = export_boxes(frames, box_centers, box_size=box_size, box_angles=box_angles)
    logging.info(f"   predicting confidence maps for {boxes.shape[0]} boxes.")
    confmaps = predict_confmaps(network, normalize_boxes(boxes))
    logging.info(f"   processing confidence maps.")
    positions, confidence = process_confmaps_simple(confmaps)

    logging.info(f"   recalculating box angles.")
    # TODO: make positions self-documenting (named_list), make these args
    head_idx = 0
    tail_idx = 11
    nb_flies = box_angles.shape[1]


    ## ADRIAN TEMP FIX TO PROBLEMS WITH RESHAPING FOR/DURING ANGLE() ####################################################################
    # Older version:

    # newbox_angles, bad_boxes = detect_bad_boxes_by_angle(positions[:, head_idx:head_idx+1, :],
    #                                                      positions[:, tail_idx:tail_idx+1, :],
    #                                                      epsilon=5)

    ## ADRIAN TEMP FIX TO PROBLEMS WITH RESHAPING FOR/DURING ANGLE() ####################################################################
    # New version:

    newheads = np.zeros((int(positions.shape[0]/nb_flies),nb_flies,2))
    newtails = np.zeros(newheads.shape)
    newheads[:,0,:], newheads[:,1,:] = positions[::2,head_idx,:], positions[1::2,head_idx,:]
    newtails[:,0,:], newtails[:,1,:] = positions[::2,tail_idx,:], positions[1::2,tail_idx,:]
    newbox_angles, bad_boxes = detect_bad_boxes_by_angle(newheads, newtails, epsilon=5)

    #####################################################################################################################################


    # TODO: figure out better way of shaping things
    bad_boxes = flatten(bad_boxes)
    logging.info(f"   found {np.sum(bad_boxes)} cases of boxes with angles above threshold.")
    logging.info(f"   re-exporting boxes.")

    # TODO: Only re-process bad_boxes - but this will require reshaping things...


    ## ADRIAN TEMP FIX TO PROBLEMS WITH RESHAPING FOR/DURING ANGLE() ####################################################################
    # Older version:

    # fixed_angles = box_angles + unflatten(newbox_angles[..., 0], nb_flies)

    ## ADRIAN TEMP FIX TO PROBLEMS WITH RESHAPING FOR/DURING ANGLE() ####################################################################
    # New version:

    fixed_angles = box_angles + newbox_angles

    #####################################################################################################################################


    boxes, fly_id, fly_frame = export_boxes(frames,
                                            box_centers,
                                            box_size=np.array([120, 120]),
                                            box_angles=fixed_angles)
    # Final predictions
    logging.info(f"   re-doing predictions.")
    confmaps = predict_confmaps(network, normalize_boxes(boxes))
    logging.info(f"   re-processing confidence maps.")
    positions, confidence = process_confmaps_simple(confmaps)
    fixed_angles = flatten(fixed_angles)
    return positions, confidence, confmaps, bad_boxes, fly_id, fly_frame, boxes, fixed_angles


def fix_tracks(track_file_name: str, save_file_name: str):
    """Load data, call fix_orientations and save data."""
    logging.info(f"   processing tracks in {track_file_name}. will save to {save_file_name}")
    # read tracking data
    data = dd.io.load(track_file_name)
    # fix lines and get chaining IndexError
    data['lines'] = fix_orientations(data['lines'])
    logging.info(f"   saving chaining data to {save_file_name}")
    # saving fixed tracking data
    dd.io.save(save_file_name, data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    defopt.run(main)
