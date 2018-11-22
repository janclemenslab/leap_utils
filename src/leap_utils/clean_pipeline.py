import os
import defopt
import logging
import deepdish as dd
import numpy as np
from videoreader import VideoReader
from leap_utils.preprocessing import export_boxes, angles, normalize_boxes, detect_bad_boxes_by_angle, fix_orientations, nframes2nboxes
from leap_utils.postprocessing import process_confmaps_simple
from leap_utils.predict import predict_confmaps
from leap_utils.utils import iswin,  ismac

# Plot stuff
play_boxes = True
inspect_flies = False

# Paths
if iswin():
    root = 'Z:/#Common/'
elif ismac():
    root = '/Volumes/ukme04/#Common/'
else:
    root = '/scratch/clemens10/'
dataPath = root+'chainingmic/dat.processed'
resPath = root+'chainingmic/res'
networkPath = root+'chainingmic/dat.processed/best_model.h5'


def main(expID: str, *, frame_start: int = 0, frame_stop: int = None, frame_step: int = 1, batch_size: int = 10):
    # Fix tracks
    # Paths
    trackPath = f"{dataPath}/{expID}/{expID}_tracks.h5"
    videoPath = f"{dataPath}/{expID}/{expID}.mp4"
    trackfixedPath = f"{resPath}/{expID}//{expID}_tracks_fixed.h5"
    savingPath = f"{resPath}/{expID}"
    posePath = f"{savingPath}/{expID}_pose.h5"
    fixedBoxesPath = f"{savingPath}/{expID}_fixedBoxes.h5"

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
        logging.info(f'   Setting frame_stop: {0}.'.format(frame_stop))
    frame_numbers = range(frame_start, frame_stop, frame_step)
    batch_idx = list(range(0, len(frame_numbers), batch_size))
    batch_idx.append(frame_stop-1)
    logging.info(f"   frame range: {frame_start}:{frame_stop}:{frame_step}.")
    logging.info(f"   processing in {len(batch_idx)} batches of size {batch_size}.")
    # pre-allocate all data stbatch_idx[batch_num+1]ructures
    box_size = [120, 120]
    nb_frames = len(frame_numbers)
    nb_parts = 12
    nb_boxes = nb_frames * nb_flies
    positions = np.zeros((nb_boxes, nb_parts, 2))
    confidence = np.zeros((nb_boxes, nb_parts, 1))
    confmaps = np.zeros((nb_boxes, *box_size, nb_parts))
    bad_boxes = np.zeros((nb_boxes, 1))
    fly_id = np.zeros((nb_boxes,))
    fly_frame = np.zeros((nb_boxes,))
    boxes = np.zeros((nb_boxes, *box_size, 3))

    for batch_num in range(len(batch_idx)-1):
        logging.info(f"PROCESSING BATCH {batch_num}.")
        batch_frame_numbers = list(range(frame_numbers[batch_idx[batch_num]], frame_numbers[batch_idx[batch_num+1]]))
        batch_box_numbers = list(range(batch_idx[batch_num]*nb_flies, batch_idx[batch_num+1]*nb_flies))

        logging.info(f"   loading frames.")
        frames = list(vr[batch_frame_numbers])
        positions[batch_box_numbers, ...], confidence[batch_box_numbers, ...], confmaps[batch_box_numbers, ...], bad_boxes[batch_box_numbers, ...], fly_id[batch_box_numbers], fly_frame[batch_box_numbers], boxes[batch_box_numbers, ...] = process_batch(frames, box_centers[batch_frame_numbers, ...], box_angles[batch_frame_numbers, ...], box_size)
    # Saving data
    logging.info(f"   saving poses to: {posePath}.")
    posedata = {'positions': positions, 'confidence': confidence, 'confmaps': confmaps, 'expID': expID,
                'frame_numbers': frame_numbers, 'fly_id': fly_id, 'fly_frame': fly_frame, 'bad_boxes': bad_boxes}
    dd.io.save(posePath, posedata)
    logging.info(f"   saving fixed boxes to: {fixedBoxesPath}.")
    fixedBoxesdata = {'boxes': boxes, 'fly_id': fly_id, 'fly_frame': fly_frame}
    dd.io.save(fixedBoxesPath, fixedBoxesdata)


def process_batch(frames, box_centers, box_angles, box_size):
    logging.info(f"   exporting boxes.")
    boxes, fly_id, fly_frame = export_boxes(frames, box_centers, box_size=box_size, box_angles=box_angles)
    boxes = normalize_boxes(boxes)

    logging.info(f"   predicting confidence maps for {boxes.shape[0]} boxes.")
    confmaps = predict_confmaps(networkPath, boxes[:, :, :, :1])
    logging.info(f"   processing confidence maps.")
    positions, confidence = process_confmaps_simple(confmaps)

    logging.info(f"   recalculating box angles.")
    newbox_angles, bad_boxes = detect_bad_boxes_by_angle(positions, epsilon=5)
    bad_boxes = nframes2nboxes(bad_boxes)
    logging.info(f"   found {np.sum(bad_boxes)} cases of boxes with angles above threshold.")
    logging.info(f"   re-exporting boxes.")
    # TODO: Only re-process bad_boxes
    boxes, fly_id, fly_frame = export_boxes(frames,
                                            box_centers,
                                            box_size=np.array([120, 120]),
                                            box_angles=box_angles + newbox_angles)
    boxes = normalize_boxes(boxes)

    # Final predictions
    logging.info(f"   re-doing predictions.")
    confmaps = predict_confmaps(networkPath, boxes[:, :, :, :1])
    logging.info(f"   re-processing confidence maps.")
    positions, confidence = process_confmaps_simple(confmaps)
    return positions, confidence, confmaps, bad_boxes, fly_id, fly_frame, boxes


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
