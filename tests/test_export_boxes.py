import os
import numpy as np
import matplotlib.pyplot as plt
from videoreader import VideoReader

from . import temp_dir

from leap_utils.preprocessing import crop_frame, export_boxes


plt.ion()
path_to_video = os.path.join(temp_dir, 'video_short.mp4')


def test_crop_frame():
    frame = np.zeros((251, 211, 3), dtype=np.uint8)

    # check that size matches
    center = np.array([100, 100], dtype=np.uintp)
    box_size = np.array([50, 50], dtype=np.uintp)
    assert np.all(crop_frame(frame, center, box_size).shape[:2] == box_size)
    box_size = np.array([51, 51], dtype=np.uintp)
    assert crop_frame(frame, center, box_size).shape[2] == frame.shape[2]
    assert np.all(crop_frame(frame, center, box_size).shape[:2] == tuple(box_size))

    # check overflow
    center = np.array([10, 100], dtype=np.intp)
    box_size = np.array([40, 50], dtype=np.intp)
    assert np.all(crop_frame(frame, center, box_size).shape[:2] == box_size)
    try:
        crop_frame(frame, center, box_size, mode='raise')
        error_raised = False
    except:
        error_raised = True
    # assert error_raised


def test_export_boxes():
    vr = VideoReader(path_to_video)
    frames = list(vr[100:110])

    nb_flies = 3
    box_centers = 400 + np.ones((1000, nb_flies, 2))
    box_centers[100, 0, :] = [10, 10]
    box_centers[100, 2, :] = [900, 900]

    box_angles = np.zeros((1000, nb_flies, 1))
    box_angles[101, 0, 0] = 45
    box_angles[101, 2, 0] = -60

    box_size = np.array([100, 100])

    boxes, fly_id, fly_frames = export_boxes(frames, box_centers, box_angles=box_angles, box_size=box_size)
    assert np.all(boxes.shape == (len(frames) * nb_flies, *box_size, vr.frame_channels))
    boxes, fly_id, fly_frames = export_boxes(frames, box_centers, box_angles=box_angles, box_size=box_size)
    assert np.all(boxes.shape == (len(frames) * nb_flies, *box_size, vr.frame_channels))
    plt.subplot(4, 1, 1)
    plt.plot(fly_id)
    plt.subplot(4, 1, 2)
    plt.plot(fly_frames)
    plt.subplot(4, 3, 7)
    plt.imshow(boxes[0, :, :, 0])
    plt.subplot(4, 3, 8)
    plt.imshow(boxes[1, :, :, 0])
    plt.subplot(4, 3, 9)
    plt.imshow(boxes[2, :, :, 0])
    plt.subplot(4, 3, 10)
    plt.imshow(boxes[3, :, :, 0])
    plt.subplot(4, 3, 11)
    plt.imshow(boxes[4, :, :, 0])
    plt.subplot(4, 3, 12)
    plt.imshow(boxes[5, :, :, 0])
    plt.pause(2)
