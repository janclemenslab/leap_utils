from leap_utils.preprocessing import detect_bad_boxes_by_angle
import numpy as np


def test_detect_bad_boxes_by_angle():
    pos = np.random.randint(0, 100, (10, 2, 2))
    fly_angles, bad_boxes = detect_bad_boxes_by_angle(pos[:, 0:1, :], pos[:, 1:2, :])
    # print(fly_angles.shape)
    # print(bad_boxes.shape)
