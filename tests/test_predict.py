import logging
import numpy as np
import h5py
import os

from leap_utils.preprocessing import normalize_matlab_boxes
from leap_utils.predict import load_network, predict_confmaps


logging.basicConfig(level=logging.DEBUG)

if os.name == 'nt':
    path_to_network = r'tests\data\model.h5'
    path_to_boxes = r'tests\data\boxes_from_matlab.h5'
else:
    path_to_network = 'tests/data/model.h5'
    path_to_boxes = 'tests/data/boxes_from_matlab.h5'

def test_load_network():
    # test load_model
    m = load_network(path_to_network)  # w/o resize
    print(m.output_shape, m.input_shape)
    assert m.output_shape == (None, 120, 120, 12)

    m = load_network(path_to_network, input_shape=(500, 500, 1))  # w/ resize
    print(m.output_shape, m.input_shape)
    assert np.all(m.output_shape == (None, 500, 500, 12))


def test_predict_confmaps():
    boxes = np.zeros((120, 120, 1), dtype=np.uint8)
    network = load_network(path_to_network, input_shape=boxes.shape)  # w/ resize
    try:
        confmaps = predict_confmaps(network, boxes)
        raised_error = False
    except ValueError:
        raised_error = True
    assert raised_error

    boxes = np.zeros((1, 100, 100, 1), dtype=np.uint8)
    confmaps = predict_confmaps(path_to_network, boxes)
    assert np.all(confmaps.shape == (1, 100, 100, 12))

    boxes = np.zeros((10, 120, 120, 1), dtype=np.uint8)
    confmaps = predict_confmaps(network, boxes)
    assert np.all(confmaps.shape == (10, 120, 120, 12))


def test_predict_confmaps_data():
    import matplotlib.pyplot as plt
    plt.ion()
    with h5py.File(path_to_boxes, 'r') as f:
        boxes = f['box'][:]
    boxes = normalize_matlab_boxes(boxes)
    plt.figure(1)
    plt.imshow(boxes[-1, ..., 0])
    plt.show()
    plt.pause(0.01)
    network = load_network(path_to_network, input_shape=boxes.shape[1:])  # w/ resize
    confmaps = predict_confmaps(network, boxes)
    plt.figure(2)
    for prt in range(12):
        plt.subplot(3, 4, prt + 1)
        plt.imshow(confmaps[-1, ..., prt])
        plt.show()
        plt.pause(0.01)
    plt.pause(2)
