import logging
import numpy as np

from leap_utils.predict import load_network, predict_confmaps


logging.basicConfig(level=logging.DEBUG)

path_to_network = '/Volumes/ukme04/#Common/adrian/Adrian/models/181029_122243-n=1450/best_model.h5'


def test_load_network(path_to_network):
    # test load_model
    m = load_network(path_to_network)  # w/o resize
    print(m.output_shape)
    assert m.output_shape == (None, 120, 120, 12)

    m = load_network(path_to_network, input_shape=(500, 500, 1))  # w/ resize
    assert np.all(m.output_shape == (None, 500, 500, 12))


def test_predict_confmaps(path_to_network):
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


test_load_network(path_to_network)
test_predict_confmaps(path_to_network)
