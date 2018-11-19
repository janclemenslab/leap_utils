from typing import Sequence, Union
import logging

import keras
import numpy as np


def load_network(model_path: str, weights_path: str = None, input_shape: Sequence = None):
    """Load network from file.

    Args:
        model_path - save with keras.save_model
        weights_path - optional - load custom weights
        input_shape - optional - re-compile network to accept different inputs
    Returns:
        keras model
    """
    logging.info(f"loading model architecture from {model_path}")
    from keras.models import load_model as keras_load_model
    from keras.models import Model
    from keras.layers import Input

    m = keras_load_model(model_path)
    if weights_path:
        logging.info(f"loading model weights from {weights_path}")
        m.load_weights(weights_path)
    if input_shape and not np.all(input_shape == m.layers[0].input_shape[1:]):
        logging.info(f"changing input shape from {m.layers[0].input_shape[1:]} to {input_shape}")
        newInput = Input(batch_shape=(None, *input_shape))
        newOutputs = m(newInput)
        m = Model(newInput, newOutputs)
    return m


def predict_confmaps(network: Union[str, keras.models.Model], boxes: np.array) -> np.array:
    """Predict confidence maps from images.

    Args:
        model: keras model
        images: image - [nbox, w, h, chans]
    Returns:
        confmaps

    """
    if boxes.ndim is not 4:
        raise ValueError(f'`boxes` needs to be 4D - (nboxes, width, height, channels) - but has shape {boxes.shape}.')

    if isinstance(network, str):
        network = load_network(network, input_shape=boxes.shape[1:4])  # this should return a compiled network

    confmaps = network.predict_on_batch(boxes)

    return confmaps
