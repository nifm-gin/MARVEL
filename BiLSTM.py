###############################
## BiLSTM network for MARVEL ##
###############################

from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Concatenate, Dropout, Input, Dense, Flatten, LSTM, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical


def Bidirectional_LSTM(layer_shapes: List[int], activations: List[str]) -> Sequential:
    """ A bidirectional recurrent neural network. 

    Parameters
    ----------
    layer_shapes: List[int]
        List of layer lengths. 
    activations: List[str]
        List of activation functions, with 2 less items than layer_shapes. 

    Returns
    -------
    NN: tensorflow neural network. 
        The BiLSTM neural network. 
    """

    # BiLSTM layer
    lst_layers = [
        Reshape((layer_shapes[0], 1)),
        Bidirectional(LSTM(layer_shapes[1], input_shape=(
            layer_shapes[0], 1), activation=activations[0], return_sequences=False))
    ]

    # following dense layers
    for (layer_shape, activation) in zip(layer_shapes[2:-1], activations[1:]):
        lst_layers.append(Dense(layer_shape, activation=activation))
    lst_layers.append(Dense(layer_shapes[-1]))

    return Sequential(lst_layers)
