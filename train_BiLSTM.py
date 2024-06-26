##########################
## Train BiLSTM network ##
##########################

# WARNING! This code requires several functions. See README.md

from BiLSTM import Bidirectional_LSTM
from sklearn.model_selection import train_test_split
import json
import sys
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_network_main.py <training_infos.json>")
        sys.exit(0)

    TRAIN_INFOS = json.load(open(sys.argv[1]))

    saving_dir = os.path.dirname(sys.argv[1])
    saving_dir_fig = os.path.join(saving_dir, 'figures')
    os.makedirs(saving_dir_fig, exist_ok=True)

    saving_weights_dir = os.path.join(saving_dir, 'weights')
    os.makedirs(saving_weights_dir, exist_ok=True)


# PATHS
saving_dir = ''
saving_dir_fig = ''
saving_weights_dir = ''

dico_filepath = ''
vessel_distrib_filepath = ''

# NETWORK PARAMS
hidden_shapes = [100, 75, 50]
activations = ['tanh', 'relu', 'sigmoid']
learning_rate = 0.001
decrease_LR = 0.8
n_epochs = 200
epochs_per_DICO = 5
batch_size = 64
SNR_range = [1, 20]

# LOAD BASE DICTIONARY
DICO_base_params, DICO_base_signals = load_dictionnary(dico_filepath)

label_parameters = ['T1', 'T2', 'B1', 'df', 'CBV', 'R']
n_parameters = len(label_parameters)
n_signals, n_pulses = DICO_base_signals.shape

# LOAD VESSEL DISTRUBUTIONS

vasc_params, vasc_distribs = load_vessel_distrib(vessel_distrib_filepath)

# LOAD NETWORK

layer_shapes = [n_pulses] + hidden_shapes + [n_parameters]
NN = Bidirectional_LSTM(layer_shapes, activations)
NN.build(input_shape=(None, n_pulses))
# Compile the model
NN.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse',
           metrics=[tf.keras.metrics.MeanSquaredError()])

# UTILS FUNCTIONS


def add_Gaussian_noise_to_DICO(DICO_base_signals, SNR_range):
    SNR = SNR_range[0] + (SNR_range[1]-SNR_range[0]) * \
        np.random.rand(DICO_base_signals.shape[0])[:, None]

    noise = np.random.randn(
        DICO_base_signals.shape[0], DICO_base_signals.shape[1])
    noise = noise * np.mean(abs(DICO_base_signals)) / SNR

    return DICO_base_signals + noise


# LAUNCH TRAIN
best_loss = np.inf
lst_loss, lst_val_loss = [], []
for epoch in range(n_epochs):

    # simulate new vasc dictionary
    if epoch % epochs_per_DICO == 0:
        print("\nEpoch {}. Generating new training dictionary. ".format(epoch+1))

        DICO_vasc_params, DICO_vasc_signals = convol_base_dico(
            DICO_base_params, DICO_base_signals, vasc_params, vasc_distribs)

    # add noise to dictionary
    X = add_Gaussian_noise_to_DICO(DICO_vasc_signals, SNR_range)

    # normalize signals of the dictionary
    X /= np.linalg.norm(X, axis=1)[:, None]

    # generate training datasets
    Y = DICO_vasc_params
    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    history = NN.fit(x_train, y_train, validation_data=(
        x_test, y_test), batch_size=batch_size, shuffle=True, epochs=1)

    lst_loss.append(history.history['loss'][0])
    lst_val_loss.append(history.history['val_loss'][0])
    # save weights if new best validation loss
    if lst_val_loss[-1] < best_loss:
        best_loss = lst_val_loss[-1]
        adding_text = '_' + str(epoch+e) + 'epochs'
        NN.save(os.path.join(saving_weights_dir,
                'model_{}epochs.h5'.format(epoch)))

    # update learning rate
    if epoch % epochs_per_DICO == 0:
        learning_rate *= decrease_LR
        NN.compile(optimizer=Adam(learning_rate=learning_rate),
                   loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
