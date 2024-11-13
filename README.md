# MARVEL

## Attached code for the paper MARVEL: MR Fingerprinting with Additional micRoVascular Estimates using bidirectional LSTMs


This repository contains attached code to our MICCAI 2024 paper MARVEL ([preprint](https://arxiv.org/abs/2407.10512) and [published](https://link.springer.com/chapter/10.1007/978-3-031-72069-7_25) versions). Please do not hesitate to contact us in case of any trouble dealing with our code. 


## Requirements


For network training, we used a Python 3.8 virtual environment with dependencies listed in the `network_requirements.txt`. 

For matching, we used a Python 3.11 virtual environment with dependencies listed in the `match_requirements.txt`. 


## Training

To train the network, use `train_BiLSTM.py`. The code requires the following functions that you need to implement depending on your data pipeline: 

- `load_dictionnary(path)`: function that loads your dictionary and returns two numpy arrays, one with its parameter values of shape `(n_signals, n_parameters)` and the second with its signals of shape `(n_signals, n_pulses)`

- `load_vessl_distrib(path)`: function that loads your vessel distributions and returns two numpy arrays, one with its vascular parameter values of shape `(n_distributions, n_parameters)` and the second with its frequency df distribution coefficients of shape `(n_distributions, n_df)`

- `convol_base_dico(DICO_base_params, DICO_base_signals, vasc_params, vasc_distribs)`: function that convolves the base dictionary along the df axis to produce the vascular dictionary. Returns two numpy arrays, one with its parameter values of shape `(n_signals, n_parameters)` and the second with its signals of shape `(n_signals, n_pulses)`


If you use our code, please cite :

`Barrier, A., Coudert, T., Delphin, A., Lemasson, B., Christen, T. (2024). MARVEL: MR Fingerprinting with Additional micRoVascular Estimates Using Bidirectional LSTMs. In: Linguraru, M.G., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. MICCAI 2024. Lecture Notes in Computer Science, vol 15002. Springer, Cham. https://doi.org/10.1007/978-3-031-72069-7_25`

