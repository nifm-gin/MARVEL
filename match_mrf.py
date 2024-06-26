import numpy as np
import numpy.ma as ma
import time
from jax import jit, vmap
import jax.numpy as jnp
import nibabel as nib
from functools import partial
from tqdm import tqdm
import h5py


def match_full(Xobs, roi_data, MRSignals, Par, Labels, batch_size):
    print('Preprocess...')
    # Mask input nifti with ROI
    roi_mask = (roi_data > 0.9)
    if len(roi_mask.shape) == 2:
        roi_mask = np.expand_dims(roi_mask, axis=2)
        # Replicate ROI mask along the fourth dimension to match the number of time frames
        roi_mask_expanded = np.repeat(
            roi_mask[:, :, np.newaxis], Xobs.shape[-1], axis=-1)
    else:
        roi_mask_expanded = np.repeat(
            roi_mask[:, :, :, np.newaxis], Xobs.shape[-1], axis=-1)

    Xobs_crop = ma.zeros((Xobs.shape))

    for t in range(0, Xobs.shape[-1]):
        im = ma.masked_where(~roi_mask_expanded[:, :, :, t], Xobs[:, :, :, t])
        Xobs_crop[:, :, :, t] = im

    Xobs = ma.filled(Xobs_crop, np.nan)

    # Options

    parameters = ['T1', 'T2', 'df', 'gamma', 'B1rel', 'Vf', 'R']
    make_score_map = True
    make_matched_vol = True
    complex_data = False  # Complex matching
    svd = False  # Compress input image using SVD dictionary
    pd = True  # compute Proton Density map
    grad = False  # generate gradient map
    method = 'DBM'  # 'DBM', 'DBMonFly', 'DBL'

    MRSignals = abs(MRSignals)
    Labels = [label.replace(" ", "") for label in Labels]
    parameter_indices = [Labels.index(
        param) if param in Labels else None for param in parameters]

    norm_Xobs = np.linalg.norm(Xobs, ord=2, axis=3)
    norm_Xobs = np.expand_dims(norm_Xobs, axis=-1)

    if pd and method != 'DBMonFly':
        Xobs_notnorm = Xobs
        Xobs_notnorm_mean = np.mean(Xobs_notnorm, axis=3)

    Xobs = Xobs / norm_Xobs

    if method != 'DBMonFly':
        norm_dico = np.linalg.norm(MRSignals, ord=2, axis=1)
        Dico_notnorm = MRSignals
        MRSignals = MRSignals / norm_dico[:, np.newaxis]

    ###########################################################################
    tic = time.time()
    MRI_map = Xobs
    n_signals, n_pulses = MRSignals.shape
    L, l, s, _ = MRI_map.shape

    MRI_map_reshaped = MRI_map.reshape(-1, n_pulses)
    test = MRI_map_reshaped.T
    print('Matching...')
    matched_indices = []
    score_map = []

    # Calculate the total number of iterations including the main loop and the additional loop for remaining elements
    total_iterations = (len(MRI_map_reshaped) // batch_size) + \
        (1 if len(MRI_map_reshaped) % batch_size != 0 else 0)

    # Initialize tqdm progress bar with the total number of iterations
    for k in tqdm(range(0, len(MRI_map_reshaped) - batch_size + 1, batch_size), total=total_iterations):
        matched_indice, score = match(
            MRI_map_reshaped[k:(k + batch_size)], MRSignals)
        matched_indices.append(matched_indice)
        score_map.append(score)

    # Handling remaining elements if len(MRI_map_reshaped) is not a multiple of batch_size
    remaining_elements = len(MRI_map_reshaped) % batch_size
    if remaining_elements > 0:
        matched_indice, score = match(
            MRI_map_reshaped[-remaining_elements:], MRSignals)
        matched_indices.append(matched_indice)
        score_map.append(score)
    print('Computing maps...')
    flattened_indices = [
        item for sublist in matched_indices for item in sublist]
    matched_indices = jnp.array(flattened_indices)
    flattened_score = [item for sublist in score_map for item in sublist]
    matched_volume_map = MRSignals[matched_indices]
    matched_volume_map = matched_volume_map.reshape(L, l, s, n_pulses)
    score_map = jnp.array(flattened_score).reshape(L, l, s)

    if pd and method != 'DBMonFly':
        minidico = Dico_notnorm[matched_indices, :]
        minidicomean = jnp.mean(minidico, axis=-1)
        PDmap = Xobs_notnorm_mean / minidicomean.reshape(L, l, -1)
    output_maps = {}

    for p in range(len(parameters)):
        map_name = prefix + parameters[p]
        ind = parameter_indices[p]
        map_param = Par[matched_indices, ind]
        map_param = map_param.reshape(L, l, s)
        if parameters[p] == 'Vf' or parameters[p] == 'SO2':
            map_param *= 1e2
        elif parameters[p] == 'R':
            map_param *= 1e6
        output_maps[map_name] = map_param
    toc = time.time()
    print('Reconstruction time: ', toc - tic, 'secondes')
    return matched_indices, score_map, matched_volume_map, output_maps, PDmap


@partial(vmap, in_axes=(None, 0,))
def dot(x, y):
    return jnp.dot(x, y)


@jit
@partial(vmap, in_axes=(0, None,))
def match(y, X):
    d = dot(y, X)
    i_max = d.argmax()
    val_max = d[i_max]
    return i_max, val_max


def save_nii(score_map, matched_volume_map, output_maps, PDmap, seq_name, prefix, input_img):
    print('Saving niftis...')
    output_path = seq_name + prefix + 'ScoreMap.nii'
    score_img = nib.Nifti1Image(
        score_map, input_img.affine, header=input_img.header)
    nib.save(score_img, output_path)

    output_path = seq_name + prefix + 'PD.nii'
    PD_img = nib.Nifti1Image(PDmap, input_img.affine, header=input_img.header)
    nib.save(PD_img, output_path)

    output_path = seq_name + prefix + 'MatchedVol.nii'
    MV_img = nib.Nifti1Image(
        matched_volume_map, input_img.affine, header=input_img.header)
    nib.save(MV_img, output_path)

    for name, mapp in output_maps.items():
        output_path = seq_name + name + '.nii'
        map_img = nib.Nifti1Image(
            mapp, input_img.affine, header=input_img.header)
        nib.save(map_img, output_path)


tic = time.time()
dico_path = 'DICO_vasc.h5'
f = h5py.File(dico_path, 'r')
print(f.keys())
Par = np.array(f.get('Parameters'))
MRSignals = np.array(f.get('MRSignals'))
Labels = ['T1', 'T2', 'df', 'gamma', 'B1rel', 'SO2', 'Vf', 'R']
MRSignals = MRSignals[:, :, 0]
print('Labels:', Labels)
toc = time.time()
print('Loading of the dico :', toc - tic, 'secondes')

roi_path = 'brain.nii'
roi_img = nib.load(roi_path)
roi_data = roi_img.get_fdata()

prefix = 'match_'  # choose a prefix for the output maps
seq_name = 'test_match_vascular'
input_path = 'acquisition.nii'
input_img = nib.load(input_path)
Xobs = input_img.get_fdata()

matched_indices, score_map, matched_volume_map, output_maps, PDmap = match_full(Xobs, roi_data, MRSignals, Par, Labels,
                                                                                batch_size=100)

save_nii(score_map, matched_volume_map, output_maps,
         PDmap, seq_name, prefix, input_img)
