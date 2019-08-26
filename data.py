import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def standardise(imgX):
    original_shape = imgX.shape
    imgX = imgX.reshape(-1, imgX.shape[-1])
    scaler = StandardScaler(copy = False)
    rescaled_imgX = scaler.fit_transform(imgX).reshape(original_shape)
    return rescaled_imgX

def load_data(fname='/home/julian/withDWI_all_2016_2017/standardized_data_set.npz'):

    raw_data = np.load(fname)
    # Loading the keys of interest from the dataset file
    ct_inputs, ct_lesions, mri_inputs, mri_lesions, masks = [raw_data[x] for x in [
        'ct_inputs',
        'ct_lesion_GT',
        'mri_inputs',
        'mri_lesion_GT',
        'brain_masks'
    ]]

    # Preprocessing functions
    reorder_channels = lambda x: x.transpose(0, 4, 1, 2, 3)
    add_mask = lambda x: np.concatenate([x, masks[:, None]], axis=1)
    pad_inputs = lambda x: np.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1), (0, 1)),
                        'edge')
    pad_outputs = lambda x: np.pad(x, ((0, 0), (0, 1), (0, 1), (0, 1)),
                                   'edge')[:, None]
    to_torch = lambda x: torch.from_numpy(x.astype('float32')).float()
    bring_z = lambda x: x.transpose(0, 4, 1, 2, 3)
    apply_mask = lambda x: masks * x

    # Preprocessing the data
    reordered = reorder_channels(ct_inputs)
    ct_inputs = pad_inputs(add_mask(reorder_channels(ct_inputs)))
    mri_inputs = pad_inputs(reorder_channels(mri_inputs))
    ct_lesions = pad_outputs(apply_mask(ct_lesions))
    mri_lesions = pad_outputs(mri_lesions)
    masks = pad_outputs(masks)

    # Converting all tensors to the Torch format
    return [to_torch(bring_z(x)) for x in [ct_inputs, ct_lesions, mri_inputs, mri_lesions, masks]]

def generate_loaders(tensors, test_size_ratio=0.2, incremental_set_ratio=0.3,
                     seed=0, batch_size=2, use_increment_set=True):

    ct_inputs, ct_lesions, mri_inputs, mri_lesions, masks = tensors
    subject_count = mri_inputs.shape[0]

    # Suffling all the indices
    indices = np.random.RandomState(seed=seed).permutation(subject_count)
    indices = torch.from_numpy(indices)

    test_set_size = int(test_size_ratio * subject_count)
    incremental_set_size = int(incremental_set_ratio * subject_count)

    test_set_indices = indices[:test_set_size]
    incremental_set_indices = indices[test_set_size:][:incremental_set_size]
    train_set_indices = indices[test_set_size + incremental_set_size:]

    flatten_data = lambda x: x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

    def generate_set(indices, inputs, outputs, is_train=False):
        sel_inputs = inputs[indices]
        sel_outputs = outputs[indices]

        ds = TensorDataset(flatten_data(sel_inputs), flatten_data(sel_outputs))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=is_train,
                        pin_memory=True)
        return dl

    if not use_increment_set:
        train_set_indices = torch.cat([train_set_indices ,incremental_set_indices])

    ct_sets = {
        'train': generate_set(train_set_indices, ct_inputs, ct_lesions, True),
        'incremental': generate_set(incremental_set_indices, ct_inputs, ct_lesions),
        'test': generate_set(test_set_indices, ct_inputs, ct_lesions)
    }

    mri_sets = {
        'train': generate_set(torch.cat([train_set_indices ,incremental_set_indices]),
                              mri_inputs, mri_lesions, True),
        'test': generate_set(test_set_indices, mri_inputs, mri_lesions)
    }

    return ct_sets, mri_sets
