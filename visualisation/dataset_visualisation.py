import sys, os, argparse, torch
sys.path.insert(0, '../')
from data import load_data, generate_loaders
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import nibabel as nib
import numpy as np


def visualize_dataset(data_dir, modality='mri', threeD=True, save_as_nifti=False):
    tensors = load_data(fname=os.path.join(data_dir, 'data_set.npz'))

    if save_as_nifti:
        reference_img = nib.load(os.path.join(data_dir, 'reference.nii'))
        coordinate_space = reference_img.affine

    n_subj = tensors[0].shape[0]
    print('Processing ', n_subj, 'subjects')

    settings = ['train', 'test']

    if threeD:
        batch_size = 1
    else:
        batch_size = tensors[0].shape[2]


    ct_sets, mri_sets = generate_loaders(tensors, batch_size=batch_size, threeD=threeD)

    print('Using modality', modality)
    if modality == 'mri':
        sets = mri_sets
    elif modality == 'ct':
        sets = ct_sets
    else:
        print('Modality has to be one of: ct, mri.', modality, 'is not valid.')
        return

    for setting in settings:
        loader = sets[setting]
        list = [(inputs, outputs) for (inputs, outputs) in tqdm(loader)]

        plt.switch_backend('agg')
        ncol = 6
        nrow = n_subj + 2
        figure = plt.figure(figsize=(ncol + 1, nrow + 1))
        gs = gridspec.GridSpec(nrow, ncol,
                               wspace=1, hspace=0.25,
                               top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                               left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
        print('Batches:', len(list))

        for subj in range(len(list)):
            subj_data = list[subj]
            # print('i/o', len(subj_data))
            # input data (shape n_batch, n_c, x ,y , z)
            # print('input', subj_data[0].shape)
            if not threeD:
                subj_data = (subj_data[0].permute(1, 2, 3, 0).unsqueeze(0), subj_data[1].permute(1, 2, 3, 0).unsqueeze(0))
            print('input', subj_data[0].shape)

            # visual_add(np.empty((3, 3, 3)), i_subj, 0, gs, subject)

            for channel in range(subj_data[0].shape[1]):
                non_zero_idx = subj_data[0][0, channel].nonzero(as_tuple=True)
                print(non_zero_idx[0], subj_data[0][0, channel].shape)
                mean_voxel_value = subj_data[0][0, channel][non_zero_idx].mean().item()
                print(mean_voxel_value, type(subj_data[0][0, channel]))
                std_voxel_value = subj_data[0][0, channel].std().item()
                visual_add_center(subj_data[0][0, channel], subj, channel, gs, image_id=str(mean_voxel_value)[0:5])
                if save_as_nifti:
                    binary_img = nib.Nifti1Image(subj_data[0].squeeze().permute(1,2,3,0).numpy(), affine=coordinate_space)
                    nib.save(binary_img, os.path.join(data_dir, str(subj) + '_' + setting + '_mri.nii'))
            # add output
            visual_add_center(subj_data[1][0, 0], subj, channel + 1, gs)
            if save_as_nifti:
                binary_img = nib.Nifti1Image(subj_data[1][0, 0].numpy(), affine=coordinate_space)
                nib.save(binary_img, os.path.join(data_dir, str(subj) + '_' + setting + '_GT.nii'))


        plt.ioff()
        plt.switch_backend('agg')
        figure_path = os.path.join(data_dir, modality + '_' + setting + '_dataloader_visualisation.svg')
        figure.savefig(figure_path, dpi='figure', format='svg')
        plt.close(figure)

# draw image slice on canvas
def visual_add(image, i_subj, i_image, gs, image_id=None):
    i_col = i_image
    i_row = i_subj

    # plot image
    ax = plt.subplot(gs[i_row, i_col])
    if image_id is not None: ax.set_title(image_id, fontdict={'fontsize': 10})
    plt.imshow(-image.T)
    plt.set_cmap('Greys')
    plt.axis('off')

# draw center image on canvas
def visual_add_center(image, i_subj, i_image, gs, image_id=None):
    n_z = image.shape[2]
    center_z = (n_z - 1) // 2
    i_col = i_image
    i_row = i_subj

    # plot image
    ax = plt.subplot(gs[i_row, i_col])
    if image_id is not None: ax.set_title(image_id, fontdict={'fontsize': 10})
    plt.imshow(-image[:, :, center_z].T)
    plt.set_cmap('Greys')
    plt.axis('off')


def plot_slices(list, gs):
    slices_in, slices_out = list[0]
    i_slice = 0
    for i_slice, slice in enumerate(slices_in):
        i_image = 0

        for channel in range(slice.shape[0]):
            visual_add(slice[channel], i_slice, i_image, gs, '')
            i_image += 1

        visual_add(slices_out[i_slice, 0], i_slice, i_image, gs, '')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_location')
    parser.add_argument("--modality", "-m", help='mri or ct?',
                        type=str, default='mri')
    args = parser.parse_args()

    visualize_dataset(args.dataset_location, args.modality)
