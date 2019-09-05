import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import numpy as np

predictions = torch.load('/Users/julian/predictions.pth', map_location=torch.device('cpu'))
lesions = torch.load('/Users/julian/lesions.pth', map_location=torch.device('cpu'))


plt.switch_backend('agg')
ncol = 2
nrow = len(predictions) + 2
figure = plt.figure(figsize=(ncol + 1, nrow + 1))
gs = gridspec.GridSpec(nrow, ncol,
                       wspace=1, hspace=0.25,
                       top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                       left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))


# %%
# draw image on canvas
def visual_add(image, i_subj, i_image, gs, image_id=None):
    i_col = i_image
    i_row = i_subj

    # plot image
    ax = plt.subplot(gs[i_row, i_col])
    if image_id is not None: ax.set_title(image_id, fontdict={'fontsize': 10})
    plt.imshow(-image.T)
    plt.set_cmap('Greys')
    plt.axis('off')


# %%
i_slice = 0
for i_slice, pred in enumerate(predictions):
    visual_add(np.squeeze(lesions[i_slice].detach().numpy()), i_slice, 0, gs, '')
    visual_add(np.squeeze(pred.detach().numpy()), i_slice, 1, gs, '')
    i_slice += 1


import os

data_dir = '/Users/julian/temp'
plt.ioff()
plt.switch_backend('agg')
figure_path = os.path.join(data_dir, 'prediction_visualisation.png')
figure.savefig(figure_path, dpi='figure', format='png')
plt.close(figure)