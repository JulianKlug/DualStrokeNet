import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch, os
import numpy as np
from metrics import dice_score, FocalTverskyLoss

data_dir = '/Users/julian/temp/dual_net_models/singleChannel_combi_loss/model_prediction'
setting = 'train'
inputs = torch.load(os.path.join(data_dir, setting + '_input.pth'), map_location=torch.device('cpu'))
predictions = torch.load(os.path.join(data_dir, setting + '_predictions.pth'), map_location=torch.device('cpu'))
lesions = torch.load(os.path.join(data_dir, setting + '_GT.pth'), map_location=torch.device('cpu'))

# as everything was processed in one batch, there is only one item in the batch
inputs = inputs[0]
predictions = predictions[0]
lesions = lesions[0]

n_z = inputs.shape[-1]
n_c = inputs.shape[1]


plt.switch_backend('agg')
ncol = 9
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
    plt.imshow(image.T)
    plt.set_cmap('gray')
    plt.axis('off')


# %%
i_slice = 0
for i_slice, pred in enumerate(predictions):
    i_col = 0
    for channel in range(n_c):
        visual_add(np.squeeze(inputs[i_slice, channel, ..., int(n_z/2)].detach().numpy()), i_slice, i_col, gs, '')
        i_col += 1
    visual_add(np.squeeze(lesions[i_slice, ..., int(n_z/2)].detach().numpy()), i_slice, i_col, gs, '')
    visual_add(np.squeeze(pred.detach().numpy()), i_slice, i_col + 1, gs, '')
    all_loss = FocalTverskyLoss().forward(pred, lesions[i_slice, ..., int(n_z/2)]).item()

    visual_add(np.squeeze(torch.sigmoid(pred).detach().numpy()), i_slice, i_col + 2, gs, 'FTL: ' + str(round(all_loss, 4)))
    hard_prediction = (pred > 0.5).float()
    dice = dice_score(hard_prediction, np.squeeze(lesions[i_slice, ..., int(n_z/2)])).item()
    print(str(dice))
    visual_add(np.squeeze(hard_prediction.detach().numpy()), i_slice, i_col + 3, gs, 'D: ' + str(round(dice, 4)))
    i_slice += 1


import os

plt.ioff()
plt.switch_backend('agg')
figure_path = os.path.join(data_dir, setting + '_prediction_visualisation.png')
print(figure_path)
figure.savefig(figure_path, dpi='figure', format='png')
plt.close(figure)