from sklearn.metrics import roc_auc_score
import torch.nn as nn

def compute_roc(prediction, outputs):
    y_true = outputs.reshape(-1)
    y_scores = prediction.reshape(-1)
    return roc_auc_score(y_true, y_scores)

def get_batch_volume(data):
    data = data.view(data.shape[0], -1)
    data = data.mean(1)
    return data

def old_dice_loss(input, target, train=False):
    if train:
        smooth = 1
    else:
        smooth = 1e-7

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc