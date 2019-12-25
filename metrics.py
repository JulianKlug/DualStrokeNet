import torch
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

def dice_score(input, target, train=False):
    if train:
        smooth = 1
    else:
        smooth = 1e-7

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2. * intersection + smooth) /
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


class BinaryCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None):

        super(BinaryCrossEntropyLoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, reduction = 'elementwise_mean')

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)  # Flatten
        targets_flat = targets.view(-1)  # Flatten
        return self.bce_loss(probs_flat, targets_flat)


class CombinedDiceEntropyLoss(nn.Module):
    def __init__(self, weight=None):

        super(CombinedDiceEntropyLoss, self).__init__()
        self.BCE_loss = torch.nn.BCEWithLogitsLoss()
        self.Dice_loss = DiceLoss()

    def forward(self, logits, targets):

        return self.BCE_loss.forward(logits, targets) + \
                    self.Dice_loss.forward(logits, targets)
