import torch
from torch import Tensor, einsum
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union


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

def tversky_loss(true, logits, alpha, beta, eps=1e-7):
    """Computes the Tversky loss [1].
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)

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

class TverskyLoss(nn.Module):
    def __init__(self, weight=None):
        super(TverskyLoss, self).__init__()
        self.alpha = 0.3
        self.beta = 0.7

    def forward(self, logits, targets):

        return tversky_loss(targets, logits, self.alpha, self.beta, eps=1e-7)


class FocalTverskyLoss(nn.Module):
    # Goal: better segmentation loss for small spots
    # from https://arxiv.org/pdf/1810.07842.pdf
    def __init__(self, weight=None):
        super(FocalTverskyLoss, self).__init__()
        # Tversky loss variables
        self.alpha = 0.3
        self.beta = 0.7
        # focal TL vars
        self.gamma = 0.75

    def forward(self, logits, targets):
        tl = tversky_loss(targets, logits, self.alpha, self.beta, eps=1e-7)
        return torch.pow((1-tl), self.gamma)


class SurfaceLoss(nn.Module):
    def __init__(self, weight=None):
        super(SurfaceLoss, self).__init__()

    def forward(self, logits, targets):
        return surface_loss(torch.sigmoid(logits), targets)


def surface_loss(probs: Tensor, dist_maps: Tensor) -> Tensor:
    # from http: // proceedings.mlr.press / v102 / kervadec19a / kervadec19a.pdf
    assert simplex(probs)
    assert not one_hot(dist_maps)

    pc = probs.type(torch.float32)
    dc = dist_maps.type(torch.float32)

    multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

    loss = multipled.mean()

    return loss

# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)
