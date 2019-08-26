from sklearn.metrics import roc_auc_score

def compute_roc(prediction, outputs):
    y_true = outputs.reshape(-1)
    y_scores = prediction.reshape(-1)
    return roc_auc_score(y_true, y_scores)

def get_batch_volume(data):
    data = data.view(data.shape[0], -1)
    data = data.mean(1)
    return data

def dice_loss(input, target, train=False):
    if train:
        smooth = 1
    else:
        smooth = 1e-7

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))
