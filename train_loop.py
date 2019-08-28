import torch
from collections import defaultdict, OrderedDict, Callable
from torch.optim import SGD, Adam
from tqdm import tqdm
import numpy as np

from metrics import get_batch_volume, dice_loss


# From: https://stackoverflow.com/questions/6190331/how-to-implement-an-ordered-default-dict
class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               OrderedDict.__repr__(self))


def forward(model, loader, criterion, optimizer=None, force_cpu=False):

    if optimizer is None:
        model.eval()
    else:
        model.train()

    metrics = DefaultOrderedDict(list)

    all_predictions = []
    all_labels = []

    for inputs, outputs in tqdm(loader):
        if not force_cpu:
            inputs = inputs.cuda(non_blocking=True)
            outputs = outputs.cuda(non_blocking=True)

        prediction = model(inputs)
        all_loss = dice_loss(torch.sigmoid(prediction), outputs, train=True)
        all_predictions.append(prediction.data.cpu().numpy().reshape(-1))
        all_labels.append(outputs.data.cpu().numpy().reshape(-1))
        loss = all_loss.mean()
        metrics['loss'].append(loss.item())

        hard_prediction = (prediction > 0).float()
        predicted_volumes = get_batch_volume(hard_prediction)
        true_volumes = get_batch_volume(outputs)

        metrics['volume_error'].append((torch.abs(predicted_volumes - true_volumes) / (true_volumes + 1e-5)).mean().item())

        dice = 1 - dice_loss(hard_prediction, outputs).item()
        metrics['dice'].append(dice)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return OrderedDict({k: np.mean(v) for (k, v) in metrics.items()})


def train(model, train_loader, val_loader, lr_1, lr_2, metrics_callback=None, epochs=10, split=150, save_path=None,
          force_cpu=False):
    criterion = torch.nn.BCEWithLogitsLoss()
    first_phase_optimizer = SGD(model.parameters(), lr=lr_1, momentum=0)
    second_phase_optimizer = SGD(model.parameters(), lr=lr_2, momentum=0.99)
    best_loss = np.inf
    for e in range(epochs):
        if e < split:
            optimizer = first_phase_optimizer
        else:
            optimizer = second_phase_optimizer
        tm = forward(model, train_loader, criterion, optimizer, force_cpu=force_cpu)
        vm = forward(model, val_loader, criterion, force_cpu=force_cpu)
        metrics = OrderedDict()
        metrics['epoch'] = e
        for k, v in tm.items():
            metrics['train_' + k] = v
        for k, v in vm.items():
            metrics['test_' + k] = v
        if metrics_callback is not None:
            metrics_callback(metrics)
        if metrics['train_loss'] < best_loss and save_path is not None:
            best_loss = metrics['train_loss']
            torch.save(model, save_path)
