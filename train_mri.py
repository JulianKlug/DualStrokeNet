import argparse
import sys

from data import load_data, generate_loaders
from train_loop import train
from unet_model import UNet
from utils import csv_callback
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", "-e", help='Number of epochs to train on',
                    type=int, default=200)
parser.add_argument("--transition", "-t", help='When to go in second phase',
                    type=int, default=100)
parser.add_argument("--channels", "-c", help='Number of initial hidden channels',
                    type=int, default=64)
parser.add_argument("--batch-size", "-b", help='batch size',
                    type=int, default=64)
parser.add_argument("--dataset-location", "-d", help='Where is the dataset',
                    type=str, required=True)
parser.add_argument("--lr-1", help='learning rate to use in the first phase',
                    type=float, default=0.1)
parser.add_argument("--lr-2", help='learning rate to use in the second phase',
                    type=float, default=0.001)
parser.add_argument("--save-model", help='where to store the model at the end',
                    type=str, default=None)
args = parser.parse_args()



if __name__ == '__main__':
    tensors = load_data(fname=args.dataset_location)
    _, mri_sets = generate_loaders(tensors, batch_size=args.batch_size)
    callback = csv_callback(sys.stdout)
    model = UNet(4, 1, args.channels)
    model.cuda()
    train(model, mri_sets['train'], mri_sets['test'],
          lr_1=args.lr_1, lr_2=args.lr_2, epochs=args.epochs,
          metrics_callback=callback, split=0, save_path=args.save_model)
