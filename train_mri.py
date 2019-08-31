import argparse
import sys, os

from data import load_data, generate_loaders
from train_loop import train
from utils import csv_callback

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
parser.add_argument('--log', '-l', help='where to store logs',
                     type=str, default=None)
parser.add_argument("--two-d", help='Use two dimensional model',
                    action='store_true', default=False)
parser.add_argument("--cpu", help='Use CPU',
                    action='store_true', default=False)
args = parser.parse_args()



if __name__ == '__main__':
    if args.two_d:
        from unet_model_2d import UNet
        dimensions = '2D'
    else:
        dimensions = '3D'
        from unet_model_3d import UNet

    if args.log is None:
        params = 'train_mri_' + dimensions \
                 + '_c' + str(args.channels) + '_b' + str(args.batch_size) + '_lr1-' + str(args.lr_1) + '_lr2-' \
                 + str(args.lr_2) + '_t' + str(args.transition)
        args.log = os.path.join(os.getcwd(), 'logs', params + '.log')
        print(args.log)

    tensors = load_data(fname=args.dataset_location)
    _, mri_sets = generate_loaders(tensors, batch_size=args.batch_size,
                                   threeD= not args.two_d)
    callback = csv_callback(open(args.log, 'w'))
    model = UNet(4, 1, args.channels)
    if not args.cpu:
        model.cuda()
    train(model, mri_sets['train'], mri_sets['test'],
          lr_1=args.lr_1, lr_2=args.lr_2, epochs=args.epochs,
          metrics_callback=callback, split=0, save_path=args.save_model, force_cpu=args.cpu, log=log)
