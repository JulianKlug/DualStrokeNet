import argparse
import sys, os

from data import load_data, generate_loaders
from train_loop import train
from utils import csv_callback, log_settings

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
parser.add_argument("--save-model", '-s', help='directory to store the model at the end',
                    type=str, default=None)
parser.add_argument('--log', '-l', help='Store logs in automatically named log file',
                     action='store_true', default=False)
parser.add_argument("--two-d", help='Use two dimensional model',
                    action='store_true', default=False)
parser.add_argument("--cpu", help='Use CPU',
                    action='store_true', default=False)
args = parser.parse_args()



if True and __name__ == '__main__':
    if args.two_d:
        from unet_model_2d import UNet
    else:
        from unet_model_3d import UNet

    tensors = load_data(fname=args.dataset_location)
    ct_sets, _ = generate_loaders(tensors, batch_size=args.batch_size,
                                  use_increment_set=False, threeD= not args.two_d)

    # Create the directory to save the model, the logs and the params
    if args.save_model is not None and not os.path.isdir(args.save_model):
        os.mkdir(args.save_model)

    log_file = log_settings(args, 'ct', args.save_model)
    callback = csv_callback(log_file)
    model = UNet(5, 1, args.channels)
    if not args.cpu:
        model.cuda()
    train(model, ct_sets['train'], ct_sets['test'],
          lr_1=args.lr_1, lr_2=args.lr_2, epochs=args.epochs,
          metrics_callback=callback, split=args.transition, save_path=args.save_model, force_cpu=args.cpu)
