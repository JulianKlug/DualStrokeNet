import argparse
import sys, os

from data import load_data, generate_loaders
from train_loop import train
from utils import metrics_callback_group, log_settings
from sample_dataset import BasicDataset
from torch.utils.data import DataLoader, random_split


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

    img_scale = 0.07
    val_percent = 0.2

    dataset = BasicDataset(os.path.join(args.dataset_location, 'train'),
                           os.path.join(args.dataset_location, 'train_masks'), img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)


    # Create the directory to save the model, the logs and the params
    if args.save_model is not None and not os.path.isdir(args.save_model):
        os.mkdir(args.save_model)

    log_file = log_settings(args, 'sample', args.save_model)
    callback = metrics_callback_group(log_file, plot_period=1)
    model = UNet(3, 1, args.channels)
    if not args.cpu:
        model.cuda()
    train(model, train_loader, val_loader,
          lr_1=args.lr_1, lr_2=args.lr_2, epochs=args.epochs,
          metrics_callback=callback, split=args.transition, save_path=args.save_model, force_cpu=args.cpu)
