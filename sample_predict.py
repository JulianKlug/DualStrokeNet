import torch, os, argparse
from tqdm import tqdm
from sample_dataset import BasicDataset
from torch.utils.data import DataLoader, random_split


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-location", "-d", help='Where is the dataset',
                    type=str, required=True)
parser.add_argument("--model-location", "-m", help='Where is the model',
                    type=str, required=True)
parser.add_argument("--input-type", "-i", help='mri or ct?',
                    type=str, required=True)
parser.add_argument("--cpu", help='Use CPU',
                    action='store_true', default=False)
args = parser.parse_args()


def sample_predict(model_path, data_path,mod, force_cpu=False):
    img_scale = 0.07
    val_percent = 0.2

    dataset = BasicDataset(os.path.join(data_path, 'train'),
                           os.path.join(data_path, 'train_masks'), img_scale)
#    dataset[0][0].shape()
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=len(dataset), shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=len(dataset), shuffle=False, num_workers=8, pin_memory=True)

    if force_cpu:
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    train_input, train_GT = map(list, zip(*[subj_data for subj_data in tqdm(train_loader)]))
    train_predictions = [model(x) for x in train_input]

    test_input, test_GT = map(list, zip(*[subj_data for subj_data in tqdm(val_loader)]))
    test_predictions = [model(x) for x in test_input]


    save_dir = os.path.join(os.path.dirname(model_path), 'model_prediction')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    torch.save(train_input, os.path.join(save_dir, 'train_input.pth'))
    torch.save(train_predictions, os.path.join(save_dir, 'train_predictions.pth'))
    torch.save(train_GT, os.path.join(save_dir, 'train_GT.pth'))
    torch.save(test_input, os.path.join(save_dir, 'test_input.pth'))
    torch.save(test_predictions, os.path.join(save_dir, 'test_predictions.pth'))
    torch.save(test_GT, os.path.join(save_dir, 'test_GT.pth'))


if __name__ == '__main__':
    sample_predict(args.model_location, args.dataset_location, args.input_type, args.cpu)
