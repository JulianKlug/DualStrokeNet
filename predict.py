import torch, os, argparse
from tqdm import tqdm
from data import load_data, generate_loaders

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


def predict(model_path, data_path, modality, force_cpu=False):
    tensors = load_data(fname=data_path)
    ct_sets, mri_sets = generate_loaders(tensors, batch_size=1,
                                   threeD=False)
    if modality == 'mri':
        data_sets = mri_sets
    else:
        data_sets = ct_sets

    if force_cpu:
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path, map_location=torch.device('cpu'))

    loader = data_sets['train']
    train_predictions = [model(inputs) for inputs, outputs in tqdm(loader)]
    train_GT = [outputs for inputs, outputs in tqdm(loader)]

    loader = data_sets['test']
    test_predictions = [model(inputs) for inputs, outputs in tqdm(loader)]
    test_GT = [outputs for inputs, outputs in tqdm(loader)]

    save_dir = os.path.join(os.path.dirname(data_path), 'model_prediction')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(train_predictions, os.path.join(save_dir, 'train_predictions.pth'))
    torch.save(train_GT, os.path.join(save_dir, 'train_GT.pth'))
    torch.save(test_predictions, os.path.join(save_dir, 'test_predictions.pth'))
    torch.save(test_GT, os.path.join(save_dir, 'test_GT.pth'))


if __name__ == '__main__':
    predict(args.model_location, args.dataset_location, args.input_type, args.cpu)
    