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
    ct_inputs, ct_lesions, mri_inputs, mri_lesions, masks = tensors
    n_subj = ct_inputs.shape[0]
    ct_sets, mri_sets = generate_loaders(tensors, batch_size=n_subj,
                                   threeD=True)

    if modality == 'mri':
        data_sets = mri_sets
        n_z = mri_inputs.shape[-1]
    else:
        data_sets = ct_sets
        n_z = ct_inputs.shape[-1]

    if force_cpu:
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    loader = data_sets['train']
    train_input = [inputs for inputs, outputs in tqdm(loader)]
    train_predictions = [model(inputs[:, :, :, :, n_z/2]) for inputs, outputs in tqdm(loader)]
    train_GT = [outputs for inputs, outputs in tqdm(loader)]

    loader = data_sets['test']
    test_input = [inputs for inputs, outputs in tqdm(loader)]
    test_predictions = [model(inputs[:, :, :, :, n_z/2]) for inputs, outputs in tqdm(loader)]
    test_GT = [outputs for inputs, outputs in tqdm(loader)]

    save_dir = os.path.join(os.path.dirname(data_path), 'model_prediction')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    torch.save(train_input, os.path.join(save_dir, 'train_input.pth'))
    torch.save(train_predictions, os.path.join(save_dir, 'train_predictions.pth'))
    torch.save(train_GT, os.path.join(save_dir, 'train_GT.pth'))
    torch.save(test_input, os.path.join(save_dir, 'test_input.pth'))
    torch.save(test_predictions, os.path.join(save_dir, 'test_predictions.pth'))
    torch.save(test_GT, os.path.join(save_dir, 'test_GT.pth'))


if __name__ == '__main__':
    predict(args.model_location, args.dataset_location, args.input_type, args.cpu)
