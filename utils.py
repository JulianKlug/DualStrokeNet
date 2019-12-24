import os, sys
import json

def csv_callback(file_descriptor):
    headers_printed = False
    def inner(metrics):
        nonlocal headers_printed
        if not headers_printed:
            file_descriptor.write(','.join(metrics.keys()))
            file_descriptor.write('\n')
            headers_printed = True
        file_descriptor.write(','.join(str(x) for x in metrics.values()))
        file_descriptor.write('\n')
        file_descriptor.flush()
    return inner

def log_settings(args, modality, model_dir=None):
    if args.two_d:
        dimensions = '2D'
    else:
        dimensions = '3D'

    params_str = 'train_' + modality + '_' + dimensions \
                 + '_c' + str(args.channels) + '_b' + str(args.batch_size) + '_lr1-' + str(args.lr_1) + '_lr2-' \
                 + str(args.lr_2) + '_t' + str(args.transition)

    if model_dir is None:
        model_dir = os.getcwd()

    if args.log:
        log_file = os.path.join(model_dir, 'logs', params_str + '.log')
        print('Logging to', log_file)
        if not os.path.isdir(os.path.join(model_dir, 'logs')):
            os.mkdir(os.path.dirname(log_file))
        log_file = open(log_file, 'w')
    else:
        log_file = sys.stdout


    # Save parameters
    params = vars(args)
    params['modality'] = modality
    params_path = os.path.join(model_dir, 'params', params_str + '_params.json')
    if not os.path.isdir(os.path.dirname(params_path)):
        os.mkdir(os.path.dirname(params_path))
    with open(params_path, 'w') as file:
        json.dump(params, file)

    return log_file
