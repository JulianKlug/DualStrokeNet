import os
import torch
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

def log_settings(args, modality):
    if args.two_d:
        dimensions = '2D'
    else:
        dimensions = '3D'

    if args.log is None:
        params_str = 'train_mri_' + dimensions \
                 + '_c' + str(args.channels) + '_b' + str(args.batch_size) + '_lr1-' + str(args.lr_1) + '_lr2-' \
                 + str(args.lr_2) + '_t' + str(args.transition)
        args.log = os.path.join(os.getcwd(), 'logs', params_str + '.log')
    print('Logging to', args.log)

    if not os.path.isdir(os.path.dirname(args.log)):
        os.mkdir(os.path.dirname(args.log))

    # Save parameters
    params = vars(args)
    params['modality'] = modality
    params_path = os.path.join(os.path.dirname(args.log), 'params', os.path.basename(args.log).split('.')[0] + '_params.json')
    if not os.path.isdir(os.path.dirname(params_path)):
        print(os.path.dirname(params_path))
        os.mkdir(os.path.dirname(params_path))
    with open(params_path, 'w') as file:
        json.dump(params, file)

    return args.log
