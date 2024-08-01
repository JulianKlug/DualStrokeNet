import os, sys
import json
from plots.plot_learning_curves import plot_learning_curves


def metrics_callback_group(log_file_descriptor, plot_period=1):
    csv_call = csv_callback(log_file_descriptor)
    plot_call = plot_callback(log_file_descriptor, plot_period)

    def inner(metrics, epoch):
        csv_call(metrics)
        plot_call(epoch)
    return inner


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


def plot_callback(log_file_descriptor, period=1):
    def inner(epoch):
        epoch += 1 # as callback is called at the end of the epoch, thus epoch + 1 epochs have been gone through
        log_path = log_file_descriptor.name
        if log_path == '<stdout>': return
        if not os.path.exists(log_path): return
        if epoch % period == 0:
            plot_learning_curves(log_path,
                                 save_path=os.path.join(os.path.dirname(log_path), 'learning_curves.png'))
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
