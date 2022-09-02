'''Parser for arguments from command line.'''
import os
import argparse
import importlib

import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names = [
    *model_names, 
    'unet3d', 'xia', 'unet', 'Inception3',
    'r3d_18', 'mc3_18', 'r2plus1d_18']


def parse_options():
    '''
    Options available from command line.
    '''
    parser = argparse.ArgumentParser(description='PyTorch Example')
    # General options
    parser.add_argument('--dataf', default='data', type=str,
                        help='path to studies')
    parser.add_argument('--maskf', default='', type=str,
                        help='path to studies masks')
    parser.add_argument('--dataset', default='', type=str,
                        help='path to csv dataset')
    parser.add_argument('--tg_dataset', default='', type=str,
                        help='path to csv dataset used as target')
    parser.add_argument('-m', '--model_name', dest='model_name',
                        choices=model_names, default='vgg11_bn',
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: vgg11_bn)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N',
                        help='mini-batch size (default: 8), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--gpu', default=None, type=str,
                        help='GPU id to use.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--resf', default='results',
                        help='folder to output images')
    parser.add_argument('--timeframe', default=0,
                        help='time frame to consider')
    parser.add_argument('--test', action='store_true', default=False,
                        help='performs only testing')
    parser.add_argument('--data_type', default='2d',
                        help='type of images used (e.g. 2d or 3d)')
    parser.add_argument('--data_format', default='nifti',
                        help='format of images used (e.g. nifti, png or npy)')
    parser.add_argument('--view', default='',
                        help='view of images used (e.g. la, sa)')
    parser.add_argument('--subcat', default='',
                        help='view subcategory of images used (e.g. 4ch, 2ch)')

    # Training options
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam. default=0.999')
    parser.add_argument('--conf', type=str, default='exp_mnms', metavar='N',
                        help='configuration file to use (default: exp_mnms)')
    parser.add_argument('--epochs', default=0, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--experiment', type=str, default='generic',
                        help='experiment name to be run (can be cifar or mnist also')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--iters', default=None, type=int,
                        help='Maximum number of iterations.')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--name', type=str, default='default', help='name for experiment')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='iterations over critic per iteration over generator')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--outf', default='checkpoints',
                        help='folder to output model checkpoints')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--dist-url', default='', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')

    args = parser.parse_args()

    # Create folders if they don't exist
    os.makedirs(args.outf, exist_ok=True)
    os.makedirs(args.resf, exist_ok=True)

    # Assert each destination folder exist
    assert os.path.exists(args.dataf), \
        'Data folder "{}" does not exist'.format(args.dataf)
    assert os.path.exists(args.outf), \
        'Output folder "{}" does not exist'.format(args.outf)
    assert os.path.exists(args.resf), \
        'Results folder "{}" does not exist'.format(args.resf)

    # Get absolute paths
    args.dataf = os.path.abspath(args.dataf)
    if args.maskf != '':
        args.maskf = os.path.abspath(args.maskf)
    args.outf = os.path.abspath(args.outf)
    args.resf = os.path.abspath(args.resf)

    # Train or test
    args.train = not args.test

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Import experiment file
    assert os.path.exists(os.path.join(
        os.path.dirname(__file__), '{}_exp.py'.format(args.experiment))), \
        'Experiment file "{}_exp.py" does not exist! '.format(args.experiment) +\
        'Please, check again the existing experiment files.'

    exp_settings = importlib.import_module(
        'config.{}_exp'.format(args.experiment)).get()

    # Setting for a dry run
    if args.dry_run:
        exp_settings['epochs'] = 1
    if args.epochs:
        exp_settings['epochs'] = args.epochs

    # TODO: Generalize this part. Overwrite
    # silently and do it in general for every model.
    # TODO: Overwrite experiment settings from
    # command line easily. Pass batch_size for
    # example and overwrite the default value.
    # We'll need to save also these settings.
    exp_settings['learning_rate'] = args.lr
    exp_settings['model_name'] = args.model_name
    exp_settings['batch_size'] = args.batch_size
    exp_settings['results_folder'] = args.resf
    exp_settings['experiment_name'] = args.name
    if args.view != '':
        exp_settings['view'] = args.view
        # If view is LA, the default subcat is 4ch
        if args.view != 'la':
            exp_settings['subcat'] = args.subcat
    if 'accumulated_grad_batches' not in exp_settings.keys():
        exp_settings['accumulated_grad_batches'] = 1
    if 'warmup_epochs' not in exp_settings.keys():
        exp_settings['warmup_epochs'] = 0
    if 'scheduler' not in exp_settings.keys():
        exp_settings['scheduler'] = None
    if 'optimizer' not in exp_settings.keys():
        exp_settings['optimizer'] = 'adam'

    return args, exp_settings
