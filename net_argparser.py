import argparse

def net_argparser():
    """
    Uses argparse to parse all commandline arguments, used in main.py but also to test other parts separately.
    Usage: args = net_argparser()

    See net_argparser.py for options.
    """
    parser = argparse.ArgumentParser(description='Train the UNet on images and binary target masks')
    parser.add_argument('--name', '-n', type=str, default='test', help='Tensorboard run name')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--momentum', '-m', metavar='M', type=float, default=0.9, help='momentum')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.1,
                        help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('--seed', '-s', metavar='S', type=int, default=None, help='Seed to get consistent outcomes')
    parser.add_argument('--total-images', '-ti', metavar='N', type=int, default=300, dest='total_images', help='total number of images in dataset')
    parser.add_argument('--net-size', '-ns', metavar='[...]', nargs='+', type=int, default=[1,128,256,512,1024], dest='net_size', help='number of filters for the 3 layers')
    parser.add_argument('--gpu-id', '-gpu', type=str, default='1', dest='gpu', help='which id gpu to utilise (if present)')

    # currently no options to use
    parser.add_argument('--optim', '-o', metavar='OPT', type=str, default='sgd', dest='optim', help='optimiser to use')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle batches')
    parser.add_argument('--dataset', type=str, default='simple01', help='dataset to use: simple01')

    # newer
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--test', action='store_true', help='Whether to test/evaluate or train')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--production', default=False, type=bool, help='Production mode: If true run in a separate folder on a copy of the python scripts')
    parser.add_argument('--network', default=0, type=int, help='network to use: 0=Net (Weights->NC->Post), 1=WeightsNet (Weights)')

    # newest
    parser.add_argument('--minify', default=False, type=bool, help='minify the weights mode (for the PreNC portion)')
    parser.add_argument('--radius', '-r', default=1, type=int, help='radius value for expected weights (only relevant for minified version)')

    # TODO : add option to switch between eqconst
    parser.add_argument('--eqconst', default=True, type=bool, help='equality constrained or non equality constrained')

    return parser.parse_args()