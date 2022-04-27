import argparse
import random
import torch

parser = argparse.ArgumentParser(description='Train the UNet on images and binary target masks')
parser.add_argument('--name', '-n', type=str, default="testing1", help='Tensorboard run name')
parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=1e-4,
                    help='Learning rate', dest='lr')
parser.add_argument('--momentum', '-m', metavar='M', type=float, default=0.9, help='momentum')
# parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
parser.add_argument('--validation', '-v', dest='val', type=float, default=0.1,
                    help='Percent of the data that is used as validation (0-1)')
parser.add_argument('--seed', '-s', metavar='S', type=int, default=0, help='Seed to get consistent outcomes')
parser.add_argument('--total-images', '-ti', metavar='C', type=int, default=300, dest='total_images', help='total number of images in dataset')
parser.add_argument('--net-size', '-ns', metavar='[...]', nargs='+', type=int, default=[1,128,256,512,1024], dest='net_size', help='number of filters for the 3 layers')
parser.add_argument('--gpu-id', '-gpu', type=str, default='1', dest='gpu', help='which id gpu to utilise (if present)')

# currently no options to use
parser.add_argument('--optim', '-o', metavar='O', type=str, default='sgd', dest='optim', help='optimiser used')
parser.add_argument('--shuffle', '-shf', type=bool, default=True, dest='shuffle', help='shuffle batches')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)