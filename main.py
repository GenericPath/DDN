# Deep Declarative Node for Normalised Cuts
# Garth Wales - 2022
import torch.utils.tensorboard as tb
import torch

import argparse, random, os, shutil

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from data import get_dataset
from model_loops import test, train, validate
from model import Net

# Maybe add this later
# from torchsummary import summary


parser = argparse.ArgumentParser(description='Train the UNet on images and binary target masks')
parser.add_argument('--name', '-n', type=str, default='', help='Tensorboard run name')
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
parser.add_argument('--dataset', type=str, default='simple01', help='dataset to use')

# newer
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--test', action='store_true', help='Whether to test/evaluate or train')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

def main():
    # Parse commandline arguments
    args = parser.parse_args()

    # Pre-model setup
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    cudnn.benchmark = True # Cuda optimisations when using a fixed input size
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    if args.name:
        results = results + args.name + '/'
        if not os.path.exists(results):
            os.makedirs(results)
            print(results + ' has been made')
        args.writer = tb.SummaryWriter(results)

    # Create the model, loss, optimizer and scheduler
    model = Net(args)
    model = model.to(device=device)
    criterion = nn.BCEWithLogitsLoss()
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr) # TODO : add weight decay and betas as options
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Load the weights of a saved network (if provided)
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            best_error = checkpoint['best_error']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
            return()

    # Load dataset, setup dict to pass to other funcs
    train_loader, val_loader = get_dataset(args)

    # Evaluate the network (and don't train)
    if args.test:
        avg_acc, avg_loss = test(val_loader, model, criterion)
        print(f'Evaluation: avg acc - {avg_acc}, avg_loss - {avg_loss}')
        return

    # Train the network (and test against the validation data)
    best_acc = 0
    best_error = float('inf')
    for epoch in range(args.start_epoch, args.epochs):

        t_acc, t_loss = train(train_loader, model, device, criterion, optimizer)
        v_acc, v_loss = validate(val_loader, model, device, criterion, scheduler)

        # Currently best is based on acc, could be changed for loss
        is_best = v_acc > best_acc
        best_error = min(v_loss, best_error)
        best_acc = min(v_acc, best_acc)

        if args.writer:
            args.writer.add_scalar("Loss/val", v_loss, epoch)
            args.writer.add_scalar("Acc/val", v_acc, epoch)
            args.writer.add_scalar("Acc/train", t_acc, epoch)
            args.writer.add_scalar("Loss/train", t_loss, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_error': best_error,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, dir=args.log_dir, filename='checkpoint_epoch_' + str(epoch + 1))

    if args.writer:
        args.writer.close()

def save_checkpoint(state, is_best, dir='', filename='checkpoint'):
    torch.save(state, dir + filename + '.pth.tar')
    if is_best:
        shutil.copyfile(dir + filename + '.pth.tar', dir + 'model_best.pth.tar')

if __name__ == '__main__':
    main()