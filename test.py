# Deep Declarative Node for Normalised Cuts
# Garth Wales - 2022

from numpy import outer
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from tqdm import tqdm
# import pickle
import time, os, random

# locally defined imports
from ddn.pytorch.node import DeclarativeLayer
from nc import NormalizedCuts
from unetmini import PreNC, PostNC #UNet
from data import data, Simple01

# to be added... there will be better code soon tm
import argparse

try:
    # Use in debug to print sizes for a given input size
    from torchsummary import summary
except:
    pass

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.preNC = PreNC()
        self.nc = NormalizedCuts(eps=1) # eps sets the absolute difference between objective solutions and 0
        self.decl = DeclarativeLayer(self.nc) # converts the NC into a pytorch layer (forward/backward instead of solve/gradient)
        self.postNC = PostNC()

    def forward(self, x):
        x = self.preNC(x) # make the affinity matrix (or something else that works with)
        x = self.decl(x) # check the size of this output...
        x = self.postNC(x)
        return x

def train(args):
    hparams = {
        "epochs":args.epochs,
        "optim":"sgd", # this is hardcoded for now..
        "lr": args.lr,
        "momentum":args.momentum,
        "batch": args.batch_size,
        "shuffle":True,
    }

    val_percent = args.val
    save_checkpoint = True
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    dataset = 'simple01/'
    results = 'experiments/'+dataset
    path = 'data/'+dataset # location to store dataset
    dir_checkpoint = 'checkpoints/'+dataset

    run = "2"
    writer = SummaryWriter(results, comment=run)
    # add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)

    # can do writer for a series of experiements done in a loop with lr*i etc etc..
    # hparams = { ... }

    data(path) # make the dataset
    train_dataset = Simple01(path+'dataset', transform=transforms.ToTensor())

    print(f'Total dataset size {len(train_dataset)}')
    # Training and Validation dataset
    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val

    train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True,
                                                batch_size=hparams['batch'], shuffle=hparams['shuffle'])
    val_loader = torch.utils.data.DataLoader(val_set, pin_memory=True,
                                                batch_size=hparams['batch'], shuffle=hparams['shuffle'])
    
    # Test dataset shapes
    # x,y = next(iter(train_loader))
    # print(x.shape)
    # print(y.shape)
    # print('Dataset : %d EA \nDataLoader : %d SET' % (len(train_dataset),len(train_loader)))

    torch.backends.cudnn.benchmark = True
    net = Net()
    net = net.to(device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=hparams['lr'], momentum=hparams['momentum'])

    # visualise predictions throughout training
    # from https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    # writer.add_figure('predictions vs. actuals',
    #                 plot_classes_preds(net, inputs, labels),
    #                 global_step=epoch * len(trainloader) + i)
    # running_loss = 0.0

    # add graph to writer (add_graph)

    # Possible metrics (from https://www.jeremyjordan.me/evaluating-image-segmentation-models/)
    # Pixel accuracy (percent of correct pixels)
    # IoU
    # precision recall curves (tensorboard use add_pr_curve)
    metrics = { # the key must be unique from anything added in add_scalar, so hparam/accuracy is used
        'hparam/accuracy': 0, 
        'hparam/loss': 0#*i as an example this could all be in a loop....
    }

    best_accuracy = 0
    for epoch in range(args.epochs):
        # TRAIN
        net.train()
        # start_time = time.time()
        for index, (input_batch, target_batch) in enumerate(tqdm(train_loader)):
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            output = net(input_batch)
            loss = criterion(output, target_batch)
            # Compute gradient and do optimizer step
            optimizer.zero_grad()
            if not torch.isnan(loss).any():
                loss.backward()
            optimizer.step()
            
            # print(f'Batch #{index},\ttime\t{time.time()-start_time}')
            # calculate accuracy, output metrics
            train_accuracy = output.eq(target_batch).float().mean()
            writer.add_scalar("Train accuracy", train_accuracy, epoch)
            writer.add_scalar("Loss/train", loss.item(), epoch)
            # start_time = time.time()

        # TEST AGAINST VALIDATION
        net.eval()
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                output = net(input_batch)
                val_accuracy = output.eq(target_batch).float().mean()

                writer.add_scalar("Validation accuracy", val_accuracy, epoch)

        # SAVE IF IT IS THE BEST
        if save_checkpoint: # maybe only save if the accuracy is the highest we have seen so far...
            if val_accuracy >= best_accuracy and val_accuracy != 0:
                if not os.path.exists(dir_checkpoint):
                    os.makedirs(dir_checkpoint)
                    print(dir_checkpoint + ' has been made')
                torch.save(net.state_dict(), str(dir_checkpoint + f'checkpoint_epoch{epoch+1}.pth'))
                print(f'Checkpoint {epoch + 1} saved!')

    metrics['hparam/accuracy'] = val_accuracy
    metrics['hparam/loss'] = loss

    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--momentum', '-m', metavar='M', type=float, default=0.9, help='momentum')
    # parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.1,
                        help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('--seed', '-s', metavar='S', type=int, default=0, help='Seed to get consistent outcomes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.use_deterministic_algorithms(True)

    train(args)
