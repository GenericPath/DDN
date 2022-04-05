from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset, random_split
import pickle
from PIL import Image
import cv2
# from tqdm import tqdm

import os, random, pickle
# import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

from ddn.pytorch.node import *

import argparse

class NormalizedCuts(EqConstDeclarativeNode):
    """
    A declarative node to embed Normalized Cuts into a Neural Network
    
    Normalized Cuts and Image Segmentation https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf
    Shi, J., & Malik, J. (2000)
    """
    def __init__(self, chunk_size=None):
        super().__init__(chunk_size=chunk_size) # input is divided into chunks of at most chunk_size
        
    def objective(self, x_batch, y_batch):
        """
        f(W,y) = y^T * (D-W) * y / y^T * D * y
        """
        for i in range(len(x_batch)): # code in ddn-3 for the torch.einsum version of this, may switch for efficiency later...
            x = x_batch[i][0]
            y = y_batch[i][0]
            # W is an NxN symmetrical matrix with W(i,j) = w_ij
            D = x.sum(1).diag() # D is an NxN diagonal matrix with d on diagonal, for d(i) = sum_j(w(i,j))
            L = D - x

            y_t = torch.t(y)
            x_batch[i][0] = torch.div(torch.matmul(torch.matmul(y_t, L),y),torch.matmul(torch.matmul(y_t,D),y))
        return x_batch
    
    def equality_constraints(self, x_batch, y_batch):
        """
        subject to y^T * D * 1 = 0
        """
        for i in range(len(x_batch)):
            x = x_batch[i][0]
            y = y_batch[i][0]
            # Ensure correct size and shape of y... scipy minimise flattens y         
            N = x.size(dim=0)
            
            #x is an NxN symmetrical matrix with W(i,j) = w_ij
            D = x.sum(1).diag() # D is an NxN diagonal matrix with d on diagonal, for d(i) = sum_j(w(i,j))
            ONE = torch.ones(N,1)   # Nx1 vector of all ones
            y_t = torch.t(y)
            x_batch[i][0] = torch.matmul(torch.matmul(y_t,D), ONE)
        return x_batch

    def solve(self, W_batch):
        for i in range(len(W_batch)):
            W = W_batch[i][0] # Each batch is passed as [batch, channels, width, height]

            D = torch.diag(torch.sum(W, 0))
            D_half_inv = torch.diag(1.0 / torch.sqrt(torch.sum(W, 0)))
            M = torch.matmul(D_half_inv, torch.matmul((D - W), D_half_inv))

            # M is the normalised laplacian

            (w, v) = torch.linalg.eigh(M)

            #find index of second smallest eigenvalue
            index = torch.argsort(w)[1]

            v_partition = v[:, index]
            # instead of the sign of a digit being the binary split, let the NN learn it
            # v_partition = torch.sign(v_partition)
        
            # return the eigenvector and a blank context
            W_batch[i][0] = v_partition
        return v_partition, None


def data(path, total_images=300):
    """ Generate a simple dataset (if it doesn't already exist) """
    img_size = (32,32) # image size (w,h)

    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' has been made')
        
    if not os.path.isfile(path+'dataset'):
        images = []
        answers = []
        for i in range(total_images):
            # L gives 8-bit pixels (0-255 range of white to black)
            w,h = (random.randint(img_size[0]//3, img_size[0]), random.randint(img_size[0]//3, img_size[0]))
            x,y = (random.randint(img_size[0]//3, img_size[0]), random.randint(img_size[0]//3, img_size[0]))

            xy = [(x-w//2,y-h//2), (x+w//2,y+h//2)]
            answer = np.zeros(img_size)
            answer[xy[0][0]:xy[1][0], xy[0][1]:xy[1][1]] = 1
            answers.append(answer)
            
            # L gives 8-bit pixels (0-255 range of white to black)
            out = Image.fromarray(np.uint8(answer * 255), 'L')
            
            name = path+"img"+str(i)+".png"
            out.save(name, "PNG")
            print(name)
            images.append(name)
            
        output = [images, answers]
        with open(path+'dataset', 'wb') as fp:
            pickle.dump(output, fp)
        print("made the dataset file")

class Simple01(Dataset):
    """ Simple white background, black rectangle dataset """
    
    def __init__(self, file, transform=None):
        """
        file (string): Path to the pickle that contains [img paths, output arrays]
        """
        with open (file, 'rb') as fp:
            output = pickle.load(fp)
            self.images = output[0] # images
            self.segmentations = output[1] # segmentation
        self.transform = transform
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = cv2.imread(self.images[index], 0)
        # or switch to PIL.Image.open() and then img.load()?
        y_label = torch.tensor(self.segmentations[index])
        
        if self.transform is not None:
            img = self.transform(img)
            
        return (img, y_label)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.ConvTranspose2d()

    def forward(self, x):
        x = self.conv1(x)


def train(logging=False,
          epochs: int = 20,
          batch_size: int = 32,
          learning_rate: float = 1e-4, # 0.0001
          momentum: float = 0.9, # unsure if this is a good value or not
          val_percent: float = 0.1,
          save_checkpoint: bool = True,
        ):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')


    dataset = 'simple01/'
    results = 'experiments/'+dataset
    path = 'data/'+dataset # location to store dataset
    dir_checkpoint = 'checkpoints/'+dataset

    if logging:
        run = "1"
        writer = SummaryWriter(results, comment=run)
    # add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)

    # can do writer for a series of experiements done in a loop with lr*i etc etc..
    hparams = {
        "epochs":epochs,
        "optim":"sgd", # this is hardcoded for now..
        "lr": learning_rate,
        "momentum":momentum,
        "batch": batch_size,
        "shuffle":True
    }
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

    net = Net()
    # loss = nn.CrossEntropyLoss()
    
    # opt = optim.SGD(net.parameters(), lr=hparams['lr'], momentum=hparams['momentum'])

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
        'hparam/accuracy': 10, 
        'hparam/loss': 10#*i as an example this could all be in a loop....
    }

    best_accuracy = 0
    for epoch in range(epochs):
        # TRAIN

        # TEST AGAINST VALIDATION
        new_accuracy = 0

        # SAVE IF IT IS THE BEST
        if save_checkpoint: # maybe only save if the accuracy is the highest we have seen so far...
            if new_accuracy >= best_accuracy:
                torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
                print(f'Checkpoint {epoch + 1} saved!')

    if logging:
        writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)

# def get_args():
#     parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
#     parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
#     parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
#     parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
#                         help='Learning rate', dest='lr')
#     parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
#     parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
#     parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
#                         help='Percent of the data that is used as validation (0-100)')
#     parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

#     return parser.parse_args()


if __name__ == '__main__':
    # get_args()
    a = torch.randn(32, 1, 100, 100, requires_grad=True)
    node = NormalizedCuts()
    a = a.detach()
    y,_ = node.solve(a)

    train()
