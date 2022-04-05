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

class NormalizedCuts(EqConstDeclarativeNode):
    """
    A declarative node to embed Normalized Cuts into a Neural Network
    
    Normalized Cuts and Image Segmentation https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf
    Shi, J., & Malik, J. (2000)
    """
    def __init__(self, chunk_size=None):
        super().__init__(chunk_size=chunk_size) # input is divided into chunks of at most chunk_size
        
    def objective(self, x, y):
        """
        f(W,y) = y^T * (D-W) * y / y^T * D * y
        """
        
        # for i in len(x):
        # W is an NxN symmetrical matrix with W(i,j) = w_ij
        D = x.sum(1).diag() # D is an NxN diagonal matrix with d on diagonal, for d(i) = sum_j(w(i,j))
        L = D - x

        y_t = torch.t(y)

        return torch.div(torch.matmul(torch.matmul(y_t, L),y),torch.matmul(torch.matmul(y_t,D),y))
    
    def equality_constraints(self, x, y):
        """
        subject to y^T * D * 1 = 0
        """
        # Ensure correct size and shape of y... scipy minimise flattens y         
        N = x.size(dim=0)
        
        #x is an NxN symmetrical matrix with W(i,j) = w_ij
        D = x.sum(1).diag() # D is an NxN diagonal matrix with d on diagonal, for d(i) = sum_j(w(i,j))
        ONE = torch.ones(N,1)   # Nx1 vector of all ones
        y_t = torch.t(y)
        
        return torch.matmul(torch.matmul(y_t,D), ONE)

    def solve(self, W):
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
        return v_partition, _


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


def main():
    dataset = 'simple01/'
    results = 'experiments/'+dataset
    path = 'data/'+dataset # location to store dataset

    run = "1"
    writer = SummaryWriter(results, comment=run)
    # add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)

    # can do writer for a series of experiements done in a loop with lr*i etc etc..
    hparams = {
        "optim":"sgd",
        "lr": 0.001,
        "momentum":0.9,
        "batch": 32,
        "shuffle":True
    }
    data(path) # make the dataset
    train_dataset = Simple01(path+'dataset', transform=transforms.ToTensor())

    print(len(train_dataset))
    # Training and Validation dataset
    val_percent = 0.1
    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val


    train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True,
                                                batch_size=hparams['batch'], shuffle=hparams['shuffle'])
    val_loader = torch.utils.data.DataLoader(val_set, pin_memory=True,
                                                batch_size=hparams['batch'], shuffle=hparams['shuffle'])
    

    x,y = next(iter(train_loader))
    print(x.shape)
    print(y.shape)
    print('Dataset : %d EA \nDataLoader : %d SET' % (len(train_dataset),len(train_loader)))

    # net = Net()
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

    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)

if __name__ == '__main__':
    main()
