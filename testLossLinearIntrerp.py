import torch
import torch.nn as nn

from net_argparser import net_argparser

from nc import NormalizedCuts, de_minW
from data import *

import matplotlib.pyplot as plt

import numpy as np

args = net_argparser(ipynb=True)
args.network = 1
args.total_images = 1
args.minify = False
args.bipart = False
args.symm_norm_L = False
args.radius = 100
args.img_size = [16,16] # the default is 32,32 anyway

train_dataset = SimpleDatasets(args, transform=transforms.ToTensor())

true = train_dataset.get_segmentation(0)
W_true = train_dataset.get_weights(0)

node = NormalizedCuts(eps=1e-3, bipart=args.bipart, symm_norm_L=args.symm_norm_L)

random_count = 3
steps = 100
criterion = nn.BCEWithLogitsLoss()



losses = []
for j in range(random_count):
    W_rand = torch.randn_like(W_true)    

    this_loss = []
    in_between = W_rand
    for i in range(0,steps):
        in_between = torch.lerp(in_between, W_true, 1/steps)

        pred = node.solve(in_between)
        loss = criterion(pred[0], true)
        this_loss.append(loss.item())
    losses.append(this_loss)
# print(losses)

x = np.linspace(0, steps, steps)
for j in range(random_count):
    plt.plot(x,losses[j], label= f'run {j}')

plt.savefig(f'experiments/test-bipart{args.bipart}-symm_L{args.symm_norm_L}-r{args.radius}-minify{args.minify}.png')