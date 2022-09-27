import torch
import torch.nn as nn

from net_argparser import net_argparser

from nc import NormalizedCuts, de_minW
from data import *

import matplotlib.pyplot as plt

import numpy as np


# def lech_loss(pred, mask):
#     """
#     pred is yhat
#     mask is y

#     _bar is -1,b becoming 0,b
#     idealy everything is close to -1,1 actual or 0,1 _bar form

#     so instead of classification loss, MSE loss for the application
#     """




#     y_bar = (y+1)/2
#     yhat_bar = (yhat+1)/2

#     loss = (1-y_bar)*(1-yhat_bar)**2 + y_bar@(nn.ReLU(-yhat_bar))

#     return loss

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
steps = 200
lerp_weight = 0.05
criterion = nn.BCEWithLogitsLoss()

losses = []
plots = []
labels = []
for j in range(random_count):
    W_rand = torch.randn_like(W_true) 

    W_rand_symm = W_rand.clone()
    W_rand_symm = torch.tril(W_rand_symm) + torch.tril(W_rand_symm,-1).mT  

    this_loss = []
    in_between = W_rand
    in_between_symm = W_rand_symm
    for i in range(0,steps+1):
        in_between = torch.lerp(in_between, W_true, lerp_weight)
        in_between_symm = torch.lerp(in_between_symm, W_true, lerp_weight)
        in_between_symm = torch.div((in_between_symm + in_between_symm.mT), 2)

        pred = node.solve(in_between)[0] # solve returns (solution, ctx)
        pred_symm = node.solve(in_between_symm)[0]

        if i == steps:
            pred = true
            pred_symm = true

        if j == 0 and i % (steps/10) == 0:
        # if i % (steps/10) == 0:
            plots.append([pred, in_between, pred_symm, in_between_symm, W_true])
            labels.append([f'{node.objective(in_between, pred).item():.5f}\nmin{torch.min(pred):.2f} max{torch.max(pred):.2f}', 'noise', f'{node.objective(in_between_symm, pred_symm).item():.5f}\nmin{torch.min(pred_symm):.2f} max{torch.max(pred_symm):.2f}','symm', f'{i}'])

        if i == steps:
            continue

        # loss = lech_loss(true, pred)

        loss = criterion(pred, true)
        this_loss.append(loss.item())
    losses.append(this_loss)


plot_multiple_images('test-view', plots, figsize=args.img_size, ipynb=False, cmap_name='gray', labels=labels)

x = np.linspace(0, steps, steps)
for j in range(random_count):
    plt.plot(x,losses[j], label= f'run {j}')

second_fig_name = 'experiments/test-loss.png'
# second_fig_name = f'experiments/test-bipart{args.bipart}-symm_L{args.symm_norm_L}-r{args.radius}-minify{args.minify}.png'
plt.savefig(second_fig_name)
print(f'saved {second_fig_name}')