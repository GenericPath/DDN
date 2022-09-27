import torch
import torch.nn as nn

from net_argparser import net_argparser

from nc import NormalizedCuts, de_minW
from data import *

import matplotlib.pyplot as plt

import numpy as np


def lech_loss(pred, mask):
    """
    pred is yhat
    mask is y

    _bar is -1,b becoming 0,b
    idealy everything is close to -1,1 actual or 0,1 _bar form

    so instead of classification loss, MSE loss for the application
    """

    # mask should be in format {-1, 1} ideally
    # mask_bar is {0,1} for regression style problem

    mask = mask.flatten(-2)[:,None,:]
    pred = pred.flatten(-2)[:,:,None]

    mask_bar = torch.div((mask+1),2) # y_n
    # pred_bar = torch.div((pred+1), 2) # yhat_n convert predicition to {0,1} roughly too
    # goes to 0 if not using above :)
    pred_bar = pred

    relu = nn.ReLU()
    # TODO:
    # 1. verify if pred_bar is used for pred
    # 2. see if i need to flatten (shouldnt as it should broadcast correctly?)
    # 3. and otherwise checking everything is all good!

    loss = torch.bmm((1-mask_bar),(pred_bar+1)**2) + torch.bmm(mask_bar, relu(-pred_bar))
    return torch.mean(loss) # avg across batch I guess

args = net_argparser(ipynb=True)
args.network = 1
args.total_images = 1
args.minify = False # TODO: test for this working properly
args.bipart = False # Obviously will make it a non-continuous function
args.symm_norm_L = False # TODO: test for this maybe? probably just leave off...
args.radius = 100
args.img_size = [16,16] # the default is 32,32 anyway

train_dataset = SimpleDatasets(args, transform=transforms.ToTensor())

true = train_dataset.get_segmentation(0)
true[true > 0] = 1
true[true <= 0] = -1
# convert true to the expectation {-1,1} instead of 0,1?


W_true = train_dataset.get_weights(0).double()

node = NormalizedCuts(eps=1e-3, bipart=args.bipart, symm_norm_L=args.symm_norm_L)

random_count = 3
steps = 100
lerp_weight = 0.05
criterion = lech_loss #nn.BCEWithLogitsLoss()

losses_2 = []
for j in range(random_count):
    this_losses2 = []
    true_rand = torch.randn_like(true)
    inbetween_true = true_rand
    for i in range(0, steps):
        inbetween_true = torch.lerp(inbetween_true, true, lerp_weight)
        loss = criterion(inbetween_true, true)
        this_losses2.append(loss.item())
    losses_2.append(this_losses2)

x = np.linspace(0, steps, steps)
for j in range(random_count):
    plt.plot(x,losses_2[j])
name = 'experiments/test-lerpoutput.png'
plt.savefig(name)
print(f'saved {name}')
plt.close()

losses = []
plots = []
labels = []
for j in range(random_count):
    W_rand = torch.randn_like(W_true, dtype=torch.double) 

    W_rand_symm = W_rand.clone()
    W_rand_symm = torch.tril(W_rand_symm) + torch.tril(W_rand_symm,-1).mT  

    this_loss = []
    in_between = W_rand
    in_between_symm = W_rand_symm
    for i in range(0,steps+1):
        in_between = torch.lerp(in_between, W_true, lerp_weight)
        in_between_symm = torch.lerp(in_between_symm, W_true, lerp_weight)
        in_between_symm = torch.div((in_between_symm + in_between_symm.mT), 2)

        if i == steps:
            in_between = W_true
            in_between_symm = W_true

        pred = node.solve(in_between)[0] # solve returns (solution, ctx)
        pred_symm = node.solve(in_between_symm)[0]

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


basic = False
if basic:
    second_fig_name = 'experiments/test-loss.png'
else:
    second_fig_name = 'experiments/test-lech.png'
    plt.title(f'bipart{args.bipart}-symm_L{args.symm_norm_L}-r{args.radius}-minify{args.minify}')
plt.savefig(second_fig_name)
plt.close()
print(f'saved {second_fig_name}')