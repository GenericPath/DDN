# from functools import partial
import torch
# import torch.nn.functional as F
from tqdm import tqdm

# from utils.dice_score import multiclass_dice_coeff, dice_coeff


import sys
sys.path.append("../..")
from testLossLinearIntrerp import lech_loss
from nc import partition, de_minW

# copied from nc.py, to avoid making an instance of NormalizedCuts i guess
def objective(x, y):
    """
    f(W,y) = y^T * (D-W) * y / y^T * D * y

    Arguments:
        y: (b, x, y) Torch tensor,
            batch of solution tensors

        x: (b, N, N) Torch tensor,
            batch of affinity/weight tensors (N = x * y)        

    Return value:
        objectives: (b, x) Torch tensor,
            batch of objective function evaluations
    """
    x = de_minW(x) # check if needs to be converted from minVer style
    y = y.flatten(-2) # converts to the vector with shape = (b, 1, N) 
    b, N = y.shape
    y = y.reshape(b,1,N) # convert to a col vector

    # d = torch.einsum('bij->bj', x) # eqv to x.sum(0) --- d vector
    d = x.sum(1, dtype=y.dtype) # 1 because 0 is batch
    D = torch.diag_embed(d) # D = matrix with d on diagonal

    L = D-x # TODO: check does this need to be symmetric too?

    objective_output = torch.div(
        torch.einsum('bij,bkj->bik', torch.einsum('bij,bkj->bik', y, L), y),
        torch.einsum('bij,bkj->bik', torch.einsum('bij,bkj->bik', y, D), y)
    ).squeeze(-2)

    return objective_output

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']/255

        mask_true[mask_true > 0] = 1
        mask_true[mask_true <= 0] = -1
        if mask_true[0][0][0] > 0:
            mask_true *= -1

        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.double)
        mask_true = mask_true.to(device=device, dtype=torch.double)
        # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # v2 eval code
            bipart = partition(mask_pred)
            loss = lech_loss(bipart, mask_true)


            objectives = objective(net.weightsNet(image), mask_pred)
            # v1 eval code
            # partition = (torch.sigmoid(mask_pred) > 0.5).double()
            # dice_score += dice_coeff(partition, mask_true)

            # og eval code v0
            # # convert to one-hot format
            # if net.n_classes == 1:
            #     mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            #     # compute the Dice score
            #     dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            # else:
            #     mask_pred = F.one_hot((F.sigmoid(mask_pred) > 0.5).float(), net.n_classes).permute(0, 3, 1, 2).float()
            #     # compute the Dice score, ignoring background
            #     dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

           

    net.train()

    return loss, objectives

    # v1, v0 return statements
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
