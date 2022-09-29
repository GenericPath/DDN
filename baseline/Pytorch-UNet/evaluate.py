# from functools import partial
import torch
# import torch.nn.functional as F
from tqdm import tqdm

# from utils.dice_score import multiclass_dice_coeff, dice_coeff


import sys
sys.path.append("../..")
from testLossLinearIntrerp import lech_loss
from nc import partition

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

    return loss

    # v1, v0 return statements
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
