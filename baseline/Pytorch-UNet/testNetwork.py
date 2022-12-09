import wandb
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from utils.data_loading import BasicDataset
from utils.dice_score import *
from evaluate import evaluate
from unet import UNet

import sys
sys.path.append("../..")
from model import Net as DDNNet
from model import WeightsNet
from data import plot_multiple_images

hyperparameter_defaults = dict(
        epochs=10, 
        batch_size = 1,

        lr = 1e-4, # will lower during training
        weight_decay=1e-8,
        momentum=0.9,
        patience=2,

        dir_img='1sample', # defaults to data/ + data_path
        dir_mask='1sample', # same as above

        # dir_img = 'simple01/16-16/images',
        # dir_mask = 'simple01/16-16/images',

        val=10.0, # Percent of the data that is used as validation (0-100)

        n_classes=2, # Number of classes
        n_channels=3, # 3 for RGB inputs

        seed=0,
        gpu=1,
        load=False, # Load model from a .pth file
        test=False,
        img_scale=1, # Downscaling factor of the images

        amp=False, # Use mixed precision
        bilinear = True, # Use bilinear upsampling

        net='DDN',
        minify=False,
        radius=10,
        eqconst=False,
        eps=1e-7,
        gamma=0,
        net_size_weights=[1,4,8,4],
        net_size_post=[1,4,8,4],

        # NOT USED YET
        optim='sgd',
        shuffle=True,

        # whether to have a network at the end
        post_net =True,
        # whether to default partion the outputs
        bipart = False, # would make the output not a smooth function and thus gradients would be useless
)

# MODE DISABLED TO TEST WITHOUT ACTUALLY SYNCING TO CLOUD
experiment = wandb.init(project='DDN-NC', config=hyperparameter_defaults, mode='disabled')
# Config parameters are automatically set by W&B sweep agent
args = wandb.config

args.img_size = (16,16)

if args.net == 'DDN':
        args.network = 0
        net = DDNNet(args).double()
elif args.net == 'DDN-weights':
        args.network = 1
        net = WeightsNet(args)
elif args.net == 'UNet':
        net = UNet(n_channels=args.n_channels, n_classes=args.n_classes, bilinear=args.bilinear)
        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
else:
        raise Exception(f"incorrect network specified - {args.net}")

dir_checkpoint = Path('../../data/tc/')
base_path = '../../data/'

# 1. Create dataset
dataset = BasicDataset(base_path+args.dir_img, base_path+args.dir_mask, scale=args.img_scale, transform = transforms.ToTensor())
# TODO: copy from https://github.com/borisdayma/lightning-kitti/blob/master/train.py

# 3. Create data loaders
loader_args = dict(batch_size=args.batch_size, pin_memory=True)
train_loader = DataLoader(dataset, shuffle=args.shuffle, drop_last=True,**loader_args)

# 4. Misc
criterion = nn.BCEWithLogitsLoss()

for batch in train_loader:
        images = batch['image']
        true_masks = batch['mask']/255

        masks_pred = net(images)

        plots = [images, true_masks, masks_pred]

        plot_multiple_images('test-network', plots, figsize=args.img_size, ipynb=False, cmap_name='jet')
        
        # 'true': wandb.Image(true_masks[0].float().cpu()),
        # 'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),

        loss = criterion(masks_pred.float(), true_masks.float())
        val_score = evaluate(net, train_loader, 'cpu')

        print(f'loss {loss} val {val_score}')