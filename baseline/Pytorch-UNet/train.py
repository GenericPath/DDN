import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

import sys
sys.path.append("../..")
from model import Net as DDNNet
from nc import NormalizedCuts, de_minW
from model import WeightsNet

dir_checkpoint = Path('../../data/tc/')
base_path = '../../data/'

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
    pred_bar = pred
    relu = nn.ReLU()
    # TODO:
    # 1. verify if pred_bar is used for pred
    # 2. see if i need to flatten (shouldnt as it should broadcast correctly?)
    # 3. and otherwise checking everything is all good!

    loss = torch.bmm((1-mask_bar),(pred_bar+1)**2) + torch.bmm(mask_bar, relu(-pred_bar))
    return torch.mean(loss) # avg across batch I guess


def train_net(net, args, experiment, save_checkpoint = True):

    # 1. Create dataset
    dataset = BasicDataset(base_path+args.dir_img, base_path+args.dir_mask, scale=args.img_scale, transform = transforms.ToTensor())
    # TODO: copy from https://github.com/borisdayma/lightning-kitti/blob/master/train.py

    # 2. Split into train / validation partitions
    val_percent = args.val / 100
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    # 3. Create data loaders
    loader_args = dict(batch_size=args.batch_size, pin_memory=True) #, num_workers=4. currently without so not multithreaded for debug
    train_loader = DataLoader(train_set, shuffle=args.shuffle, drop_last=True,**loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    test_node = NormalizedCuts()

    logging.info(f'''Starting training:
        Net:             {args.net}
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {args.img_scale}
        Mixed Precision: {args.amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if args.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise Exception(f'Optimizer not supported - {args.optim}')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = lech_loss
    global_step = 0

    # 5. Begin training
    for epoch in range(1, args.epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']/255
                true_masks[true_masks > 0] = 1
                true_masks[true_masks <= 0] = -1
                if true_masks[0][0][0] > 0:
                    true_masks *= -1

                images = images.to(device=device, dtype=torch.double)
                true_masks = true_masks.to(device=device, dtype=torch.double)

                with torch.cuda.amp.autocast(enabled=args.amp):
                    masks_pred = net(images)

                    loss = criterion(masks_pred.double(), true_masks.double())
                    # dice = dice_loss(masks_pred, true_masks)
                    # bce = criterion(masks_pred, true_masks)
                    # loss = bce * args.bce_weight + dice * (1 - args.bce_weight)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * args.batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # histograms = {}
                        # for tag, value in net.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, objectives = evaluate(net, val_loader, device)
                        scheduler.step(val_score)


                        obj_mean = torch.mean(objectives)
                        obj_std = torch.std(objectives)

                        # partition = (torch.sigmoid(mask_pred) > 0.5).double()
                        # log this instead?

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation loss': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'weights' : wandb.Image(de_minW(net.weightsNet(images[0][None,:])).float().cpu()),
                            'weights2' : wandb.Image(net.weightsNet(images[0][None,:]).float().cpu()),
                            # 'objective' : test_node.objective(images[0][None,:].cpu(), masks_pred[0][None,:].float()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(masks_pred[0].float().cpu()),
                            },
                            'obj func mean' : obj_mean,
                            'obj func std' : obj_std,
                            'step': global_step,
                            'epoch': epoch,
                            # **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / f'checkpoint-{experiment.name}_epoch-{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

if __name__ == '__main__':
    # Default parameters
    hyperparameter_defaults = dict(
        epochs=20, 
        batch_size = 50,

        lr = 1e-3, # will lower during training
        weight_decay=0,
        momentum=0.9,
        patience=5,

        # dir_img='1sample', # defaults to data/ + data_path
        # dir_mask='1sample', # same as above

        dir_img = 'simple01/16-16/images',
        dir_mask = 'simple01/16-16/images',

        val=10, # Percent of the data that is used as validation (0-100)

        n_classes=1, # Number of classes
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
        radius=150,
        eqconst=False,
        eps=1e-4,
        gamma=0.9,
        net_size_weights=[1,8,8,4],
        net_size_post=[1,4,8,4],
        img_size = (16,16),

        # NOT USED YET
        optim='adam',
        shuffle=True,

        # whether to have a network at the end
        post_net =False,
        # whether to default partion the outputs
        bipart = False, # would make the output not a smooth function and thus gradients would be useless

        bce_weight = 0.5,

        symm_norm_L = False,
)

    experiment = wandb.init(project='DDN-NC', config=hyperparameter_defaults)
    # Config parameters are automatically set by W&B sweep agent
    args = wandb.config

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

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

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net, args, experiment, save_checkpoint=True)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
