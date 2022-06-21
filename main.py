# Deep Declarative Node for Normalised Cuts
# Garth Wales - 2022
import torch

import random, os, shutil

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

# from data import get_dataset, plot_multiple_images
from data import *
from model_loops import test, train, validate
from model import Net, WeightsNet
from net_argparser import net_argparser

import torch.utils.tensorboard as tb
# import wandb # replacing tensorboard, fun to try out

# Maybe add this later
# from torchsummary import summary

def main():
    # Parse commandline arguments
    args = net_argparser()

    # Pre-model setup
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    cudnn.benchmark = True # Cuda optimisations when using a fixed input size

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu' # hardcode to use cpu when neccessary
    print(f'Using device {device}')

    if args.name:
        results = args.name + '/'
        if not os.path.exists(results):
            os.makedirs(results)
        args.writer = tb.SummaryWriter(results)

    # Create the model, loss, optimizer and scheduler
    if args.network == 1:
        model = WeightsNet(args)
    else:
        model = Net(args)

    # wandb.init(project='ddn')    
    # wandb.config = args # NOTE: not sure if this is gonna work
    # model = model.to(device=device)
    # wandb.watch(model)

    # TODO: add logging for images e.g. wandb.log({"examples" : [wandb.Image(im) for im in images_t]})
    # TODO: add table for images https://docs.wandb.ai/guides/integrations/pytorch


    criterion = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss to replace BCE with Sigmoid
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr) # TODO : add weight decay and betas as options
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5) # TODO : adjust patience

    # Load the weights of a saved network (if provided)
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            best_error = checkpoint['best_error']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
            return()

    # Load dataset, setup dict to pass to other funcs
    train_loader, val_loader = get_dataset(args)

    # Evaluate the network (and don't train)
    if args.test:
        avg_acc, avg_loss = test(val_loader, model, criterion, device, args)
        print(f'Evaluation: avg acc - {avg_acc}, avg_loss - {avg_loss}')
        return

    # Train the network (and test against the validation data)
    best_acc = 0
    best_error = float('inf')
    train_dataset = SimpleDatasets(args, transform=transforms.ToTensor())
    for epoch in range(args.start_epoch, args.epochs):

        t_acc, t_loss = train(train_loader, model, device, criterion, optimizer) # TODO : check if this scheme makes sense (with opt and scheduler...)
        v_acc, v_loss = validate(val_loader, model, device, criterion, scheduler) # scheduler will change LR on val plateau, optim will

        if epoch % 10 == 0: # every 10, output what everything looks like
            data = [train_dataset.get_image(0)[None,:], train_dataset.get_segmentation(0), de_minW(train_dataset.get_weights(0))]
            imgs = [data[1], model.forward_plot(data[0])]
            plot_multiple_images(epoch, imgs, cmap_name='jet')

        # Currently best is based on acc, could be changed for loss
        is_best = v_acc > best_acc
        best_error = min(v_loss, best_error)
        best_acc = min(v_acc, best_acc)

        # wandb.log({"loss/val": v_loss,
        #             "acc/val": v_acc,
        #             "loss/train":t_loss,
        #             "acc/train": t_acc})
        if args.writer:
            args.writer.add_scalar("Loss/val", v_loss, epoch)
            args.writer.add_scalar("Acc/val", v_acc, epoch)
            args.writer.add_scalar("Acc/train", t_acc, epoch)
            args.writer.add_scalar("Loss/train", t_loss, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_error': best_error,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, dir=results, filename='latest_epoch')

    if args.writer:
        args.writer.close()

def save_checkpoint(state, is_best, dir='', filename='checkpoint'):
    torch.save(state, dir + filename + '.pth.tar')
    if is_best:
        shutil.copyfile(dir + filename + '.pth.tar', dir + 'model_best.pth.tar')

if __name__ == '__main__':
    main()