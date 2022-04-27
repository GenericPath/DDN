# Deep Declarative Node for Normalised Cuts
# Garth Wales - 2022

from numpy import outer
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from tqdm import tqdm
# import pickle
import time, os, random

# locally defined imports
from unetmini import Net #UNet
from data import data, Simple01

from torchsummary import summary

def train(args):
    hparams = vars(args)

    val_percent = args.val
    save_checkpoint = True
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    dataset = 'simple01/'
    results = 'experiments/'+dataset
    path = 'data/'+dataset # location to store dataset
    dir_checkpoint = 'checkpoints/'+dataset

    run = "2"
    if args.name is not None:
        dir_checkpoint = dir_checkpoint + args.name + '/'
        results = results + args.name + '/'
        if not os.path.exists(results):
                    os.makedirs(results)
                    print(results + ' has been made')
        if not os.path.exists(dir_checkpoint):
                    os.makedirs(dir_checkpoint)
                    print(dir_checkpoint + ' has been made')
        
        
    writer = SummaryWriter(results, comment=run)
    # add_hparams(hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None)

    # can do writer for a series of experiements done in a loop with lr*i etc etc..
    # hparams = { ... }

    data(path, args.total_images) # make the dataset
    path = path + str(args.total_images) + '/'

    train_dataset = Simple01(path+'dataset', transform=transforms.ToTensor())

    print(f'Total dataset size {len(train_dataset)}')
    # Training and Validation dataset
    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val

    train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True,
                                                batch_size=args.batch_size, shuffle=args.shuffle)
    val_loader = torch.utils.data.DataLoader(val_set, pin_memory=True,
                                                batch_size=args.batch_size, shuffle=args.shuffle)
    
    # Test dataset shapes
    # x,y = next(iter(train_loader))
    # print(x.shape)
    # print(y.shape)
    # print('Dataset : %d EA \nDataLoader : %d SET' % (len(train_dataset),len(train_loader)))

    torch.backends.cudnn.benchmark = True
    net = Net(args)
    net = net.to(device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

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
        'hparam/accuracy': 0, 
        'hparam/loss': 0#*i as an example this could all be in a loop....
    }

    best_accuracy = 0
    for epoch in range(args.epochs):
        # TRAIN
        net.train()
        per_epoch_loss = 0
        # start_time = time.time()
        for index, (input_batch, target_batch) in enumerate(tqdm(train_loader)):
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            output = net(input_batch)
            loss = criterion(output, target_batch)
            per_epoch_loss += loss
            # Compute gradient and do optimizer step
            optimizer.zero_grad()
            if not torch.isnan(loss).any():
                loss.backward()
            optimizer.step()
            
            # print(f'Batch #{index},\ttime\t{time.time()-start_time}')
            # calculate accuracy, output metrics
            global_step = epoch * len(train_loader) + index

            output = (output > 0.5).float()
            train_accuracy = output.eq(target_batch).float().mean()
            
            writer.add_scalar("Acc/train", train_accuracy, global_step=global_step)
            writer.add_scalar("Loss/train", loss.item(), global_step=global_step)
            # start_time = time.time()
        
        per_epoch_loss /= index
        writer.add_scalar("AvgLoss/train", per_epoch_loss, global_step=epoch)

        # TEST AGAINST VALIDATION
        net.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                val_loss = criterion(output, target_batch)
                val_epoch_loss = val_loss / len(val_loader)
                scheduler.step(val_loss) # Reduce LR on plateu

                output = net(input_batch)
                output = (output > 0.5).float()
                val_accuracy = output.eq(target_batch).float().mean()
                
                writer.add_scalar("Loss/val", val_epoch_loss, epoch)
                writer.add_scalar("Acc/val", val_accuracy, epoch)

        # SAVE IF IT IS THE BEST
        if save_checkpoint: # maybe only save if the accuracy is the highest we have seen so far...
            if val_accuracy >= best_accuracy and val_accuracy != 0:
                best_accuracy = val_accuracy
                if not os.path.exists(dir_checkpoint):
                    os.makedirs(dir_checkpoint)
                    print(dir_checkpoint + ' has been made')
                torch.save(net.state_dict(), str(dir_checkpoint + f'checkpoint_epoch{epoch+1}.pth'))
                print(f'Checkpoint {epoch + 1} saved!')

    # Create metrics, and convert hparams to strings to store with tensorboard
    metrics['hparam/accuracy'] = best_accuracy
    metrics['hparam/loss'] = loss.item()
    for keys in hparams:
            hparams[keys] = str(hparams[keys])
    hparams = {str(j) : str(i) for i,j in enumerate(hparams)}
    writer.add_hparams(hparam_dict=hparams, metric_dict=metrics)
    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.use_deterministic_algorithms(True)

    # train(args)

    # Variables
    checkpoint_name = '/checkpoints/simple01/checkpoint_epoch48.pth'
    total_images = 300
    val_percent = 0.1


    # Setup dataset
    dataset = 'simple01/'
    path = 'data/'+dataset # location to store dataset
    data(path, total_images) # make the dataset
    train_dataset = Simple01(path+'dataset', transform=transforms.ToTensor())
    print(f'Total dataset size {len(train_dataset)}')
    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val

    train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True,
                                                batch_size=1, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, pin_memory=True,
                                                batch_size=1, shuffle=False)


    # Load the model
    net = Net(args)

    checkpoint = torch.load(checkpoint_name)
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()
    with torch.no_grad():
        for input_batch, target_batch in val_loader:
            input_batch, target_batch = input_batch, target_batch

            output = net(input_batch)
            output = (output > 0.5).float()
            val_accuracy = output.eq(target_batch).float().mean()
