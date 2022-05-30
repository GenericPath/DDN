# Deep Declarative Node for Normalised Cuts
# Garth Wales - 2022
import torch, os
from tqdm import tqdm

# local imports
from data import plot_multiple_images

def test(val_loader, model, criterion, device, args):
    model.eval()

    if not args.name:
        args.name = '.'
    dir = args.name + '/outputs'
    if not os.path.exists(dir):
            os.makedirs(dir)
    with torch.no_grad():
        avg_acc, avg_loss = 0,0
        i = 0
        for input_batch, target_batch in tqdm(val_loader, desc=avg_acc, ascii=True):
            i += 1
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            output = model(input_batch)
            val_loss = criterion(output, target_batch)

            output = (output > 0.5).float()
            b,c,x,y = target_batch.shape
            batch_accuracy = (output.eq(target_batch).float().sum(dim=(-2,-1)) / (x*y)) * 100 # outputs: [b * 1] where 1 is the percentage accuracy
            test_accuracy = output.eq(target_batch).float().mean()
            
            avg_acc += test_accuracy
            avg_loss += val_loss.item()

            plot_multiple_images(i, [input_batch, output], dir=dir, labels=batch_accuracy, figsize=[x,y])

        avg_acc /= len(val_loader)
        avg_loss /= len(val_loader)

    # TODO: save the outputs with this code
    # dir = 'results/'+args.name
    # os.makedirs(dir)
    # with open(dir+'output.txt', 'w') as file:
    #     file.write()
    return avg_acc, avg_loss

def validate(val_loader, model, device, criterion, scheduler):
    model.eval()
    with torch.no_grad():
        val_accuracy, val_loss = 0,0
        for input_batch, target_batch in val_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            output = model(input_batch)
            val_loss += criterion(output, target_batch)
            scheduler.step(val_loss) # Reduce LR on plateu

            output = (output > 0.5).float()         
            val_accuracy += output.eq(target_batch).float().mean()
            
        val_accuracy /= len(val_loader)
        val_loss /= len(val_loader)
    return val_accuracy, val_loss

def train(train_loader, model, device, criterion, optimizer):
    model.train()
    train_accuracy, train_loss = 0,0
    for input_batch, target_batch in tqdm(train_loader, ascii=True):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        output = model(input_batch)
        loss = criterion(output, target_batch)
        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        if not torch.isnan(loss).any():
            loss.backward()
        optimizer.step()

        test_output = (output > 0.5).float()
        train_accuracy += output.eq(test_output).float().mean()
        train_loss += loss.item()

    train_accuracy /= len(train_loader)
    train_loss/= len(train_loader) # Loss is averaged for batch size, to avoid having to tune for scaling size of learning rates etc
    return train_accuracy, train_loss

def accuracy(output, target):
    output = (output > 0.5).float()
    return output.eq(target).float().mean()

if __name__ == '__main__':
    import torch.nn as nn
    from torchvision import transforms

    # local imports
    from net_argparser import net_argparser
    from data import get_dataset, SimpleDatasets
    from model import Net, WeightsNet

    # 1. Verify accuracy works on known samples (and pertubations), plus does the model sizes match expected
    args = net_argparser()

    # hardcode some values :)
    args.batch_size = 1
    args.minify = True
    args.dataset = 'simple01'

    args.network = 0
     # load everything
    if args.network == 1:
        model = WeightsNet(args)
    else:
        model = Net(args)
    
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = SimpleDatasets(args, transform=transforms.ToTensor())
    sample=[train_dataset.get_image(0)[None,:], train_dataset.get_segmentation(0), train_dataset.get_weights(0)]
    # add batch dimension to input image with [None,:]

    print(f'accuracy input input {accuracy(sample[0], sample[0])}')
    print(f'accuracy seg seg {accuracy(sample[1], sample[1])}')
    print(f'accuracy net0(input) seg {accuracy(model(sample[0]), sample[1])}')

    model = WeightsNet(args)
    print(f'accuracy net0(input) weights {accuracy(model(sample[0]), sample[2])}')

    # TODO: test loss, test accuracy with progressively more random flipped bits
    # see if it all functions

    # TODO: test if the way it is used in main via train/validate makes any sense

    # for input_batch, target_batch in tqdm(train_loader, ascii=True):
    #     output = model(input_batch)
    #     train_loss = criterion(output, target_batch)
    #     # convert to binary classification outputs?
    #     output_new = (output > 0.5).float()

    #     b,c,x,y = target_batch.shape
    #     batch_accuracy = (output_new.eq(target_batch).float().sum(dim=(-2,-1)) / (x*y)) * 100 # outputs: [b * 1] where 1 is the percentage accuracy
    #     test_accuracy = output_new.eq(target_batch).float().mean()



    # TODO: also double check everything is good with the model_loops :)
    # t_acc, t_loss = train(train_loader, model, 'cpu', criterion, optimizer)
    # v_acc, v_loss = validate(val_loader, model, 'cpu', criterion, scheduler)

