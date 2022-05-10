# Deep Declarative Node for Normalised Cuts
# Garth Wales - 2022
import torch, os
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_multiple_images(batch_no, images, labels=None, figsize=[32,32]):
    """
    Images [input_batch, output_batch, ...]
    """
    # settings
    N = min(map(len, images)) # length of the shortest array
    nrows, ncols = N, len(images)  # array of sub-plots

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    # can access individual plot with ax[row][col]

    # plot image on each sub-plot
    for i, row_ax in enumerate(ax): # could flatten if not explicitly doing in pairs (ax.flat)
        for j in range(ncols):
            row_ax[j].imshow(images[j][i].cpu())
            if labels is not None:
                row_ax[j].set_title(str(labels[i]))

    plt.tight_layout(True)
    plt.savefig('batch-'+str(batch_no)+'.png')
    plt.close()

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
        for input_batch, target_batch in tqdm(val_loader, desc=avg_acc):
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

            # labels = accuracy per image in batch?
            plot_multiple_images(i, [input_batch, output], labels=batch_accuracy, figsize=[x,y])

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
    for input_batch, target_batch in tqdm(train_loader):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        output = model(input_batch)
        loss = criterion(output, target_batch)
        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        if not torch.isnan(loss).any():
            loss.backward()
        optimizer.step()

        output = (output > 0.5).float()
        train_accuracy += output.eq(target_batch).float().mean()
        train_loss += loss.item()

    train_accuracy /= len(train_loader)
    train_loss/= len(train_loader) # Loss is averaged for batch size, to avoid having to tune for scaling size of learning rates etc
    return train_accuracy, train_loss
