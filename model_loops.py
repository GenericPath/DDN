# Deep Declarative Node for Normalised Cuts
# Garth Wales - 2022
import torch, os
from tqdm import tqdm
import matplotlib.pyplot as plt

def save_images(in_batch, out_batch, name):
    in_batch, out_batch = in_batch.squeeze(), out_batch.squeeze()

    n = len(in_batch) * 2
    f = plt.figure()
    for i in range(0, n, 2):
        # Debug, plot figure
        f.add_subplot(1, n, i + 1)
        plt.imshow(in_batch[i])
        f.add_subplot(1, n, i + 2)
        plt.imshow(out_batch[i])

    plt.savefig(name+'.png')

def test(val_loader, model, criterion, device, args):
    model.eval()

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
            test_accuracy = output.eq(target_batch).float().mean()
            
            avg_acc += test_accuracy
            avg_loss += val_loss.item()

            save_images(target_batch, output, 'batch'+str(i))

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
