# Deep Declarative Node for Normalised Cuts
# Garth Wales - 2022
import torch
from tqdm import tqdm

def test(val_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        avg_acc = 0
        avg_loss = 0
        for input_batch, target_batch in tqdm(val_loader, desc=val_accuracy):
            input_batch, target_batch = input_batch, target_batch

            output = model(input_batch)
            val_loss = criterion(output, target_batch)

            output = (output > 0.5).float()
            val_accuracy = output.eq(target_batch).float().mean()
            
            avg_acc += val_accuracy
            avg_loss += val_loss.item()
            
        avg_acc /= len(val_loader)
        avg_loss /= len(val_loader)
    return avg_acc, avg_loss

def validate(val_loader, model, device, criterion, scheduler):
    model.eval()
    with torch.no_grad():
        for input_batch, target_batch in val_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            output = model(input_batch)
            val_loss = criterion(output, target_batch)
            scheduler.step(val_loss) # Reduce LR on plateu

            output = (output > 0.5).float()         
            val_accuracy = output.eq(target_batch).float().mean()

    return val_accuracy, val_loss

def train(train_loader, model, device, criterion, optimizer):
    model.train()
    per_epoch_loss = 0
    for input_batch, target_batch in tqdm(train_loader):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        output = model(input_batch)
        loss = criterion(output, target_batch)
        per_epoch_loss += loss
        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        if not torch.isnan(loss).any():
            loss.backward()
        optimizer.step()

        output = (output > 0.5).float()
        train_accuracy = output.eq(target_batch).float().mean()
        
    return train_accuracy, loss.item()
