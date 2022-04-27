from model import *
# Variables
checkpoint_name = 'checkpoints/checkpoint_epoch48.pth'
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
net = Net()

checkpoint = torch.load(checkpoint_name)
net.load_state_dict(checkpoint['model_state_dict'])

net.eval()
with torch.no_grad():
    for input_batch, target_batch in val_loader:
        input_batch, target_batch = input_batch, target_batch

        output = net(input_batch)
        output = (output > 0.5).float()
        val_accuracy = output.eq(target_batch).float().mean()