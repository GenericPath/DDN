import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import cv2

import sys
sys.path.append("..")
from nc_suite import *

# Initialize W&B
wandb.init(project="pixelwise-similarity-matrix", name='baby_weight_test', mode='disabled')

# Define the CNN architecture
class SimilarityCNN(nn.Module):
    def __init__(self):
        super(SimilarityCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256 * n * n, n * n)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 256 * n * n)
        x = self.fc(x)
        return x

# params
lr = 0.001

# Set the size of the image and the number of pixels
n = 28
num_pixels = n * n

# Generate a random image with 3 color channels
img_baby = cv2.imread("../data/test/3.jpg",0)
x = torch.tensor(cv2.resize(img_baby, (28,28)))

# Generate a random target similarity matrix
y = torch.tensor(intens_posit_wm(img_baby))

# Create the CNN model
model = SimilarityCNN()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Log the model and hyperparameters to W&B
wandb.watch(model)
wandb.config.learning_rate = lr

# Train the model for 100 epochs
for epoch in range(100):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    output = model(x)
    loss = criterion(output.view(num_pixels, num_pixels), y)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Log the loss to W&B
    wandb.log({"Loss": loss.item()})

    # Print the loss
    print("Epoch {}: Loss = {}".format(epoch+1, loss.item()))
