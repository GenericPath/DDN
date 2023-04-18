import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelSimilarityCNN(nn.Module):
    def __init__(self, input_shape):
        super(PixelSimilarityCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64*54*54, out_features=128)
        
    def forward(self, x):
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten feature maps
        x = x.view(-1, 64*54*54)
        
        # Apply fully connected layer
        x = F.relu(self.fc1(x))
        
        # Compute pairwise similarities between feature maps
        similarities = torch.mm(x, x.t())
        
        # Rescale similarities to be between 0 and 1
        similarities = similarities / (torch.sum(similarities, dim=-1, keepdim=True) + 1e-7)
        
        return similarities
