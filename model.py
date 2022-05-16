import torch
import torch.nn as nn

# local imports
from nc import NormalizedCuts
from node import DeclarativeLayer
from data import get_weights_vars

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.weightsNet = WeightsNet(args)
        self.nc = NormalizedCuts(eps=1) # eps sets the absolute difference between objective solutions and 0
        self.decl = DeclarativeLayer(self.nc) # converts the NC into a pytorch layer (forward/backward instead of solve/gradient)
        self.postNC = PostNC()

    def forward(self, x):
        x = self.weightsNet(x) # make the affinity matrix (or something else that works with)
        x = self.decl(x) # check the size of this output...
        x = self.postNC(x)
        return x

class WeightsNet(nn.Module):
    def __init__(self, args):
        super(WeightsNet, self).__init__()
        self.r, self.min = get_weights_vars(args)
        self.min = True # TODO : not hardcode this... (datasets simple/weights with a min option for either)
        self.block1 = self.conv_block(c_in=args.net_size[0], c_out=args.net_size[1], kernel_size=3, stride=1, padding=1)
        self.block2 = self.conv_block(c_in=args.net_size[1], c_out=args.net_size[2], kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=args.net_size[2], c_out=args.net_size[3], kernel_size=3, stride=1, padding=1)
        if self.min: self.lastcnn = nn.Conv2d(in_channels=args.net_size[3], out_channels=self.r+1, kernel_size=3, stride=1, padding=1)
        else: self.lastcnn = nn.Conv2d(in_channels=args.net_size[3], out_channels=args.net_size[4], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.lastcnn(x))

        # combine the 32x32 image filters into the correct output size (full matrix or not...)
        if self.min: x = x.view(x.size(0), 1, self.r+1, 1024) # r+1 as radius + itself
        else: x = x.view(x.size(0), 1, 1024, 1024) # full matrix (with majority zeros)

        return x
    def conv_block(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU()
            )
        return seq_block


class PostNC(nn.Module):
    def __init__(self):
        super(PostNC, self).__init__()
        self.block1 = self.conv_block(c_in=1, c_out=128, kernel_size=3, stride=1, padding=1)
        self.block2 = self.conv_block(c_in=128, c_out=256, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = x.view(x.size(0), 1, 32, 32)
        x = self.block1(x)
        x = self.block2(x)
        x = self.lastcnn(x)
        # x = torch.sigmoid(x) # this is replaced with using the correct loss function (BCEWithLogitsLoss as it is more stable!)
        return x
    def conv_block(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU()
            )
        return seq_block