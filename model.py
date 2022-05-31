import torch
import torch.nn as nn

# local imports
from nc import NormalizedCuts, de_minW
from node import DeclarativeLayer

class Net(nn.Module):
    """
    WeightsNet -> NC -> PostNC
    """
    def __init__(self, args):
        super(Net, self).__init__()
        self.weightsNet = WeightsNet(args)
        self.nc = NormalizedCuts(eps=1) # eps sets the absolute difference between objective solutions and 0
        self.decl = DeclarativeLayer(self.nc) # converts the NC into a pytorch layer (forward/backward instead of solve/gradient)
        self.postNC = PostNC(args)

    def forward(self, x):
        x = self.weightsNet(x) # make the affinity matrix (or something else that works with)
        x = self.decl(x) # check the size of this output...
        x = self.postNC(x)
        return x

class WeightsNet(nn.Module):
    """
    Just learns the weights (to feed into NC)
    if args.min then learns only the diagonals (as the rest is sparse)
    """
    def __init__(self, args):
        super(WeightsNet, self).__init__()
        self.r = args.radius
        self.min = args.minify
        self.net_no = args.network
        self.img_size = args.img_size
        self.last_dim = self.img_size[0] * self.img_size[1]

        # CNN layers
        self.block1 = self.conv_block(c_in=args.net_size[0], c_out=args.net_size[1], kernel_size=3, stride=1, padding=1)
        self.block2 = self.conv_block(c_in=args.net_size[1], c_out=args.net_size[2], kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=args.net_size[2], c_out=args.net_size[3], kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=args.net_size[3], out_channels=args.net_size[4], kernel_size=3, stride=1, padding=1)
        # NOTE: net_size[4] is automatically changed to self.r in run_model if needed, but otherwise it is automatically changed
        self.restrict = nn.Sigmoid() # was previously a ReLU

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.restrict(self.lastcnn(x))

        # combine the 32x32 image filters into the correct output size (full matrix or not...)
        if self.min: 
            if self.net_no == 0: # Passes into NC node
                x = x.view(x.size(0), 1, self.r, self.last_dim)
                x = de_minW(x)
            elif self.net_no == 1: # Just trains for weights as output (minified)
                x = x.view(x.size(0), 1, self.r, self.last_dim)
        else: 
            x = x.view(x.size(0), 1, self.last_dim, self.last_dim) # full matrix (with majority zeros)
        return x

    def conv_block(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU()
            )
        return seq_block


class PostNC(nn.Module):
    def __init__(self, args):
        super(PostNC, self).__init__()
        self.args = args
        self.img_size = args.img_size
        self.block1 = self.conv_block(c_in=1, c_out=4, kernel_size=3, stride=1, padding=1)
        self.block2 = self.conv_block(c_in=4, c_out=8, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.view(x.size(0), 1, self.img_size[0], self.img_size[1])
        x = self.block1(x)
        x = self.block2(x)
        x = self.lastcnn(x)
        # x = torch.sigmoid(x) # NOTE: this is replaced with using the correct loss function (BCEWithLogitsLoss as it is more stable!)
        return x

    def conv_block(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU()
            )
        return seq_block

# TODO: add tests which verify input, output pairs have correct dimensions