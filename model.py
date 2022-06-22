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
        # enforce the weights being on the same device
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        # the actual layers (nc is placed into dec layer to convert to general pytorch layer)
        self.weightsNet = WeightsNet(args).to(device)
        self.nc = NormalizedCuts(eps=1) # eps sets the absolute difference between objective solutions and 0
        self.decl = DeclarativeLayer(self.nc).to(device) # converts the NC into a pytorch layer (forward/backward instead of solve/gradient)
        self.postNC = PostNC(args).to(device)

    def forward(self, x):
        x = self.weightsNet(x) # make the affinity matrix (or something else that works with)
        x = self.decl(x) # check the size of this output...
        x = self.postNC(x)
        return x

    def forward_plot(self,x):
        x1 = self.weightsNet(x) # make the affinity matrix (or something else that works with)
        x2 = self.decl(x1) # check the size of this output...
        x3 = self.postNC(x2)
        return de_minW(x1),x2,x3

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
        self.last_dim = args.img_size[0] * args.img_size[1]

        # TODO: add a channels field for forward(...), but currently only training B/W images
        # TODO: add args for these (not worried until everything works flawlessly)
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1

        self.last_size = self.last_dim # either its last_dim * last_dim
        if args.minify:
            args.last_size = args.radius # or if minified its just radius x dim

        self.layers = nn.ModuleList()

        for k in range(len(args.net_size_weights)-1): # there is probably some better looking loop, but it works
            self.layers.append(self.conv_block(c_in=args.net_size_weights[k], c_out=args.net_size_weights[k+1], 
                                kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
        self.lastcnn = nn.Conv2d(in_channels=args.net_size_weights[-1], out_channels=self.last_size, kernel_size=3, stride=1, padding=1)
        self.restrict = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.lastcnn(x)
        x = self.restrict(x)

        # combine the 32x32 * last_size into the correct output size (full matrix or not...)
        x = x.view(x.size(0), self.last_size, self.last_dim)
        if self.net_no == 0: # Passes into NC node by converting to full
            x = de_minW(x)
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

        # TODO: add a channels field for forward(...), but currently only training B/W images
        # TODO: add args for these (not worried until everything works flawlessly)
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1

        self.layers = nn.ModuleList()
        for k in range(len(args.net_size_post)-1):
            self.layers.append(self.conv_block(c_in=args.net_size_post[k], c_out=args.net_size_post[k+1], 
                                kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))

        # out channles of 1 for original input size (assuming consistent kernels, stride and padding values)
        self.lastcnn = nn.Conv2d(in_channels=args.net_size_post[-1], out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, x):
        # TODO: future model with support for multiple cuts will not hardcore channel number as 1?
        x = x.view(x.size(0), 1, self.img_size[0], self.img_size[1]) # convert from NC node into this
        for layer in self.layers:
            x = layer(x)
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