import torch
import torch.nn as nn

# import torch
# from torch.nn import Module
# from torch.nn import Sequential
# from torch.nn import Conv2d, Dropout2d, MaxPool2d, ReLU, UpsamplingNearest2d


# # Based on https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/unet.py#L19
# class UNet(Module):

#     def __init__(self, num_classes):
#         super(UNet, self).__init__()

#         # Use padding 1 to mimic `padding='same'` in keras,
#         # use this visualization tool https://ezyang.github.io/convolution-visualizer/index.html
#         self.block1 = Sequential(
#             Conv2d(1, 32, kernel_size=3, padding=1),
#             ReLU(),
#             Dropout2d(0.2),
#             Conv2d(32, 32, kernel_size=3, padding=1),
#             ReLU(),
#         )
#         self.pool1 = MaxPool2d((2, 2))

#         self.block2 = Sequential(
#             Conv2d(32, 64, kernel_size=3, padding=1),
#             ReLU(),
#             Dropout2d(0.2),
#             Conv2d(64, 64, kernel_size=3, padding=1),
#             ReLU(),
#         )
#         self.pool2 = MaxPool2d((2, 2))

#         self.block3 = Sequential(
#             Conv2d(64, 128, kernel_size=3, padding=1),
#             ReLU(),
#             Dropout2d(0.2),
#             Conv2d(128, 128, kernel_size=3, padding=1),
#             ReLU()
#         )

#         self.up1 = UpsamplingNearest2d(scale_factor=2)
#         self.block4 = Sequential(
#             Conv2d(192, 64, kernel_size=3, padding=1),
#             ReLU(),
#             Dropout2d(0.2),
#             Conv2d(64, 64, kernel_size=3, padding=1),
#             ReLU()
#         )

#         self.up2 = UpsamplingNearest2d(scale_factor=2)
#         self.block5 = Sequential(
#             Conv2d(96, 32, kernel_size=3, padding=1),
#             ReLU(),
#             Dropout2d(0.2),
#             Conv2d(32, 32, kernel_size=3, padding=1),
#             ReLU()
#         )

#         self.conv2d = Conv2d(32, num_classes, kernel_size=1)

#     def forward(self, x):
#         out1 = self.block1(x)
#         out_pool1 = self.pool1(out1)

#         out2 = self.block2(out_pool1)
#         out_pool2 = self.pool1(out2)

#         out3 = self.block3(out_pool2)

#         out_up1 = self.up1(out3)
#         # return out_up1
#         out4 = torch.cat((out_up1, out2), dim=1)
#         out4 = self.block4(out4)

#         out_up2 = self.up2(out4)
#         out5 = torch.cat((out_up2, out1), dim=1)
#         out5 = self.block5(out5)

#         out = self.conv2d(out5)

#         return out

class PreNC(nn.Module):
    def __init__(self):
        super(PreNC, self).__init__()
        self.block1 = self.conv_block(c_in=1, c_out=128, kernel_size=3, stride=1, padding=1)
        self.block2 = self.conv_block(c_in=128, c_out=256, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=256, c_out=512, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc = nn.Linear(out_features=10404)
    def forward(self, x):
        x = self.block1(x)
        # x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.maxpool(x)
        x = self.relu(self.lastcnn(x))
        # combine the 1024 filters of 32x32 images into a singular 1024x1024 matrix (affinity matrix)
        x = x.view(x.size(0), 1, 1024, 1024)
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
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = x.view(x.size(0), 1, 32, 32)
        x = self.block1(x)
        x = self.block2(x)
        x = self.lastcnn(x)
        return x
    def conv_block(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU()
            )
        return seq_block