import torch
import torch.nn as nn

def main():
    input = torch.randn(32, 1, 32, 32)
    print(f'input {input.shape}')
    for i in input:
        print(input[i,...])

    upsample = nn.ConvTranspose2d(1, 1, kernel_size=5, stride=1, padding=0) # in_channels, out_channels = 1 for black and white
    upsample2 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=0)
    upsample3 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=0)
    y = upsample3(upsample2(upsample(input)))
    print(f'output {y.shape}')
    # Affinity matrix is supposed to be NxN where N = w*h of input.. clearly isn't going to grow quickly enough...
    # instead found a number of papers about affinity matrix with convolutional neural networks.


if __name__ == '__main__':
    main()
