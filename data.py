import os, random, pickle
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import cv2

from torchvision import transforms
from torch.utils.data import random_split

from nc import de_minW, manual_weight

import argparse

# plot_multiple_images
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def plot_multiple_images(batch_no, images, labels=None, minified=False, figsize=[32,32]):
    """
    Images [input_batch, output_batch, ...]
    """
    # TODO : use https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html instead?

    # settings
    N = min(map(len, images)) # length of the shortest array
    nrows, ncols = N, len(images)  # array of sub-plots

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    # can access individual plot with ax[row][col]

    # plot image on each sub-plot
    for i, row_ax in enumerate(ax): # could flatten if not explicitly doing in pairs (ax.flat)
        for j in range(ncols):
            img = images[j][i].cpu()
            if j == 1 and minified:
                img = de_minW(img[None,None,:])[0][0] # exapnd with dummy batches and channels  
            img = F.to_pil_image(img)
            row_ax[j].imshow(np.asarray(img))
            if labels is not None:
                row_ax[j].set_title(str(labels[i]))

    plt.tight_layout()
    plt.savefig('experiments/batch-'+str(batch_no)+'.png')
    plt.close()

def get_dataset(args):
    train_dataset = SimpleDatasets(args, transform=transforms.ToTensor()) #path/dataset is a pickle containing (image paths, targets)

    print(f'Total dataset size {len(train_dataset)}')
    # Training and Validation dataset
    n_val = int(len(train_dataset) * args.val)
    n_train = len(train_dataset) - n_val

    train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)) # Consistent splits for everything
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True,
                                                batch_size=args.batch_size, shuffle=args.shuffle)
    val_loader = torch.utils.data.DataLoader(val_set, pin_memory=True,
                                                batch_size=args.batch_size, shuffle=args.shuffle)

    return train_loader, val_loader

def data(path, args, img_size=(32,32)):
    """ Generate a simple dataset (if it doesn't already exist) 
    path - example 'data/simple01/'
    total_images - total number of images to create for the dataset
    image size - (w,h)
    """

    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' has been made')
        
    if not os.path.isfile(path+'dataset'):
        images = []
        answers = []
        for i in range(args.total_images):
            # L gives 8-bit pixels (0-255 range of white to black)
            w,h = (random.randint(img_size[0]//3, img_size[0]), random.randint(img_size[0]//3, img_size[0]))
            x,y = (random.randint(img_size[0]//3, img_size[0]), random.randint(img_size[0]//3, img_size[0]))

            xy = [(x-w//2,y-h//2), (x+w//2,y+h//2)]
            answer = np.zeros(img_size)
            answer[xy[0][0]:xy[1][0], xy[0][1]:xy[1][1]] = 1

            # L gives 8-bit pixels (0-255 range of white to black)
            out = Image.fromarray(np.uint8(answer * 255), 'L')

            name = path+"img"+str(i)+".png"
            out.save(name, "PNG")
            images.append(name)
            
            if 'weights' == args.dataset: # append weights matrix as answer TODO : make it create a separate weights dataset when making the main one... so they are the same images but for different stuff?
                answers.append(manual_weight(name, r=args.radius, minVer=args.minify).squeeze()) 
            else: 
                answers.append(answer) # otherwise append output as answer (for simple01, etc)

        # write the answers to a txt file to visually inspect (while initially setting everything up)
        ans_out = open(path+'answers'+'.txt', 'w')
        # if 'weights' == args.dataset:
        #     for row in answers:
        #         for item in row:
        #             ans_out.write(str(item.item()) + ' ')
        #         ans_out.write('\n')
        #     ans_out.write('\n---\n')
        # else:
        for answer in answers: # [[b,c,row,item],...]
            for rows in answer:
                for item in rows:
                    ans_out.write(str(item.item()) + ' ')
                ans_out.write('\n')
            ans_out.write('\n---\n')
        ans_out.close()

        # save the input, output pairs to a file
        output = [images, answers]
        with open(path+'dataset', 'wb') as fp:
            pickle.dump(output, fp)
        print("made the dataset file")

class SimpleDatasets(Dataset):
    """ Simple white background, black rectangle dataset """
    
    def __init__(self, args, transform=None):
        """
        file (string): Path to the pickle that contains [img paths, output arrays]
        """
        self.args = args
        # append = '../../' if args.production else ''
        append = ''
        path = append + 'data/' + args.dataset + '/' + str(args.total_images) # location to store dataset
        if args.minify:
            path += 'min' + str(args.radius) + '/'
        else:
            path += '/'
        data(path, args) # make the dataset

        with open (path+'dataset', 'rb') as fp:
            output = pickle.load(fp)
            self.images = output[0] # images
            self.segmentations = output[1] # segmentation
        self.transform = transform
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = cv2.imread(self.images[index], 0)
        # or switch to PIL.Image.open() and then img.load()?
        y_label = self.segmentations[index]
        
        if self.transform is not None:
            img = self.transform(img)
            if 'weights' not in self.args.dataset:
                y_label = self.transform(y_label)
            
        return (img, y_label)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='debugging arguments')
    # main args
    parser.add_argument('--dataset', type=str, default='weights', help='dataset to use: weights, simple01')
    parser.add_argument('--total-images', '-ti', metavar='N', type=int, default=10, dest='total_images', help='total number of images in dataset')
    parser.add_argument('--production', default=False, type=bool, help='Production mode: If true run in a separate folder on a copy of the python scripts')

    # minify arguments
    parser.add_argument('--minify', default=True, type=bool, help='minify the weights mode (for the PreNC portion)')
    parser.add_argument('--radius', '-r', default=1, type=int, help='radius value for expected weights (only relevant for minified version)')

    # get_dataset args
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.1,
                    help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle batches')

    args = parser.parse_args()
    args.production = False

    append = ''
    path = append + 'data/' + args.dataset + '/' + str(args.total_images) # location to store dataset
    if args.minify:
        path += 'min' + str(args.radius) + '/'
    else:
        path += '/'

    data(path, args, img_size=(32,32))
    train_loader, val_loader = get_dataset(args)
    for i, (input, output) in enumerate(train_loader):
        plot_multiple_images(i, [input, output], minified=(args.dataset == 'weights'))
        print(i)
        if i > 10:
            exit()

    # test the dataset (and test plot_multiple_images)
    # plot_multiple_images

