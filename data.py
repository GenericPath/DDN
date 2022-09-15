import os, random, pickle
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import random_split

# local imports
from nc import de_minW, manual_weight

# imports for plot_multiple_images
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

# TODO: salt and pepper noise (e.g. 10% example below from https://github.com/loli/medpy/blob/master/notebooks/Simple%20binary%20image%20processing.ipynb)
# NOTE: equivalent to doing textures vs colour stuff individually, so maybe don't need to do this
# i, h = load("flair.nii.gz")
# i[np.random.randint(0, i.shape[0], int(0.05 * i.size)), np.random.randint(0, i.shape[1], int(0.05 * i.size))] = i.min()
# i[np.random.randint(0, i.shape[0], int(0.05 * i.size)), np.random.randint(0, i.shape[1], int(0.05 * i.size))] = i.max()
# plt.imshow(i, cmap = cm.Greys_r);

def get_dataset(args):
    """
    Creates the train_loader, val_loader
    calls SimpleDatasets() which creates the data (if needed) using data()
    """
    train_dataset = SimpleDatasets(args, transform=transforms.ToTensor())
    print(f'Total dataset size {len(train_dataset)}')

    # Training and Validation dataset
    n_val = int(len(train_dataset) * args.val)
    n_train = len(train_dataset) - n_val

    train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)) # Consistent splits for everything, TODO: args.seed
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True,
                                                batch_size=args.batch_size, shuffle=args.shuffle)
    val_loader = torch.utils.data.DataLoader(val_set, pin_memory=True,
                                                batch_size=args.batch_size, shuffle=args.shuffle)

    return train_loader, val_loader


def plot_multiple_images(batch_no, images, dir='experiments/',labels=None, figsize=[32,32], ipynb=False, cmap_name='gray'):
    """
    Images [input_batch, output_batch, weights]
    provide None for elements not present

    NOTE: use de_minW(img[None,:])[0] or similar for any minified weights
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(dir + ' has been made')
    
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
            img = images[j][i]
            img = F.to_pil_image(img)
            row_ax[j].imshow(np.asarray(img), cmap=plt.get_cmap(cmap_name))

            # useful labels include the calculated accuracy or losses...
            if labels is not None:
                row_ax[j].set_title(str(labels[j]))

    plt.tight_layout()
    if not ipynb:
        name = dir+str(batch_no)+'.png'
        plt.savefig(name)
        print(f'saved {name}')
        plt.close()
    else:
        plt.show()

def make_paths(args, path=None):
    img_size = args.img_size
    if path == None:
        path = 'data/' + args.dataset + '/'
        path = path+f'{img_size[0]}-{img_size[1]}/'    
        if not os.path.exists(path+'images/'):
            os.makedirs(path+'images/')
            print(path+'images/' + ' has been made')
    weights_name ='weights-min'+str(args.minify)+'-r'+str(args.radius)+'v2'

    return path, weights_name

def load_dataset(full_path, weights_path, num_images):
    """
    opens the path/dataset file and returns
    (inputs (images), outputs (segmentations), intermediate (weights))
    """
    inputs,outputs,weights = [],[],[]
    if os.path.isfile(full_path):
        with open (full_path, 'rb') as fp:
            output = pickle.load(fp)
            inputs = output[0][:num_images] # images
            outputs = output[1][:num_images] # segmentations
    
    if os.path.isfile(weights_path):
        with open (weights_path, 'rb') as fp:
                output = pickle.load(fp)
                weights = output[:num_images] # weights
    return inputs, outputs, weights

def data(full_path, weights_name, args):
    """ Generate a simple dataset (if it doesn't already exist) 
    path - example 'data/simple01/'
    total_images - total number of images to create for the dataset
    image size - (w,h)
    """
    img_size = args.img_size
    images, answers, weights = load_dataset(full_path+'dataset', full_path+weights_name, args.total_images)

    start = len(images)
    start_weights = len(weights)

    print(f'loaded {start} existing images from dataset')
    print(f'loaded {start_weights} existing weights from {weights_name}')

    # create all the missing weights (as the images may have been created with a different minify or radius)
    if start_weights < start:
        for i_w in tqdm(range(start_weights, start), desc='new weights'):
            name = images[i_w]
            weights.append(manual_weight(name, r=args.radius, minVer=args.minify).squeeze(0))           

    # create all the (new) images to reach total images count (appends new ones)
    if start < args.total_images:
        for i in tqdm(range(start, args.total_images), desc='new images, segs, weights'):
            # TODO: When creating dataset, enforce constraint of 50% white, 50% black to ensure nothing funny is happening

            # L gives 8-bit pixels (0-255 range of white to black)
            w,h = (random.randint(img_size[0]//3, img_size[0]), random.randint(img_size[0]//3, img_size[0]))
            x,y = (random.randint(img_size[0]//3, img_size[0]), random.randint(img_size[0]//3, img_size[0]))

            xy = [(x-w//2,y-h//2), (x+w//2,y+h//2)]
            answer = np.zeros(img_size)
            answer[xy[0][0]:xy[1][0], xy[0][1]:xy[1][1]] = 1

            # L gives 8-bit pixels (0-255 range of white to black)
            out = Image.fromarray(np.uint8(answer * 255), 'L')

            name = full_path+'images/'+"img"+str(i)+".png"
            out.save(name, "PNG")

            images.append(name)
            answers.append(answer) 
            if i >= start_weights:
                weights.append(manual_weight(name, r=args.radius, minVer=args.minify).squeeze(0)) 
                # squeeze(0) to remove batch dimension (but maintain channels or radius which could be 1)

    if start_weights != args.total_images or start_weights < start:
            with open(full_path+weights_name, 'wb') as fp:
                pickle.dump(weights, fp)
            print(f"made the {weights_name} file, {len(weights)-start_weights} new weights")

    if start != args.total_images: # if there were new images to create, then save the new pickle
        # write the new contents to disk
        output = [images, answers]
        with open(full_path+'dataset', 'wb') as fp:
            pickle.dump(output, fp)
        print(f"made the dataset file, {i - (start-1)} new images")

        # plot one example of the image, segmentation and weights
        print('create batch-num.pngs')
        train_dataset = SimpleDatasets(args, transform=transforms.ToTensor())

        num = 5 # the last 'num' images of the new stuff
        b_start = max(args.total_images-num,0)
        b_stop = args.total_images
        batch = [torch.stack([train_dataset.get_image(i) for i in range(b_start,b_stop)]), 
                torch.stack([train_dataset.get_segmentation(i) for i in range(b_start,b_stop)]), 
                torch.stack([train_dataset.get_weights(i).squeeze(0) for i in range(b_start,b_stop)])]
        plot_multiple_images(f'{b_start}-{b_stop}', batch, dir=full_path)

class SimpleDatasets(Dataset):
    """ Simple white background, black rectangle dataset """
    
    def __init__(self, args, transform=None):
        """
        file (string): Path to the pickle that contains [img paths, output arrays]
        Creates the dataset if needed, and then loads it into class instance
        """
        self.network = args.network
        self.total_images = args.total_images
        self.size = args.img_size
        self.transform = transform

        full_path, weights_name = make_paths(args)

        # make the dataset (if needed)
        data(full_path, weights_name, args)
        # load the dataset
        self.images, self.segmentations,self.weights = load_dataset(full_path+'dataset', full_path+weights_name, self.total_images)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # 1. load image
        img = cv2.imread(self.images[index], 0) # or switch to PIL.Image.open() and then img.load()?
        if self.transform is not None:
            img = self.transform(img)
        
        # 2. load target (based on network)
        if self.network == 1: # weights
            y_label = self.weights[index]
        else: # simple01
            y_label = self.segmentations[index]
            if self.transform is not None: 
                y_label = self.transform(y_label)
        return (img, y_label)
    
    # TODO: actually use these helper functions
    def get_image(self, index):
        image = cv2.imread(self.images[index], 0)
        return self.transform(image) if self.transform is not None else image

    def get_segmentation(self, index):
        segmentation = self.segmentations[index]
        return self.transform(segmentation) if self.transform is not None else segmentation

    def get_weights(self, index):
        return self.weights[index][None,:]

class CustomFolders(Dataset):
    """ Simple dataset from folders """
    
    def __init__(self, args, transform=None):
        """
        file (string): Path to the pickle that contains [img paths, output arrays]
        Creates the dataset if needed, and then loads it into class instance
        """
        self.network = args.network
        self.total_images = args.total_images
        self.size = args.img_size
        self.transform = transform

        img_path = args.img_path # TODO: add to argsparse, TODO: default to None
        seg_path = args.seg_path # TODO: add to argsparse

        full_path, weights_name = make_paths(args, path = img_path)

        # make the dataset (if needed)
        # data_texture_color(full_path, weights_name, args) # TOOD: fix this
        # TODO : plan: separate .py generates folders
        # TODO : load as folders specified on command line!
        # TODO : so not data_texture_colour here...? except weights creation...
        # TODO : just have to loop through all files in the dir to pass https://discuss.pytorch.org/t/dataloader-for-semantic-segmentation/48290/10


        # load the dataset
        self.images, self.segmentations,self.weights = load_dataset(full_path+'dataset', full_path+weights_name, self.total_images)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # 1. load image
        img = cv2.imread(self.images[index], 0) # or switch to PIL.Image.open() and then img.load()?
        if self.transform is not None:
            img = self.transform(img)
        
        # 2. load target (based on network)
        if self.network == 1: # weights
            y_label = self.weights[index]
        else: # simple01
            y_label = self.segmentations[index]
            if self.transform is not None: 
                y_label = self.transform(y_label)
        return (img, y_label)
    
    # TODO: actually use these helper functions
    def get_image(self, index):
        image = cv2.imread(self.images[index], 0)
        return self.transform(image) if self.transform is not None else image

    def get_segmentation(self, index):
        segmentation = self.segmentations[index]
        return self.transform(segmentation) if self.transform is not None else segmentation

    def get_weights(self, index):
        return self.weights[index][None,:]

if __name__ == '__main__':
    from net_argparser import net_argparser

    args = net_argparser()
    args.network = 1
    args.total_images = 10
    args.minify = True
    args.radius = 20
    args.img_size = [32,32] # the default is 32,32 anyway

    # TODO: check loading minified weights and non-minified weights is equivalent
    # TODO: check if the data default should be 1 or 0... torch.zeros vs torch.ones... for weights only training
    
    train_loader, val_loader = get_dataset(args)

    print('load again to create experiments/')
    train_dataset = SimpleDatasets(args, transform=transforms.ToTensor())
    for i in range(5):
        row = [train_dataset.get_image(i), train_dataset.get_segmentation(i), de_minW(train_dataset.get_weights(i))]
        plot_multiple_images(i, row, dir='experiments/')