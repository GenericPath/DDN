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

    train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)) # Consistent splits for everything
    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True,
                                                batch_size=args.batch_size, shuffle=args.shuffle)
    val_loader = torch.utils.data.DataLoader(val_set, pin_memory=True,
                                                batch_size=args.batch_size, shuffle=args.shuffle)

    return train_loader, val_loader


def plot_multiple_images(batch_no, images, dir='experiments/',labels=None, figsize=[32,32]):
    """
    Images [input_batch, output_batch, weights]
    provide None for elements not present
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
            if j == 2:
                # de_minW checks if it is minified weights (r,N) or already something expanded (N,N)
                img = de_minW(img[None,:])[0] # expand with dummy batches (may need to reduce to [0][0] for single channel plots?)

            img = F.to_pil_image(img)
            row_ax[j].imshow(np.asarray(img), cmap='gray')

            # useful labels include the calculated accuracy or losses...
            if labels is not None:
                row_ax[j].set_title(str(labels[i]))

    plt.tight_layout()
    plt.savefig(dir+'batch-'+str(batch_no)+'.png')
    plt.close()

def load_dataset(path, weights_path, num_images):
    """
    opens the path/dataset file and returns
    (inputs (images), outputs (segmentations), intermediate (weights))
    """
    inputs,outputs,weights = [],[],[]
    if os.path.isfile(path+'dataset'):
        with open (path+'dataset', 'rb') as fp:
            output = pickle.load(fp)
            inputs = output[0][:num_images] # images
            outputs = output[1][:num_images] # segmentations
    
    if os.path.isfile(weights_path):
        with open (weights_path, 'rb') as fp:
                output = pickle.load(fp)
                weights = output[:num_images] # images
    return inputs, outputs, weights

def data(path, args, img_size=(32,32)):
    """ Generate a simple dataset (if it doesn't already exist) 
    path - example 'data/simple01/'
    total_images - total number of images to create for the dataset
    image size - (w,h)
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' has been made')

    if not os.path.exists(path+'images/'):
        os.makedirs(path+'images/')
        print(path+'images/' + ' has been made')
    
    full_path = path+'images/'+f'{img_size[0]}-{img_size[1]}/'
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(full_path + ' has been made')
    
    weights_name ='weights-min'+str(args.minify)+'-r'+str(args.radius)
    weights_path = full_path+weights_name
    images, answers, weights = load_dataset(full_path, weights_path, args.total_images)
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

            name = full_path+"img"+str(i)+".png"
            out.save(name, "PNG")

            images.append(name)
            answers.append(answer) 
            if i >= start_weights:
                weights.append(manual_weight(name, r=args.radius, minVer=args.minify).squeeze(0)) 
                # squeeze(0) to remove batch dimension (but maintain channels or radius which could be 1)

    if start_weights != args.total_images or start_weights < start:
            with open(weights_path, 'wb') as fp:
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
        plot_multiple_images(f'{b_start}-{b_stop}', batch, dir=path)

class SimpleDatasets(Dataset):
    """ Simple white background, black rectangle dataset """
    
    def __init__(self, args, transform=None, size=(32,32)):
        """
        file (string): Path to the pickle that contains [img paths, output arrays]
        Creates the dataset if needed, and then loads it into class instance
        """
        self.network = args.network
        self.total_images = args.total_images
        self.transform = transform

        path = 'data/' + args.dataset + '/' # location to store dataset
        weights_path = path+'weights-min'+str(args.minify)+'-r'+str(args.radius)

        # make the dataset (if needed)
        data(path, args, img_size=size)
        # load the dataset
        self.images, self.segmentations,self.weights = load_dataset(path, weights_path, self.total_images)
            
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
        return de_minW(self.weights[index][None,:])

if __name__ == '__main__':
    from net_argparser import net_argparser

    args = net_argparser()
    args.network = 1
    args.total_images = 1000
    args.minify = True
    args.radius = 1

    # TODO: check loading minified weights and non-minified weights is equivalent
    
    train_loader, val_loader = get_dataset(args)

    print('load again to create experiments/')
    train_dataset = SimpleDatasets(args, transform=transforms.ToTensor())
    for i in range(5):
        row = [train_dataset.get_image(i), train_dataset.get_segmentation(i), train_dataset.get_weights(i)]
        plot_multiple_images(i, row, dir='experiments/')