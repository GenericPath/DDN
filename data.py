import os, random, pickle
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import cv2

from torchvision import transforms
from torch.utils.data import random_split


def get_datasets(args):
    dataset = 'simple01/'
    path = 'data/'+dataset # location to store dataset

    data(path, args.total_images) # make the dataset
    path = path + str(args.total_images) + '/'

    train_dataset = Simple01(path+'dataset', transform=transforms.ToTensor())

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


def data(path, total_images=300):
    """ Generate a simple dataset (if it doesn't already exist) """
    img_size = (32,32) # image size (w,h)

    path = path + str(total_images) + '/'

    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' has been made')
        
    if not os.path.isfile(path+'dataset'):
        images = []
        answers = []
        for i in range(total_images):
            # L gives 8-bit pixels (0-255 range of white to black)
            w,h = (random.randint(img_size[0]//3, img_size[0]), random.randint(img_size[0]//3, img_size[0]))
            x,y = (random.randint(img_size[0]//3, img_size[0]), random.randint(img_size[0]//3, img_size[0]))

            xy = [(x-w//2,y-h//2), (x+w//2,y+h//2)]
            answer = np.zeros(img_size)
            answer[xy[0][0]:xy[1][0], xy[0][1]:xy[1][1]] = 1
            answers.append(answer)
            
            # L gives 8-bit pixels (0-255 range of white to black)
            out = Image.fromarray(np.uint8(answer * 255), 'L')
            
            name = path+"img"+str(i)+".png"
            out.save(name, "PNG")
            print(name)
            images.append(name)
            
        output = [images, answers]
        with open(path+'dataset', 'wb') as fp:
            pickle.dump(output, fp)
        print("made the dataset file")

class Simple01(Dataset):
    """ Simple white background, black rectangle dataset """
    
    def __init__(self, file, transform=None):
        """
        file (string): Path to the pickle that contains [img paths, output arrays]
        """
        with open (file, 'rb') as fp:
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
            y_label = self.transform(y_label)
            
        return (img, y_label)