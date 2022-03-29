import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from PIL import Image

class Simple01(Dataset):
    """ Simple white background, black rectangle dataset """
    
    def __init__(self, file):
        """
        file (string): Path to the pickle that contains [img paths, output arrays]
        """
        with open (file, 'rb') as fp:
            output = pickle.load(fp)
            self.images = output[0] # images
            self.segmentations = output[1] # segmentation
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
            
        img = Image.open(self.images[index])
        y_label = torch.tensor(self.segmentations[index])
        
        if self.transform is not None:
            img = self.transform(img)
            
        return (img, y_label)