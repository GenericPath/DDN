import os, pickle
from PIL import Image
import numpy as np

full_path = './data/simple01/16-16/dataset'
num_images = 500

if os.path.isfile(full_path):
    with open (full_path, 'rb') as fp:
        output = pickle.load(fp)
        inputs = output[0][:num_images] # images
        outputs = output[1][:num_images] # segmentations

print(f'{len(inputs)} {type(inputs[0])} {len(outputs)} {type(outputs[0])}')