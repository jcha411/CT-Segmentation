import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage.filters import threshold_multiotsu
import os

# Adjustable parameters
img_filepath = './images/P23-8-Slices-cropped-bleach-corrected.tif' # Path to image
n_classes = 5 # Number of classes for segmentation

os.makedirs('./otsu_out', exist_ok=True) 

img = skimage.io.imread(img_filepath)
img = (img >> 8).astype('uint8')

thresholds = threshold_multiotsu(img[0, :, :], classes=n_classes)
regions = np.digitize(img, bins=thresholds)

for i in range(img.shape[0]):
    plt.imsave(f'./otsu_out/{i:03}.png', regions[i], cmap='gray')

