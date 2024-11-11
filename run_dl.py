from DL import *
import os

os.makedirs('./dl_out', exist_ok=True) 

# Preprocess image
img_full = skimage.io.imread('../images/P23-8-Slices-cropped-bleach-corrected.tif')
img_full = (img_full >> 8).astype('uint8')

us = UnsupervisedSegmentation(img_full=img_full, # Image data
                              n_iterations=25, # Number of iterations/epochs for training
                              lr=0.2, # Learning rate for training
                              n_conv=2) # Number of convolutional layers in CNN
us.prepare_slice(0)
us.train()

for i in range(us.img_full.shape[0]):
    us.prepare_slice(i)
    final_segmentation = us.infer()
    plt.imsave(f'./dl_out/{i:03}.png', final_segmentation, cmap='gray')