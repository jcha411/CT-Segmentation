from Kriging import *
import time
import os

os.makedirs('./dk_out', exist_ok=True) 

# Preprocess image
img_full = skimage.io.imread('../images/P23-8-Slices-cropped-bleach-corrected.tif')
img_full = (img_full >> 8).astype('uint8')

dk = DeepKriging(img_full=img_full, # Image data
                 n_clusters=4, # Number of clusters
                 max_iter_uncertain=10, # Max num of iterations to find uncertain region(s)
                 zeta0=1.96, # Uncertainty factor
                 p_uncertain=0.2, # Desired uncertainty region probability
                 n_epochs=5, # Number of epochs for training neural network
                 batch_size=32) # Batch size for training data

st = time.time()
dk.train()
print(f'Completed training in {time.time()-st:.2f} seconds.')

for i in range(dk.img_full.shape[0]):
    st = time.time()
    out = dk.infer(i)
    print(f'Completed slice {i} in {time.time()-st:.2f} seconds.')
    plt.imsave(f'./dk_out/{i:03}.png', out, cmap='gray')