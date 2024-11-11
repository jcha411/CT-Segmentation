from Kriging import *
import time
import os
import skimage

os.makedirs('./tdmrk_out', exist_ok=True) 

# Preprocess image
img_full = skimage.io.imread('../data/P23-8-Slices-cropped-bleach-corrected.tif')
img_full = (img_full >> 8).astype('uint8')

tdmrk = TwoDimMultiRegionKriging(img_full=img_full, # Image data
                                 n_clusters=5, # Number of clusters
                                 max_iter_uncertain=10, # Max num of iterations to find uncertain region(s)
                                 zeta0=1.96, # Uncertainty factor
                                 p_uncertain=0.3, # Desired uncertainty region probability
                                 vario_distance=30, # Max distance on semivariogram
                                 krig_win_radius=3) # Kriging window radius

st = time.time()
tdmrk.train()
print(f'Completed training in {time.time()-st:.2f} seconds.')

for i in range(tdmrk.img_full.shape[0]):
    st = time.time()
    out = tdmrk.infer(i)
    print(f'Completed slice {i} in {time.time()-st:.2f} seconds.')
    plt.imsave(f'./tdmrk_out/{i:03}.png', out, cmap='gray')