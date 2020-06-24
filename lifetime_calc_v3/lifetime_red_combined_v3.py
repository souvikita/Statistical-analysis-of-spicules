import matplotlib.pyplot as plt
from scipy.io import readsav
import numpy as np
import copy
from helita.io import lp
from astropy.io import fits
import sunpy.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import label, regionprops
import multiprocessing 
from multiprocessing import Pool
from skimage.morphology import binary_opening, binary_closing
from skimage.morphology import diamond
import h5py

def compute_lifetime_parallel(label_number):
	"""This is to compute the lifetime
	for different labels per cluster"""
	indices = np.where(labeled_3d_red[0] == label_number)
	lifetime = len(np.unique(indices[2]))*13.6
	label_num = int(label_number)
	return lifetime, label_num


dpath_cluster_fits= '/mn/stornext/d9/souvikb/K_means_results/'

cluster_mask = fits.getdata(dpath_cluster_fits+'clusters_mask.fits',ext=0)# Reading the Cluster mask time series
size = cluster_mask.shape

#Labelling the clusters in 3D by considering all the structures. 
cluster_interest_red = [18,46,26,36,8,16] # considering the shadows as well. 
master_aux_cube_red=cluster_mask*0
for clust_index in range(len(cluster_interest_red)):
    master_aux_cube_red[np.where(cluster_mask == cluster_interest_red[clust_index])] = 1.

selem = diamond(1)
morph_processed_red = master_aux_cube_red*0.
for scan in range(425):
    morph_processed_red[:,:,scan] = binary_closing(binary_opening(master_aux_cube_red[:,:,scan],selem),selem) # Morph_open followed by Morph_closing operation to get rid of the salt

labeled_3d_red = label(morph_processed_red,return_num=True, connectivity=2)


label_nos_red = []
for region in regionprops(labeled_3d_red[0]):
    label_nos_red.append(region.label)

print("****Entering power parallel mode****")

labels_of_interest = label_nos_red
pool = Pool(40)
result = pool.map(compute_lifetime_parallel, labels_of_interest)
data = np.array(result)
pool.close()
pool.join()
np.savez(dpath_cluster_fits+'lifetime_red_combined_v3', data)
