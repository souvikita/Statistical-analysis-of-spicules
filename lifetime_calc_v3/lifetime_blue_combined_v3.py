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
	indices = np.where(labeled_3d[0] == label_number)
	lifetime = len(np.unique(indices[2]))*13.6
	label_num = int(label_number)
	return lifetime, label_num

dpath_cluster_fits= '/mn/stornext/d9/souvikb/K_means_results/'

cluster_mask = fits.getdata(dpath_cluster_fits+'clusters_mask.fits',ext=0)# Reading the Cluster mask time series
size = cluster_mask.shape

#Labelling the clusters in 3D by considering all the structures. 
cluster_interest_blue = [12,48,49,25,11,5] # considering the shadows as well. 
master_aux_cube_blue=cluster_mask*0
for clust_index in range(len(cluster_interest_blue)):
    master_aux_cube_blue[np.where(cluster_mask == cluster_interest_blue[clust_index])] = 1.

# Morphological operations as discussed so as to avoid the 1 pixel like connectivities.
# So, we decided to perform morph_open followed by morph_close
selem = diamond(1)
morph_processed_blue = master_aux_cube_blue*0.
for scan in range(425):
    morph_processed_blue[:,:,scan] = binary_closing(binary_opening(master_aux_cube_blue[:,:,scan],selem),selem) # Morph_open followed by Morph_closing operation to get rid of the salt

#labelling the 3d cube morphed processed cube
labeled_3d = label(morph_processed_blue,return_num=True, connectivity=2)

label_nos_blue = []
for region in regionprops(labeled_3d[0]):
    label_nos_blue.append(region.label)

print("****Entering power parallel mode****")

labels_of_interest = label_nos_blue
pool = Pool(120)
result = pool.map(compute_lifetime_parallel, labels_of_interest)
data = np.array(result)
pool.close()
pool.join()
np.savez(dpath_cluster_fits+'lifetime_blu_combined_v3', data)
