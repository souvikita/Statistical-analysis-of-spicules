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

dpath = '/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/crispex/09:12:00/'
dpath_area = '/mn/stornext/d9/souvikb/K_means_results/savefiles/'
dpath_cluster_fits= '/mn/stornext/d9/souvikb/K_means_results/'
dpath_timesteps ='/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/calib_tseries/'

cluster_mask = fits.getdata(dpath_cluster_fits+'clusters_mask.fits',ext=0)# Reading the Cluster mask time series
hdrH_im =lp.getheader(dpath+'crispex_6563_08:05:00_aligned_3950_2017-05-25T09:12:00_scans=0-424_rotated2iris.icube')
hdrH_sp = lp.getheader(dpath+'crispex_6563_08:05:00_aligned_3950_2017-05-25T09:12:00_scans=0-424_rotated2iris_sp.icube')
dimH_im = hdrH_im[0]
dimH_sp = hdrH_sp[0]
cubeH = lp.getdata(dpath+'crispex_6563_08:05:00_aligned_3950_2017-05-25T09:12:00_scans=0-424_rotated2iris.icube')
cubeH = np.reshape(cubeH,[dimH_im[0],dimH_im[1],dimH_sp[1],dimH_sp[0]])
time_sst = readsav(dpath_timesteps+'tseries_3950_2017-05-25T09:12:00_scans=0-424_calib.sav')
time_sst=time_sst['time']
hdr_Mag = lp.getheader(dpath+'Blos_6302_08:05:00_aligned_3950_2017-05-25T09:12:00_scans=0-424_rotated2iris.icube')
dim_Mag = hdr_Mag[0]
cube_Mag = lp.getdata(dpath+'Blos_6302_08:05:00_aligned_3950_2017-05-25T09:12:00_scans=0-424_rotated2iris.icube')
cube_Mag = np.reshape(cube_Mag,[dim_Mag[0],dim_Mag[1],dim_Mag[2]])
cube_Mag=np.swapaxes(cube_Mag,0,1)
h_alpha=np.swapaxes(cubeH,0,1)

#mean_abs_mag=np.mean(np.abs(cube_Mag),axis=2)

# cluster_interest_blue_fast=[12,48] #Check COG.ipynb for this 
# master_aux_cube_blue_fast=cluster_mask*0
# for clust_index in range(len(cluster_interest_blue_fast)):
# 	master_aux_cube_blue_fast[np.where(cluster_mask == cluster_interest_blue_fast[clust_index])] = 1.

cluster_interest_blue = [12,48,49,25,11,5] # considering the shadows as well. 
master_aux_cube_blue=cluster_mask*0
for clust_index in range(len(cluster_interest_blue)):
    master_aux_cube_blue[np.where(cluster_mask == cluster_interest_blue[clust_index])] = 1.

# Morph_open followed by a morph_close
selem = diamond(1)
morph_processed_blue = master_aux_cube_blue*0.
for scan in range(425):
    morph_processed_blue[:,:,scan] = binary_closing(binary_opening(master_aux_cube_blue[:,:,scan],selem),selem) # Morph_open followed by Morph_closing operation to get rid of the salt
labeled_3d = label(morph_processed_blue,return_num=True, connectivity=2)

cluster_interest_blue_fast=[12,48] #Check COG.ipynb for this 
master_aux_cube_blue_fast=cluster_mask*0
for clust_index in range(len(cluster_interest_blue_fast)):
    master_aux_cube_blue_fast[np.where(cluster_mask == cluster_interest_blue_fast[clust_index])] = 1.

size = master_aux_cube_blue_fast.shape
label_3d_blue_fast =np.zeros((size[0],size[1],size[2]),dtype='int')

for scan in range(425):
    label_3d_blue_fast[:,:,scan]=master_aux_cube_blue_fast[:,:,scan]*labeled_3d[0][:,:,scan]
 
label_nos_blue_fast = []
for region in regionprops(label_3d_blue_fast):
    label_nos_blue_fast.append(region.label)
    
def compute_lifetime_parallel(label_number):
	"""This is to compute the lifetime
	for different labels per cluster"""
	indices = np.where(label_3d_blue_fast == label_number)
	lifetime = len(np.unique(indices[2]))*13.6
	label_num = int(label_number)
	return lifetime, label_num

print("****Entering power parallel mode****")


labels_of_interest = label_nos_blue_fast

pool = Pool(120)
result = pool.map(compute_lifetime_parallel, labels_of_interest)
data = np.array(result)
pool.close()
pool.join()
np.savez(dpath_cluster_fits+'lifetime_blu_fast_uniq', data)

