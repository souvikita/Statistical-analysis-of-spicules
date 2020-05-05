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
	indices = np.where(label_3d_red_slow == label_number)
	lifetime = len(np.unique(indices[2]))*13.6
	label_num = int(label_number)
	return lifetime, label_num


dpath = '/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/crispex/09:12:00/'
dpath_area = '/mn/stornext/d9/souvikb/K_means_results/savefiles/'
dpath_cluster_fits= '/mn/stornext/d9/souvikb/K_means_results/'
dpath_timesteps ='/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/calib_tseries/'

cluster_mask = fits.getdata(dpath_cluster_fits+'clusters_mask.fits',ext=0)# Reading the Cluster mask time series
size = cluster_mask.shape

labeled_cube = h5py.File(dpath_cluster_fits+'label_red_slow.hdf5','r')
label_3d_red_slow = np.array(labeled_cube['label_3d_red_slow'])

label_nos_red_slow = []
for region in regionprops(label_3d_red_slow):
    label_nos_red_slow.append(region.label)

print("****Entering power parallel mode****")

labels_of_interest = label_nos_red_slow
pool = Pool(120)
result = pool.map(compute_lifetime_parallel, labels_of_interest)
data = np.array(result)
pool.close()
pool.join()
np.savez(dpath_cluster_fits+'lifetime_red_slow_uniq', data)

