import matplotlib.pyplot as plt
from scipy.io import readsav
import numpy as np
import copy
from helita.io import lp
from astropy.io import fits
import sunpy.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, binary_closing
from skimage.morphology import diamond,ball
import h5py
from scipy.spatial import distance
from multiprocessing import Pool

def area_weighted_cog(x,y,area):
    '''Computing COG weighted by area
    
    Inputs:
    x: an array of x coordinates of 2D centroids.
    y: an array of y coordinates of 2D centroids.
    area: area for each of the 2D labels.
    ####--all three must be of the same length'''
    if len(x) and len(y) !=0:
        x_cog = np.zeros((len(x)))
        y_cog = x_cog*0
        for l in range(len(x)):
            x_cog[l] = x[l]*area[l]
            y_cog[l] = y[l]*area[l]
        X_COG = np.sum(x_cog)/np.sum(area)
        Y_COG = np.sum(y_cog)/np.sum(area)
    else:
        X_COG = np.nan
        Y_COG = np.nan
    return X_COG, Y_COG
    
dpath = '/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/crispex/09:12:00/'
dpath_area = '/mn/stornext/d9/souvikb/K_means_results/savefiles/'
dpath_cluster_fits= '/mn/stornext/d9/souvikb/K_means_results/'
dpath_timesteps ='/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/calib_tseries/'
dpath_npz = '/mn/stornext/d9/souvikb/K_means_results/'

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

#Labelling the clusters in 3D by considering all the structures. 
cluster_interest_red = [18,46,26,36,8,16] # considering the shadows as well. 
master_aux_cube_red=cluster_mask*0
for clust_index in range(len(cluster_interest_red)):
    master_aux_cube_red[np.where(cluster_mask == cluster_interest_red[clust_index])] = 1.

selem = diamond(1)
morph_processed_red = master_aux_cube_red*0.
for scan in range(425):
    morph_processed_red[:,:,scan] = binary_closing(binary_opening(master_aux_cube_red[:,:,scan],selem),selem) # Morph_open followed by Morph_closing operation to get rid of the salt

#labelling the 3d cube
labeled_3d_red = label(morph_processed_red,return_num=True, connectivity=2)

labels_in_roi =[]
for region in regionprops(labeled_3d_red[0]):
    labels_in_roi.append(region.label)

label_2d_cube=[]
count =0
new_count = np.zeros((425)) # used to store those label numbers which are treeated as background in sequential labelling

#Yes, similar analyses, but the FOV is limited
for time in range(425): 
	#reference_mask= closing(master_aux_cube[:,:,time],selem=np.ones((3,3),np.uint8)) # doing a moprh closing on the BW mask
	reference_mask = morph_processed_red[:,:,time]
	label_2d = label(reference_mask,connectivity=2,return_num=True)
	label_numbers = label_2d[1]
	if time ==0:
		label_2d_cube.append(label_2d[0])
	elif time>0:
		label_2d_cube.append(label_2d[0]+count) # serial labels.
	count = count +label_numbers # keeprs track of the labels
	new_count[time] = count
    #print(count,label_numbers)

cube_with_2d_label = np.array(label_2d_cube) # converts the list to a numpy array suited to our analysis. 
label_2d_proper_dim=np.swapaxes(np.swapaxes(cube_with_2d_label,0,2),0,1) # changing the dimension to the dimension of labelled_unmorphed[0]

####-----------computing the parameters for each of the labels in 2D and storing them-------
centroid1=[]
label_2d=[]
area_2d=[]

print("Now computing the centroids from the 2D labelled 3D cube")
for time in range(425):
    for region in regionprops(label_2d_proper_dim[:,:,time]):
        if region.label in new_count: # checking if the labels are from the background labels. If so, then  
            continue                  # skip this and return the control to the beginning to the inner for loop         
        cog = region.centroid
        centroid1.append(cog)
        lab_no=region.label
        label_2d.append(lab_no)
        area_roi = region.area
        area_2d.append(area_roi)

centroid1 = np.array(centroid1)
label_2d = np.array(label_2d)
area_2d = np.array(area_2d)

def computing_pos_labels(labels_in_roi):
    '''This function is designed to
    compute the position of COG per label
    over the whole time of its existence'''
    time_indices = np.where(labeled_3d_red[0] == labels_in_roi)
    unique_time_indices = np.unique(np.sort(time_indices[2]))
    position_per_time = []
    for time in range(len(unique_time_indices)):
        spatial_index=np.where(labeled_3d_red[0][:,:,unique_time_indices[time]] == labels_in_roi)
        unique_spatial_index = np.unique(label_2d_proper_dim[spatial_index[0],spatial_index[1],unique_time_indices[time]])
        index1 = np.where(np.isin(unique_spatial_index,new_count)== False) # index1 is making sure unique_spatial_index is not in new_count. 
        if len(index1[0])==0:
            position = (np.nan,np.nan)
            position_per_time.append(position)
        else:
            index_2Dlab =np.where(np.isin(label_2d,unique_spatial_index[index1])==True)
            area_index_2Dlab = area_2d[index_2Dlab[0]]
            position = area_weighted_cog(centroid1[index_2Dlab[0],0],centroid1[index_2Dlab[0],1],area_index_2Dlab)
            position_per_time.append(position)
    return position_per_time

print('*****Going Parallel******')
pool=Pool(120)
result = pool.map(computing_pos_labels, labels_in_roi)
data = np.array(result)
pool.close()
pool.join()
np.savez(dpath_cluster_fits+'rre_COG_positions_per_label',data)
