import matplotlib.pyplot as plt
from scipy.io import readsav
import numpy as np
import multiprocessing 
from multiprocessing import Pool
from helita.io import lp
from astropy.io import fits
from skimage.measure import label, regionprops
from skimage.morphology import opening,closing
import h5py
from IPython import embed
from skimage.morphology import binary_opening, binary_closing
from skimage.morphology import diamond

dpath = '/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/crispex/09:12:00/'
dpath_area = '/mn/stornext/d9/souvikb/K_means_results/savefiles/'
dpath_cluster_fits= '/mn/stornext/d9/souvikb/K_means_results/'
dpath_timesteps ='/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/calib_tseries/'

#---Reading the original cluster mask-------
cluster_mask = fits.getdata(dpath_cluster_fits+'clusters_mask.fits',ext=0)
cluster_interest_red = [18,46,26,36,8,16] # considering the shadows as well. 
master_aux_cube_red=cluster_mask*0
for clust_index in range(len(cluster_interest_red)):
    master_aux_cube_red[np.where(cluster_mask == cluster_interest_red[clust_index])] = 1.
# Morphological operations as discussed so as to avoid the 1 pixel like connectivities.
# So, we decided to perform morph_open followed by morph_close
selem = diamond(1)
morph_processed_red = master_aux_cube_red*0.
for scan in range(425):
	morph_processed_red[:,:,scan] = binary_closing(binary_opening(master_aux_cube_red[:,:,scan],selem),selem)

labeled_3d_red = label(morph_processed_red,return_num=True, connectivity=2)

labels_of_interest=[]
for region in regionprops(labeled_3d_red[0]):
	lab = region.label
	labels_of_interest.append(lab)

#------- Creating a 3D cube where we label each 2D image per time step and then store it as a  3D cube with serial order labels
label_2d_cube=[]
count =0
new_count = np.zeros((425)) # used to store those label numbers which are treeated as background in sequential labelling

print("entering the 2D labelling loop")
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
length_2d=[]
area_2d=[]
eccen_2d=[]
label_2d=[]

print("Now computing the maximum lengths from the 2D labelled 3D cube")
for time in range(425):
    for region in regionprops(label_2d_proper_dim[:,:,time]):
        if region.label in new_count: # checking if the labels are from the background labels. If so, then  
            continue                  # skip this and return the control to the beginning to the inner for loop         
        lab_no = region.label 
        length = region.major_axis_length*0.037*0.722 #length in Mm
        area = region.area*0.037*0.037*0.722*0.722 #area in Mm^2
        eccen = region.eccentricity # ecentricity of an ellipse
        length_2d.append(length)
        area_2d.append(area)
        eccen_2d.append(eccen)
        label_2d.append(lab_no)


#--------Converting the lists to numpy arrays-------
label_2d = np.array(label_2d)
length_2d = np.array(length_2d)
area_2d = np.array(area_2d)
eccen_2d = np.array(eccen_2d)

def compute_length_stats(label_number_3d):
    index = np.where(labeled_3d_red[0]==label_number_3d)
    xx = np.unique(label_2d_proper_dim[index[0],index[1],index[2]])
    if xx.size != 0:
        index1 = np.where(np.isin(xx,new_count)== False)
        if len(index1[0])!=0: # this will prevent an empty list/tuple from entering the stats
            max_length = np.max(length_2d[np.isin(label_2d,xx[index1])])
            lab2d_max = xx[np.argmax(length_2d[np.isin(label_2d,xx[index1])])]
            yy = np.where(label_2d==lab2d_max)
            max_area = area_2d[yy]
            max_eccen = eccen_2d[yy]
        else:
            max_length =np.nan
            max_area = np.nan
            max_eccen = np.nan
    else:
        max_length =np.nan
        max_area = np.nan
        max_eccen =np.nan
    return max_length, max_area, max_eccen
###------End of the function--------

print("****Going Parallel********")
pool=Pool(120)
result = pool.map(compute_length_stats, labels_of_interest)
data = np.array(result)
pool.close()
pool.join()
np.savez(dpath_cluster_fits+'rbe_stats_parallel_improved_shadow_RREs',data)

