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
dpath = '/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/crispex/09:12:00/'
dpath_area = '/mn/stornext/d9/souvikb/K_means_results/savefiles/'
dpath_cluster_fits= '/mn/stornext/d9/souvikb/K_means_results/'
dpath_timesteps ='/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/calib_tseries/'

#---Reading the original cluster mask-------
cluster_mask = fits.getdata(dpath_cluster_fits+'clusters_mask.fits',ext=0)
cluster_interest_blu =[12,49,25,48] ## Might wanna add some 
master_aux_cube=cluster_mask*0.
skel = cluster_mask*0.
for clust_index in range(len(cluster_interest_blu)):
    master_aux_cube[np.where(cluster_mask == cluster_interest_blu[clust_index])] = 1. #This selects only the clusters of interest and creates a 3D cube with only clusters of interest

label_unmorphed = label(master_aux_cube,return_num=True, connectivity=2) #Labelling the 3D cube in 3D
labels_of_interest=[]
for region in regionprops(label_unmorphed[0]):
	lab = region.label
	labels_of_interest.append(lab) 
#------- Creating a 3D cube where we label each 2D image per time step and then store it as a  3D cube with serial order labels
label_2d_cube=[]
count =0
new_count = np.zeros((425)) # used to store those label numbers which are treeated as background in sequential labelling

print("entering the 2D labelling loop")
for time in range(425): 
	reference_mask= closing(master_aux_cube[:,:,time],selem=np.ones((3,3),np.uint8)) # doing a moprh closing on the BW mask
	#reference_mask = master_aux_cube[:,:,time]
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
        if region.label in new_count: # checking if the labels are from the background labels. If so, then do nothing. 
            pass #do nothing
        else:
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

def compute_stats(label_number_3d):
    """This is a method to compute the 
    statistics of RBEs for a given label 
    number obtained from labelled_unmorphed[1]
    """
    label_number_3d = int(label_number_3d)
    index = np.where(label_unmorphed[0]==label_number_3d)
    xx = np.unique(label_2d_proper_dim[index])
    if xx.size == 0 or np.isin(xx,new_count).all() == True: #This condition is important because the morph_closing operation in line 34 can also remove some features thereby causing an empty 
                                         ##array in the 3D labelled_unmorphed[0] cube. Hence checking if they are a part 
        pass
    else:
        max_length = np.max(length_2d[np.isin(label_2d,xx)])
        lab2d_max = xx[np.argmax(length_2d[np.isin(label_2d,xx)])]
        yy = np.where(label_2d==lab2d_max)
        max_area = area_2d[yy]
        max_eccen =eccen_2d[yy]
        return max_length, max_area, max_eccen

print("****Going Parallel********")
#labels_of_interest = list(range(1,32842)) #label number 31336 has raised this ValueError: zero-size array to reduction operation maximum. 
#labels_of_interest = list(range(31300,32842))
#embed()
pool=Pool(120)
#with Pool as pool:
result = pool.map(compute_stats, labels_of_interest)
data = np.array(result)
pool.close()
pool.join()
np.savez(dpath_cluster_fits+'rbe_stats_parallel',data)