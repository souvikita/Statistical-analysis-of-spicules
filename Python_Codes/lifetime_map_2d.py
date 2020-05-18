import matplotlib.pyplot as plt
from scipy.io import readsav
import numpy as np
import copy
from helita.io import lp
from astropy.io import fits
import sunpy.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import label, regionprops
from numpy import linspace
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

mean_abs_mag=np.mean(np.abs(cube_Mag),axis=2)
def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap
hmi_mag=plt.get_cmap('hmimag') #HMI magnetic field map from Sunpy
mycmap = transparent_cmap(plt.cm.jet)

#-----------FASTEST RBEs--------------------------
cluster_interest_blue_fast=[12,48] #Check COG.ipynb for this 
master_aux_cube_blue_fast=cluster_mask*0
for clust_index in range(len(cluster_interest_blue_fast)):
	master_aux_cube_blue_fast[np.where(cluster_mask == cluster_interest_blue_fast[clust_index])] = 1.
label_blu_fast = label(master_aux_cube_blue_fast,return_num=True, connectivity=2)
label_3d_fast=label_blu_fast[0]
data1 = np.load(dpath_npz+'lifetime_blu_fast.npz',allow_pickle=True)
new_data1 = data1['arr_0']
size=label_3d_fast.shape
ww_list = list(new_data1)#converting into list for ease of operation
lifetime_blue_fast = [k[0] for k in ww_list]
label_blue_fast = [k[1] for k in ww_list]
lifetime_blue_fast = np.array(lifetime_blue_fast)
label_blue_fast = np.array(label_blue_fast)
w1 = np.where(lifetime_blue_fast>=140.) # Forcing pixels with values >99% of the distribution of lifetime to be =0
#lifetime_blue_fast[w1[0]]=0.
life_median_blue_fast = np.zeros((size[0],size[1]))
for row in range(size[0]):
	for col in range(size[1]):
		if label_3d_fast[row,col,:].any()!=0:
			u = np.unique(label_3d_fast[row,col,:])
			u_imp = np.where(np.isin(u,label_blue_fast[w1[0]]) == False)
			life_median_blue_fast[row,col] = np.median(lifetime_blue_fast[u[u_imp][np.where(u[u_imp]!=0)[0]]-1])
		else:
			life_median_blue_fast[row,col] = 0.

#-----------INTERMEDIATE RBES------------------------
cluster_interest_blue_inter=[49,25] #Check COG.ipynb for this 
master_aux_cube_blue_inter=cluster_mask*0
for clust_index in range(len(cluster_interest_blue_inter)):
	master_aux_cube_blue_inter[np.where(cluster_mask == cluster_interest_blue_inter[clust_index])] = 1.
label_blu_inter = label(master_aux_cube_blue_inter,return_num=True, connectivity=2)
label_3d_inter=label_blu_inter[0]
label_blu_inter = label(master_aux_cube_blue_inter,return_num=True, connectivity=2)
label_3d_inter=label_blu_inter[0]
data2 = np.load(dpath_npz+'lifetime_blu_inter.npz',allow_pickle=True)
new_data2 = data2['arr_0']
ww_list = list(new_data2)#converting into list for ease of operation
lifetime_blue_inter = [k[0] for k in ww_list]
label_blue_inter = [k[1] for k in ww_list]
lifetime_blue_inter = np.array(lifetime_blue_inter)
label_blue_inter = np.array(label_blue_inter)
w2 = np.where(lifetime_blue_inter>=140) # Forcing pixels with values >99% of the distribution of lifetime to be =0
#lifetime_blue_inter[w2[0]]=0.
life_median_blue_inter = np.zeros((size[0],size[1]))
for row in range(size[0]):
    for col in range(size[1]):
        if label_3d_inter[row,col,:].any()!=0:
        	u = np.unique(label_3d_inter[row,col,:])
        	u_imp = np.where(np.isin(u,label_blue_inter[w2[0]]) == False)
        	life_median_blue_inter[row,col] = np.median(lifetime_blue_inter[u[u_imp][np.where(u[u_imp]!=0)[0]]-1])
        else:
            life_median_blue_inter[row,col] = 0.

#------------SLOW RBES--------------------------------
cluster_interest_blue_slow=[11,5] #Check COG.ipynb for this 
master_aux_cube_blue_slow=cluster_mask*0
for clust_index in range(len(cluster_interest_blue_slow)):
    master_aux_cube_blue_slow[np.where(cluster_mask == cluster_interest_blue_slow[clust_index])] = 1.
label_blu_slow = label(master_aux_cube_blue_slow,return_num=True, connectivity=2)
label_3d_slow=label_blu_slow[0]
data3 = np.load(dpath_npz+'lifetime_blu_slow.npz',allow_pickle=True)
new_data3 = data3['arr_0']
ww_list = list(new_data3)#converting into list for ease of operation
lifetime_blue_slow = [k[0] for k in ww_list]
label_blue_slow = [k[1] for k in ww_list]
lifetime_blue_slow = np.array(lifetime_blue_slow)
label_blue_slow = np.array(label_blue_slow)
w3 = np.where(lifetime_blue_slow>=140.)
#lifetime_blue_slow[w3[0]]=0.
life_median_blue_slow = np.zeros((size[0],size[1]))
for row in range(size[0]):
    for col in range(size[1]):
        if label_3d_slow[row,col,:].any()!=0:
            u = np.unique(label_3d_slow[row,col,:])
            u_imp = np.where(np.isin(u,label_blue_slow[w3[0]]) == False)
            life_median_blue_slow[row,col] = np.median(lifetime_blue_slow[u[u_imp][np.where(u[u_imp]!=0)[0]]-1])
        else:
            life_median_blue_slow[row,col] = 0.



#---------------FASTEST RRES-------------------------
cluster_interest_gt_5=[18,46] #Check COG.ipynb for this 
master_aux_cube_gt_5=cluster_mask*0
for clust_index in range(len(cluster_interest_gt_5)):
    master_aux_cube_gt_5[np.where(cluster_mask == cluster_interest_gt_5[clust_index])] = 1.

label_red_fast1= label(master_aux_cube_gt_5,return_num=True,connectivity=2)
label_3d_red_fast1 = label_red_fast1[0]
data4 = np.load(dpath_npz+'lifetime_red_fast.npz',allow_pickle=True)
new_data4 = data4['arr_0']
ww_list = list(new_data4)#converting into list for ease of operation
lifetime_red_fast = [k[0] for k in ww_list]
label_red_fast = [k[1] for k in ww_list]
lifetime_red_fast = np.array(lifetime_red_fast)
label_red_fast = np.array(label_red_fast)
w4 = np.where(lifetime_red_fast>=140.)
#lifetime_red_fast[w4[0]]=0.
life_median_red_fast = np.zeros((size[0],size[1]))
for row in range(size[0]):
    for col in range(size[1]):
        if label_3d_red_fast1[row,col,:].any()!=0:
            u = np.unique(label_3d_red_fast1[row,col,:])
            u_imp = np.where(np.isin(u,label_red_fast[w4[0]]) == False)
            life_median_red_fast[row,col] = np.median(lifetime_red_fast[u[u_imp][np.where(u[u_imp]!=0)[0]]-1])
        else:
            life_median_red_fast[row,col] = 0.

#---------------INTERMEDIATE RRES---------------------
cluster_interest_lt_5=[26,36] #Check COG.ipynb for this 
master_aux_cube_lt_5=cluster_mask*0
for clust_index in range(len(cluster_interest_lt_5)):
	master_aux_cube_lt_5[np.where(cluster_mask == cluster_interest_lt_5[clust_index])] = 1.

label_red_inter1= label(master_aux_cube_lt_5,return_num=True,connectivity=2)
label_3d_red_inter1 = label_red_inter1[0]
data5 = np.load(dpath_npz+'lifetime_red_inter.npz',allow_pickle=True)
new_data5 = data5['arr_0']
ww_list = list(new_data5)#converting into list for ease of operation
lifetime_red_inter = [k[0] for k in ww_list]
label_red_inter = [k[1] for k in ww_list]
lifetime_red_inter = np.array(lifetime_red_inter)
label_red_inter = np.array(label_red_inter)
w5 = np.where(lifetime_red_inter>=140.)
#lifetime_red_inter[w5[0]]=0.
life_median_red_inter = np.zeros((size[0],size[1]))
for row in range(size[0]):
    for col in range(size[1]):
        if label_3d_red_inter1[row,col,:].any()!=0:
            u = np.unique(label_3d_red_inter1[row,col,:])
            u_imp = np.where(np.isin(u,label_red_inter[w5[0]]) == False)
            life_median_red_inter[row,col] = np.median(lifetime_red_inter[u[u_imp][np.where(u[u_imp]!=0)[0]]-1])
        else:
            life_median_red_inter[row,col] = 0.

#-----------------SLOW RRES-----------------------------
cluster_interest_slow_rre=[8,16] #Check COG.ipynb for this 
master_aux_cube_slow_rre=cluster_mask*0
for clust_index in range(len(cluster_interest_slow_rre)):
	master_aux_cube_slow_rre[np.where(cluster_mask == cluster_interest_slow_rre[clust_index])] = 1.
label_red_slow1= label(master_aux_cube_slow_rre,return_num=True,connectivity=2)
label_3d_red_slow1 = label_red_slow1[0]
data6 = np.load(dpath_npz+'lifetime_red_slow.npz',allow_pickle=True)
new_data6 = data6['arr_0']
ww_list = list(new_data6)#converting into list for ease of operation
lifetime_red_slow = [k[0] for k in ww_list]
label_red_slow = [k[1] for k in ww_list]
lifetime_red_slow = np.array(lifetime_red_slow)
label_red_slow = np.array(label_red_slow)
w6 = np.where(lifetime_red_slow>=140.)
#lifetime_red_slow[w6[0]]=0.
life_median_red_slow = np.zeros((size[0],size[1]))
for row in range(size[0]):
    for col in range(size[1]):
        if label_3d_red_slow1[row,col,:].any()!=0:
            u = np.unique(label_3d_red_slow1[row,col,:])
            u_imp = np.where(np.isin(u,label_red_slow[w6[0]]) == False)
            life_median_red_slow[row,col] = np.median(lifetime_red_slow[u[u_imp][np.where(u[u_imp]!=0)[0]]-1])
        else:
            life_median_red_slow[row,col] = 0.

mycmap= transparent_cmap(plt.cm.jet)
mask = mean_abs_mag*0
mask[np.where(mean_abs_mag>=100)]=1.
print("-------------Entering the giant plotting routine------------------------------")

fig, axs =plt.subplots(2,3,figsize=(15,10),facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.1,wspace=0.05,left=0.05,right=0.95,top=0.93,bottom=0.07)
axs=axs.ravel()
axs[0].imshow(h_alpha[:,:,10,3],origin='lower',cmap='Greys_r',extent=[0,56,0,61])
im=axs[0].contourf(life_median_blue_fast,origin='lower',cmap=mycmap,levels=linspace(0, 150, 2500),extent=[0,56,0,61])
im1=axs[0].contour(mask,cmap='Greys',levels=110,extent=[0,56,0,61])
divider1 = make_axes_locatable(axs[0])
cax0 = divider1.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im,cax=cax0)
plt.setp(axs[0].get_xticklabels(), visible=False)
#plt.setp(axs[0].get_yticklabels(), visible=False)
axs[0].set_title(r'v$_{LOS}$ < -38.1 km s$^{-1}$',fontsize=14)
axs[0].set_ylabel('Y (Mm)')
axs[0].text(3.5,55,'(a)',color='white',fontsize=17)
#axs[0].text(36,4,r'N$_{tot}$ = '+format(label_blu_fast[1]),color='white',fontsize=13)

#axs[0].tick_params(axis='both', which='both', length=0)
axs[1].imshow(h_alpha[:,:,10,3],origin='lower',cmap='Greys_r',extent=[0,56,0,61])
im=axs[1].contourf(life_median_blue_inter,origin='lower',cmap=mycmap,levels=linspace(0, 150, 2500),extent=[0,56,0,61])
im1=axs[1].contour(mask,cmap='Greys',levels=110,extent=[0,56,0,61])
divider1 = make_axes_locatable(axs[1])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im,cax=cax1)
plt.setp(axs[1].get_xticklabels(), visible=False)
plt.setp(axs[1].get_yticklabels(), visible=False)
axs[1].set_title(r' -38.1 km s$^{-1}$< v$_{LOS}$ < -29.8 km s$^{-1}$',fontsize=14)
axs[1].text(3.5,55,'(b)',color='white',fontsize=17)
#axs[1].text(36,4,r'N$_{tot}$ = '+format(label_blu_inter[1]),color='white',fontsize=13)

axs[2].imshow(h_alpha[:,:,10,3],origin='lower',cmap='Greys_r',extent=[0,56,0,61])
im=axs[2].contourf(life_median_blue_slow,origin='lower',cmap=mycmap,levels=linspace(0, 150, 2500),extent=[0,56,0,61])
im1=axs[2].contour(mask,cmap='Greys',levels=110,extent=[0,56,0,61])
plt.setp(axs[2].get_xticklabels(), visible=False)
plt.setp(axs[2].get_yticklabels(), visible=False)
divider1 = make_axes_locatable(axs[2])
cax2 = divider1.append_axes("right", size="5%", pad=0.05)
cbar2=plt.colorbar(im,cax=cax2)
cbar2.set_label('Lifetime (s)',fontsize=13)
axs[2].set_title(r'-29.8 km s$^{-1}$< v$_{LOS}$ < -25.1 km s$^{-1}$',fontsize=14)
axs[2].text(3.5,55,'(c)',color='white',fontsize=17)
#axs[2].text(36,4,r'N$_{tot}$ = '+format(label_blu_slow[1]),color='white',fontsize=13)

axs[3].imshow(h_alpha[:,:,10,3],origin='lower',cmap='Greys_r',extent=[0,56,0,61])
im=axs[3].contourf(life_median_red_fast,origin='lower',cmap=mycmap,levels=linspace(0, 150, 2500),extent=[0,56,0,61])
im1=axs[3].contour(mask,cmap='Greys',levels=110,extent=[0,56,0,61])
divider1 = make_axes_locatable(axs[3])
cax3 = divider1.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im,cax=cax3)
axs[3].set_title(r'v$_{LOS}$ > 40 km s$^{-1}$',fontsize=14)
axs[3].set_ylabel('Y (Mm)')
axs[3].set_xlabel('X (Mm)')
axs[3].text(3.5,55,'(d)',color='white',fontsize=17)
#axs[3].text(36,4,r'N$_{tot}$ = '+format(label_red_fast1[1]),color='white',fontsize=13)

axs[4].imshow(h_alpha[:,:,10,3],origin='lower',cmap='Greys_r',extent=[0,56,0,61])
im=axs[4].contourf(life_median_red_inter,origin='lower',cmap=mycmap,levels=linspace(0,150, 2500),extent=[0,56,0,61])
im1=axs[4].contour(mask,cmap='Greys',levels=110,extent=[0,56,0,61])
plt.setp(axs[4].get_yticklabels(), visible=False)
divider1 = make_axes_locatable(axs[4])
cax4 = divider1.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im,cax=cax4)
axs[4].set_xlabel('X (Mm)')
axs[4].set_title(r'29.8 km s$^{-1}$< v$_{LOS}$ < 40 km s$^{-1}$',fontsize=14)
axs[4].text(3.5,55,'(e)',color='white',fontsize=17)
#axs[4].text(36,4,r'N$_{tot}$ = '+format(label_red_inter1[1]),color='white',fontsize=13)

axs[5].imshow(h_alpha[:,:,10,3],origin='lower',cmap='Greys_r',extent=[0,56,0,61])
im=axs[5].contourf(life_median_red_slow,origin='lower',cmap=mycmap,levels=linspace(0, 150, 2500),extent=[0,56,0,61])
im1=axs[5].contour(mask,cmap='Greys',levels=110,extent=[0,56,0,61])
plt.setp(axs[5].get_yticklabels(), visible=False)
divider1 = make_axes_locatable(axs[5])
cax5 = divider1.append_axes("right", size="5%", pad=0.05)
cbar5=plt.colorbar(im,cax=cax5)
cbar5.set_label('Lifetime (s)',fontsize=13)
axs[5].set_title(r'25.1 km s$^{-1}$< v$_{LOS}$ < 29.8 km s$^{-1}$',fontsize=14)
axs[5].set_xlabel('X (Mm)')
axs[5].text(3.5,55,'(f)',color='white',fontsize=17)
#axs[5].text(36,4,r'N$_{tot}$ = '+format(label_red_slow1[1]),color='white',fontsize=13)
#plt.subplot_adjust()
plt.savefig('/mn/stornext/u3/souvikb/paper4_images/2d_events_spicules_lifetime_less_140_v3.png')
