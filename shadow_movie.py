import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.io import readsav
from astropy.io import fits
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,mark_inset

dpath_clusters ='/mn/stornext/d9/souvikb/K_means_results/'
dpath = '/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/crispex/09:12:00/'
cluster_mask = fits.getdata(dpath_clusters+'clusters_mask.fits',ext=0)

#now make a movie for RPs in the same way as Fig 1. c) of your paper II. 
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


rbe = np.array([5,49,11, 25, 12, 48,]) #Clusters of interest including the shadow. Arranged in increased order of shift from COG of cluster 40.
rre = np.array([16, 39, 8, 26, 46, 18,])
rbe_cn = np.zeros(rbe.size,dtype='int32')
rre_cn = np.zeros(rre.size,dtype='int32')

def image_plot(image):
	"""Just for the plot of discreet colorbars"""

	rbeim = np.full(image.shape,-1)
	rreim = np.full(image.shape,-1)

	for ii in range(rbe.size):
		ss=image[:,:] == rbe[ii]
		rbeim[ss] = ii
		rbe_cn[ii]=ii

	for ii in range(rre.size):
		ss = image[:,:]== rre[ii]
		rreim[ss]=ii+rbe.size
		rre_cn[ii]= ii+rbe.size

	cmap = plt.get_cmap('Blues',rbe.size)
	cmap2 = truncate_colormap(cmap, 0.5, 1.0, rbe.size)
	cmap = plt.get_cmap('Reds',rre.size)
	cmap3 = truncate_colormap(cmap, 0.5, 1.0, rre.size)

	mrbeim = np.ma.masked_where(rbeim < 0, rbeim)
	mrreim = np.ma.masked_where(rreim < 0, rreim)
	return mrbeim, mrreim, rbe_cn, rre_cn, cmap2, cmap3

for time in range(425):

	image = cluster_mask[:,:,time]
	processed_images = image_plot(image)
	fig, ax = plt.subplots(figsize=(10, 10))
	im1=ax.imshow(processed_images[0],origin='lower',cmap=processed_images[4])
	im2=ax.imshow(processed_images[1],origin='lower',cmap=processed_images[5])
	axins_cb2 = inset_axes(ax,
	                       width="5%",  # width = 5% of parent_bbox width
	                       height="48%",  # height : 50%
	                       loc='lower left',
	                       bbox_to_anchor=(1.05, 0., 1, 1),
	                       bbox_transform=ax.transAxes,
	                       borderpad=0,
	                       )

	axins_cb3 = inset_axes(ax,
	                       width="5%",  # width = 5% of parent_bbox width
	                       height="48%",  # height : 50%
	                       loc='upper left',
	                       bbox_to_anchor=(1.05, 0., 1, 1),
	                       bbox_transform=ax.transAxes,
	                       borderpad=0,
	                       )

	plt.colorbar(im1,cax=axins_cb2,ticks = np.arange(np.min(processed_images[2]),np.max(processed_images[2])+1),orientation="vertical")
	axins_cb2.xaxis.set_ticks_position("default")

	#cax3 = divider.append_axes("right", size="3%", pad=0.04)
	plt.colorbar(im2,cax=axins_cb3,ticks = np.arange(np.min(processed_images[3]),np.max(processed_images[3])+1),orientation="vertical")
	axins_cb3.xaxis.set_ticks_position("default")
	ax.set_title('Frame:'+str(time))
	plt.savefig('/mn/stornext/u3/souvikb/paper4_images/Doppler_movie/RBE_RRE_shadow'+str(time)+'.png',dpi=600,orientation='landscape')
	#plt.show()
