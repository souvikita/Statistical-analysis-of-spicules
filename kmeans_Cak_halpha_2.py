import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.decomposition import PCA
from astropy.io import fits
import sys
from helita.io import lp
import pickle
dpath = '/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/crispex/09:12:00/'
# Reading the data
hdrCa_im = lp.getheader(dpath+'crispex_3950_2017-05-25T09:12:00_scans=0-424_time-corrected_rotated2iris.fcube')
hdrCa_sp = lp.getheader(dpath+'crispex_3950_2017-05-25T09:12:00_scans=0-424_time-corrected_rotated2iris_sp.fcube')
hdrH_im =lp.getheader(dpath+'crispex_6563_08:05:00_aligned_3950_2017-05-25T09:12:00_scans=0-424_rotated2iris.icube')
hdrH_sp = lp.getheader(dpath+'crispex_6563_08:05:00_aligned_3950_2017-05-25T09:12:00_scans=0-424_rotated2iris_sp.icube')

dimCa_im = hdrCa_im[0]
dimCa_sp = hdrCa_sp[0]
dimH_im = hdrH_im[0]
dimH_sp = hdrH_sp[0]

cubeCa = lp.getdata(dpath+'crispex_3950_2017-05-25T09:12:00_scans=0-424_time-corrected_rotated2iris.fcube')
cubeH = lp.getdata(dpath+'crispex_6563_08:05:00_aligned_3950_2017-05-25T09:12:00_scans=0-424_rotated2iris.icube')

cubeCa = np.reshape(cubeCa,[dimCa_im[0],dimCa_im[1],dimCa_sp[1],dimCa_sp[0]])
cubeH = np.reshape(cubeH,[dimH_im[0],dimH_im[1],dimH_sp[1],dimH_sp[0]])

scan = ((154,155,164,165,167,168,194,195,196,197)) #--For training we just chose 10 scans spread over the FOV where we can see spicules clearly
m1a = np.mean(cubeCa[400:699,600:899,scan,41])
m2a = np.mean(cubeH[400:699,600:899,scan,31])

cubeCa = cubeCa[:,:,scan,:]/m1a
cubeH = cubeH[:,:,scan,:]/m2a

meanCa = np.mean(cubeCa[:,:,:,[0,40]],axis=3)
#mean2 = np.mean(cube2[:,:,[0,30]],axis=2)
meanH = cubeH[:,:,:,31]
  
m1b = np.median(cubeCa[:,:,:,[0,40]])
m2b = np.median(cubeH[:,:,:,[0,30]])

cubeCa = cubeCa[:,:,:,:]/meanCa[:,:,:,None]*m1b
cubeH = cubeH[:,:,:,:]/meanH[:,:,:,None]*m2b
###----Between lines 33 and 47, it involves the normalization of the spectral profiles. 
szz=cubeCa.shape
print(cubeH.shape)

cube_comb=np.zeros([szz[0],szz[1],szz[2],72]) # Combining the spectral cubes
cube_comb[:,:,:,:41]=cubeCa[:,:,:,:41]
cube_comb[:,:,:,41:]=cubeH[:,:,:,:31]

sz = np.shape(cube_comb)
x = np.reshape(cube_comb,(sz[0]*sz[1]*sz[2],sz[3])) # basically of the shape (n_samples, n_params)

print('Now starting k means with 50 clusters')

kmean = KMeans(n_clusters=50, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
                verbose=0, random_state=None, copy_x=True, n_jobs=-1, algorithm='auto') #I tried with n_jobs =-2 or with 1. It doesn't make much difference. 
kmean.fit(x)

clabels = kmean.labels_
cl = np.reshape(clabels,(sz[0],sz[1],sz[2]))
cc = kmean.cluster_centers_

pickle_out1 = open('kmeans_training.pickle','wb')
pickle_out2 = open('kmeans_labels.pickle','wb')
pickle.dump(cc,pickle_out1)
pickle.dump(cl,pickle_out2)
pickle.dump(kmean,open('kmean_model.pickle','wb'))
pickle_out1.close()
pickle_out2.close()



