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
from scipy.io.idl import readsav
import matplotlib

dpath_kmean_bose = '/mn/stornext/d9/souvikb/K_means_results/'
data = pickle.load(open(dpath_kmean_bose+'kmeans_training.pickle','rb')) #(50,72)
dpath = '/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/crispex/09:12:00/'

wave_Ca =readsav(dpath+'spectfile.3950.idlsave')
wave_H= readsav(dpath+'spectfile.6563.idlsave')
wave_H=wave_H['spect_pos']
wave_Ca=wave_Ca['spect_pos']

line_core = 3933.67
nlam = len(wave_Ca)
rows = 10
cols = 5
in_text = 10

matplotlib.rc('xtick', labelsize=8)     
matplotlib.rc('ytick', labelsize=8)
grps = [12,48,49,25,11,5,
        18,46,26,36,8,16]
clrs = ['blue','blue','blue','blue','blue','blue',
       'red','red','red','red','red','red']
clr_dic = dict(zip(grps, clrs))

fig, axs = plt.subplots(rows, cols, figsize=(10,20))
fig.subplots_adjust(hspace = 0.3, wspace=0.3,left=0.06,right=0.95,top=0.95,bottom=0.05)
ax=axs.ravel()
for k in range(50):
    ax[k].plot(wave_Ca[:41], data[k,:41], color='Grey', linewidth=2, linestyle='--',)
    ax[k].plot(wave_Ca[:41], data[k,:41], color=clr_dic.get(k,'Grey'), linewidth=2)
    ax[k].plot(wave_Ca[:41], data[40,:41],color='Grey',linestyle='dashed')
    ax[k].axvline(x=line_core,color='black',linewidth = 1,linestyle='dashed')
    ax[k].text(.85, .065, str(k), transform=ax[k].transAxes, size=in_text,color='Green',fontweight='bold')

fig.text(0.5, 0.97, 'Ca II K  clusters', ha='center', va='center', rotation='horizontal',size=12,fontweight='bold')
fig.text(0.5, 0.01, 'Wavelength [$\AA$]', ha='center', va='center', rotation='horizontal',size=12,fontweight='bold')
fig.text(0.01, 0.5, 'Normalized intensity', ha='center', va='center', rotation='vertical',size=12,fontweight='bold')
plt.savefig('/mn/stornext/u3/souvikb/paper4_images/Cluster_Cak.pdf')
plt.show()
