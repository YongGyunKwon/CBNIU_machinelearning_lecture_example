from sklearn.datasets import make_blobs 
from sklearn.cluster import AgglomerativeClustering 
import numpy as np 
from matplotlib import pyplot as plt 
X,Y = make_blobs(n_samples=2000, n_features=2, centers=8, cluster_std=2.0) 
plt.scatter(X[:,0], X[:,1], s=4) 
plt.title('Generated Data') 
plt.show() 
Z = AgglomerativeClustering(n_clusters=8, linkage='complete') 
P = Z.fit_predict(X) 
colormap = np.array(['r', 'g' ,'b', 'k', 'y', 'c', 'm', 'orange']) 
plt.scatter(X[:,0], X[:,1], s=4, c=colormap[P]) 
plt.title('Clustering results') 
plt.show()
