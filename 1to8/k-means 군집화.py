import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.cluster import KMeans 

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0) 
plt.scatter(X[:,0], X[:,1], s=4) 
plt.title('Generated Data') 
plt.show() 
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0) 
pred_y = kmeans.fit_predict(X) 
colormap = np.array(['c','g', 'b', 'm']) 
plt.scatter(X[:,0], X[:,1], s=4, c=colormap[pred_y]) 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s=100, c='red') 
plt.title('Clustering Results with Centers') 
plt.show() 