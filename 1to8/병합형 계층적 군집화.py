from sklearn.datasets import make_blobs 
from scipy.spatial.distance import pdist 
from scipy.cluster.hierarchy import linkage 
from scipy.cluster.hierarchy import dendrogram 
import numpy as np 
from matplotlib import pyplot as plt 
X,Y = make_blobs(n_samples=25, n_features=2, centers=3, cluster_std=1.5) 
colormap = np.array(['r','g', 'b']) 
plt.scatter(X[:,0], X[:,1], c=colormap[Y]) 
plt.title('Generated Data') 
plt.show() 
Xdist = pdist(X, metric='euclidean') 
Z = linkage(Xdist, method='ward') 
Zd = dendrogram(Z) 
plt.title('Dendrogram') 
plt.show()
