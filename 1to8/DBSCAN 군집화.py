from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN 
import numpy as np 
from matplotlib import pyplot as plt
X,Y = make_moons(n_samples=1000, noise=0.05)
plt.title('Half moons') 
plt.scatter(X[:,0], X[:,1]) 
plt.show()
dbs = DBSCAN(eps=0.1) 
Z = dbs.fit_predict(X)

colormap = np.array(['r', 'b']) 
plt.scatter(X[:,0], X[:,1], c=colormap[Z]) 
plt.title('DBSCAN for half moons') 
plt.show()
