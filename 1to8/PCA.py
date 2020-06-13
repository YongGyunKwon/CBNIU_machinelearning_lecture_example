from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import numpy as np 

iris=datasets.load_iris()
print(list(iris.keys()))
X=iris["data"][:,0:4]
label=iris["target"]

pca=PCA(n_components=2)
X2D=pca.fit_transform(X)

for i,j in enumerate(np.unique(label)):
    plt.scatter(X2D[label==j,0],X2D[label==j,1],
    c=ListedColormap(('red','green','blue'))(i),label=j)

plt.legend()
plt.show()