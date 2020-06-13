from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

X,y = make_blobs(n_samples = 300, centers = 4, cluster_std=0.60, random_state=0)
plt.scatter(X[:,0], X[:,1])
plt.show()

SSEs = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit_predict(X)
    SSEs.append(kmeans.inertia_)

plt.plot(range(1,11),SSEs, marker='o')
plt.show()