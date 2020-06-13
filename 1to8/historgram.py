import numpy as np 
from matplotlib import pyplot as plt 

X1= np.random.random(1000)
X2= 10+ np.random.randn(1000)

plt.figure(figsize=(10,6))
plt.hist(X1,bins=20,alpha=0.4)
plt.hist(X2,bins=20,alpha=0.4)
plt.show()