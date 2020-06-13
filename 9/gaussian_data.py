import numpy as np 
import matplotlib.pyplot as plt 

m=10
sigma=2
x1=np.random.randn(10000)
X2=m+sigma*np.random.randn(10000)

plt.figure(figsize=(10,6))
plt.hist(x1,bins=20,alpha=0.4,label='N(0,1)')
plt.hist(X2,bins=20,alpha=0.4,label='N(10,2)')
plt.legend()
plt.show()