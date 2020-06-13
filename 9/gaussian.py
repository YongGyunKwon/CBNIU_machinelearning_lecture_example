import numpy as np
import matplotlib.pyplot as plt

def gauss(mu,sigma):
    return np.exp(-(x-mu)**2/sigma**2) / (np.sqrt(2*np.pi)*sigma)

x=np.linspace(-4,8,100)

plt.figure(figsize=(4,4))
plt.plot(x,gauss(0,1), 'black', linewidth=2, label='N(0,1)')
plt.plot(x,gauss(2,3), 'gray', linewidth=3, label='N(2,3)')
plt.legend()
plt.ylim(-0.1,0.5)
plt.xlim(-4,8)
plt.xlabel('x')
plt.ylabel('probability density')
plt.title('Gaussian distribution')
plt.grid(True)
plt.show()