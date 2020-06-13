import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D

def multivariate_gaussian(pos,mu,Sigma):
    n=mu.shape[0]
    Sigma_det=np.linalg.det(Sigma)
    Sigma_inv=np.linalg.inv(Sigma)
    N=np.sqrt((2*np.pi)**n*Sigma_det)
    fac=np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac/2)/N

N=60
X=np.linspace(-3,3,N)
Y=np.linspace(-3,4,N)
X,Y= np.meshgrid(X,Y)

mu=np.array([0.,1.])
Sigma=np.array([[1.,-0.5],[-0.5,1.5]])
pos=np.empty(X.shape+(2,))

pos[:,:,0]=X
pos[:,:,1]=Y
Z=multivariate_gaussian(pos,mu,Sigma)
fig=plt.figure()
ax=fig.gca(projection='3d')
ax.plot_surface(X,Y,Z,rstride=3,cstride=3,linewidth=1,antialiased=True,cmap=cm.viridis)
cset=ax.contourf(X,Y,Z,zdir='z',offset=-0.2,cmap=cm.viridis)
ax.set_zlim(-0.2,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27,-21)
plt.show()