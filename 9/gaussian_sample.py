import numpy as np 
import matplotlib.pyplot as plt 

def gaussPlot(r,c,idx,mean,cov,title):
    x,y=np.random.multivariate_normal(mean,cov,400).T 
    ax=plt.subplot(r,c,idx)
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_title(title)
    plt.plot(x,y,':')

mean=[0,0]
cov=[[1,0],[0,1]]
gaussPlot(2,2,1,mean,cov,'N([0,0],[[1,0],[0,1]])')

mean=[4,4]
cov=[[1,0],[0,1]]
gaussPlot(2,2,2,mean,cov,'N([4,4],[[1,0],[0,1]])')


mean=[0,0]
cov=[[1,0],[0,5]]
gaussPlot(2,2,3,mean,cov,'N([0,0],[[1,0],[0,5]])')

mean=[-2,-3]
cov=[[1,2],[1,1]]
gaussPlot(2,2,4,mean,cov,'N([-2,-3],[[1,2],[1,1]])')

plt.show()