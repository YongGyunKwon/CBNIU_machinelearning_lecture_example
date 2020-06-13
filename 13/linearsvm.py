import numpy as np 
from sklearn.datasets import make_blobs 
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot as plt 
from sklearn.svm import LinearSVC 
from sklearn.metrics import confusion_matrix 

X,y=make_blobs(n_samples=250, centers=2, random_state=0, cluster_std=0.85)
y[y==0] = -1 
y=y.astype(float)
X_train, X_test,y_train,y_test=train_test_split(X,y,random_state=0)

svc=LinearSVC(C=0.1)
svc.fit(X_train,y_train)

y_pred=svc.predict(X_train)
print('training data = \n',confusion_matrix(y_train,y_pred))

y_pred=svc.predict(X_test)
print('testing data =\n ', confusion_matrix(y_test,y_pred))


def f(x,w,b,c=0):
    return (-w[0]*x-b+c)/w[1]

plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap='winter')
ax=plt.gca()
xlim=ax.get_xlim()
w=svc.coef_[0]
a= -w[0]/w[1]
xx=np.linspace(xlim[0],xlim[1])
yy=a*xx-svc.intercept_[0]/w[1]
plt.plot(xx,yy)
xx=a*xx-(svc.intercept_[0]-1)/w[1]
plt.plot(xx,yy,'k--')
yy=a*xx-(svc.intercept_[0]+1)/w[1]
plt.plot(xx,yy,'k--')
plt.xlim(-1,4)
plt.title('Linear SVM(C=0.1)')

plt.show()