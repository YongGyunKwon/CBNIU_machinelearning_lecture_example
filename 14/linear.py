import numpy as np

class LinearRegressionGD(object):
    def __init__(self,eta=0.001, n_iter=20):
        self.eta=0.001
        self.n_iter=20

    
    def fit(self,X,y):
        self.w_=np.zeros(1+X.shape[1])
        self.cost_=[]

        for i in range(self.n_iter):
            output=self.net_input(X)
            errors=(y-output)
            self.w_[1:]+=self.eta*X.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            cost=(errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]

    def predict(self,X):
        return self.net_input(X)


import pandas as pd 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

boston=load_boston() 
data=pd.DataFrame(boston.data,columns=boston.feature_names)

X=data[['RM','PTRATIO']].values
y=boston.target

print(data.head())
print(y[0:5])

sc_x=StandardScaler()
sc_y=StandardScaler()
X_std=sc_x.fit_transform(X)
y_std=sc_y.fit_transform(y[:,np.newaxis]).flatten()
X_train,X_test,y_train,y_test=train_test_split(X_std,y_std,test_size=0.2,random_state=123)

lr=LinearRegressionGD()
lr.fit(X_train,y_train)


import matplotlib.pyplot as plt 

plt.plot(range(1,lr.n_iter+1),lr.cost_)
plt.ylabel('Sum of Squared Error')
plt.xlabel('Iterations')
plt.show()

from sklearn.metrics import mean_squared_error
preds=lr.predict(X_test)
mse=mean_squared_error(y_test,preds)
print('Root mean squared error:',np.sqrt(mse))