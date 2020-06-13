import numpy as np 
import pandas as pd 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet 
from sklearn.metrics import mean_squared_error 

boston=load_boston()
data=pd.DataFrame(boston.data,columns=boston.feature_names)
X=data[['RM','PTRATIO','RAD','TAX','LSTAT','CRIM','NOX','B']].values
y=boston.target


sc_x=StandardScaler()
sc_y=StandardScaler()
X_std=sc_x.fit_transform(X)
y_std=sc_y.fit_transform(y[:,np.newaxis]).flatten()
X_train,X_test,y_train,y_test=train_test_split(X_std,y_std,test_size=0.2,random_state=123)

linear=LinearRegression() 
ridge=Ridge(alpha=1.0,random_state=0)
lasso=Lasso(alpha=1.0,random_state=0)
enet=ElasticNet(alpha=1.0,l1_ratio=0.5)

linear.fit(X_train,y_train)
ridge.fit(X_train,y_train)
lasso.fit(X_train,y_train)
enet.fit(X_train,y_train)


linear_pred=linear.predict(X_train)
ridge_pred=ridge.predict(X_train)
lasso_pred=lasso.predict(X_train)
enet_pred=enet.predict(X_train)

print('Linear - RMSE for training data:', np.sqrt(mean_squared_error(y_train,linear_pred)))
print('Ridge - RMSE for training data:', np.sqrt(mean_squared_error(y_train,ridge_pred)))
print('Lasso - RMSE for training data:', np.sqrt(mean_squared_error(y_train,lasso_pred)))
print('Elastic Net - RMSE for training data:', np.sqrt(mean_squared_error(y_train,enet_pred)))


linear_pred=linear.predict(X_test)
ridge_pred=ridge.predict(X_test)
lasso_pred=lasso.predict(X_test)
enet_pred=enet.predict(X_test)

print('Linear - RMSE for test data:', np.sqrt(mean_squared_error(y_test,linear_pred)))
print('Ridge - RMSE for test data:', np.sqrt(mean_squared_error(y_test,ridge_pred)))
print('Lasso - RMSE for test data:', np.sqrt(mean_squared_error(y_test,lasso_pred)))
print('Elastic Net - RMSE for test data:', np.sqrt(mean_squared_error(y_test,enet_pred)))



