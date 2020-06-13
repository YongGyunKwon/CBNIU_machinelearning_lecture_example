import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import AdaBoostRegressor

rng = np.random.RandomState(1)
X= np.linspace(0,6,100)[:,np.newaxis]
y=np.sin(X).ravel()+np.sin(6*X).ravel()+rng.normal(0,0.1,X.shape[0])

regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=100,random_state=rng)

regr_1.fit(X,y)
regr_2.fit(X,y)
y_1=regr_1.predict(X)
y_2=regr_2.predict(X)

plt.figure()
plt.scatter(X,y,c="k",label="training samples")
plt.plot(X,y_1,c="g",label="n_estimators=1",linewidth=2)
plt.plot(X,y_2,c="r",label="n_estimators=100",linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title(" AdaBoost Regression")
plt.legend()
plt.show()