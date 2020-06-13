from sklearn.tree import DecisionTreeRegressor
import pandas as pd 
import matplotlib.pyplot as plt 

df=pd.read_csv('housing.data',header=None,sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
print(df.head())
X=df[['LSTAT']].values
y=df['MEDV'].values

tree=DecisionTreeRegressor(max_depth=3)
tree.fit(X,y)

sort_idx=X.flatten().argsort()
plt.scatter(X[sort_idx],y[sort_idx],c='lightblue')
plt.plot(X[sort_idx],tree.predict(X[sort_idx]),color='red',linewidth=2)
plt.xlabel('LSTAT(% Lower Status of the Popuation)')
plt.ylabel('MEDV(Price in $1000)')
plt.show()