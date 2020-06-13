import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

pdiabetes=pd.read_csv('diabetes.csv',header=None)
print(pdiabetes[0:5])

x=pdiabetes.iloc[1:,:8]
y=pdiabetes.iloc[1:,8:].values.flatten()

print('x shape:',x.shape,'y shape',y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

std_scl=StandardScaler()
std_scl.fit(x_train)

x_train=std_scl.transform(x_train)
x_test=std_scl.transform(x_test)

svc=SVC(kernel='rbf')
svc.fit(x_train,y_train)

print('학습 데이터 정확도:',svc.score(x_train,y_train))
print('테스트 데이터 정확도 :',svc.score(x_test,y_test))