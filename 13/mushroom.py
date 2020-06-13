import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC

mushroom=pd.read_csv('mushrooms.csv',header=None)
print(mushroom.head(4))

X=[]
Y=[]

for idx , row in mushroom[1:].iterrows():
    Y.append(row.loc[0])
    row_x=[]
    for v in row.loc[1:]:
        row_x.append(ord(v))
    X.append(row_x)

print("\n속성:\n",X[0:3])
print("\n부류\n",Y[0:3])
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25)

svc=SVC()
svc.fit(x_train,y_train)

print("학습 데이터 정확도:",svc.score(x_train,y_train))
print("테스트 데이터 정확도:",svc.score(x_test,y_test))