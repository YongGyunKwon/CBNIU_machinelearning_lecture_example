import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import BernoulliNB 
from sklearn.metrics import accuracy_score 

email_list=[
    {'title':'free game only today','spam':True},
    {'title':'cheapest flight deal','spam':True},
    {'title':'limited time offer only today only today','spam':True},
    {'title':'your flight schedule attached','spam':False},
    {'title':'today meeting schedule','spam':False},
    {'title':'your credit card statement','spam':False}]

df=pd.DataFrame(email_list)

df['label']=df['spam'].map({True:1,False:0})
df_x=df['title']
df_y=df['label']
print("** 데이터 **\n",df)

cv=CountVectorizer(binary=True)
x_traincv=cv.fit_transform(df_x)
print("\n변환된 데이터 표현(1~2번째 이메일) **\n",x_traincv[0:2])

encoded_input=x_traincv.toarray() 
print("\n 이진 벡터 표현 ** \n",encoded_input)

print("\n**벡터 위치별 대응 단어 **\n",cv.get_feature_names())
print("\n**첫번째 벡터에 대응하는 단어들 **\n",cv.inverse_transform(encoded_input[0]))

Bnb=BernoulliNB()
y_train=df_y.astype('int')
Bnb.fit(x_traincv,y_train)

test_email_list=[
    {'title':'free flight offer','spam':True},
    {'title':'hey traveler free flight deal','spam':True},
    {'title':'limited free game offer','spam':True},
    {'title':'today flight schedule','spam':False},
    {'title':'your credit card attached','spam':False},
    {'title':'free credit card offer only today','spam':False}]

test_df=pd.DataFrame(test_email_list)
test_df['label']=test_df['spam'].map({True:1,False:0})
test_x=test_df['title']
test_y=test_df['label']
x_testcv=cv.transform(test_x)

predicted_y=Bnb.predict(x_testcv)
print("\n** Predicted Label **\n",predicted_y)
accuracy = accuracy_score(test_y,predicted_y)
print("\nAccuracy = ",accuracy)