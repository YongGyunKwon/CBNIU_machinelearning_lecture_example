import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score

review_list =[
    {'review':'This is a great movie. I will watch again','type':'positive'},
    {'review':'I like this movie','type':'positive'},
    {'review':'amazing movie this year','type':'positive'},
    {'review':'cool my boyfriend also said the movie is cool','type':'positive'},
    {'review':'awesome of the awesome movie ever','type':'positive'},
    {'review':'shame I waste money ','type':'negative'},
    {'review':'regret on this movie.I will never what movie from this director','type':'negative'},
    {'review':'I do not like this movie','type':'negative'},
    {'review':'I do not like actors in this movie','type':'negative'},
    {'review':'boring boring sleeping movie','type':'negative'}]

df=pd.DataFrame(review_list)
df['label']=df['type'].map({'positive':1,'negative':0})
df_x=df['review']
df_y=df['label']

cv=CountVectorizer() 
x_traincv=cv.fit_transform(df_x)
encoded_input=x_traincv.toarray()
print("\n벡터 표현\n",encoded_input)
print("\n벡터의 원소 위치별 단어\n",cv.get_feature_names())

test_review_list =[
    {'review':'great great great movie ever','type':'positive'},
    {'review':'I like this amazing movie','type':'positive'},
    {'review':'my boyfriend said great movie ever','type':'positive'},
    {'review':'cool cool cool','type':'positive'},
    {'review':'awesome boyfried said cool movie ever','type':'positive'},
    {'review':'shame shame shame','type':'negative'},
    {'review':'awesome director shame movie boring movie','type':'negative'},
    {'review':'do not like this movie','type':'negative'},
    {'review':'I do not ike this boring movie','type':'negative'},
    {'review':'aweful terrible boring movie','type':'negative'}]

test_df=pd.DataFrame(test_review_list)
test_df['label']=test_df['type'].map({'positive':1,'negative':0})
test_x=test_df['review']
test_y=test_df['label']

x_testcv=cv.transform(test_x)

Mnb=MultinomialNB()
y_train=df_y.astype('int')
Mnb.fit(x_traincv,y_train)
predicted_y=Mnb.predict(x_testcv)
print("\n** ground truth **\n",test_y)
print("\n** 에측치 **\n",predicted_y)
accuray=accuracy_score(test_y,predicted_y)
print("\n** 정확도 **\n",accuray)