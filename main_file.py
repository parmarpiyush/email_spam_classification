import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('mail_data.csv')
print(df)

data = df.where(pd.notnull(df))
data.head(7)

data.info()

data.shape

data.loc[data['Category'] == 'spam','Category',] = 0
data.loc[data['Category'] == 'ham','Category',] = 1

X = data['Message']
Y = data['Category']

print(X)
print(Y)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 3)
feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english',lowercase = True)
x_train_feature = feature_extraction.fit_transform(x_train)
x_test_feature = feature_extraction.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

print(x_train)
print(x_test_feature)

model = LogisticRegression()

model.fit(x_train_feature,y_train)

y_hat = model.predict(x_train_feature)
acc_train = accuracy_score(y_train,y_hat)
print(acc)

y_hat_test = model.predict(x_test_feature)
acc_test = accuracy_score(y_test,y_hat_test)
print(acc_test)


txt  = 'SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info'
lower_txt = txt.lower()

mail = [lower_txt]
feature_mail = feature_extraction.transform(mail)
new_predict = model.predict(feature_mail)

if(new_predict == 1):
    print('This is Not Spam Mail')
else:
    print('This is Spam Mail')

