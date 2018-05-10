# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 17:07:06 2018

@author: chenyiping
这里采用逻辑回归进行预测
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier


data_train=pd.read_csv('train2.csv')
X=data_train[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','E_S','E_C','E_Q','T0','T1','T2']]
y=data_train['Survived']
#设定0.4的测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#print(X.info())

#标准化
sc = StandardScaler()
sc.fit(X)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_s=sc.transform(X)

adaDt=AdaBoostClassifier(n_estimators=20)

scores=cross_val_score(adaDt,X_s,y,cv=6)
print(scores.mean())

adaDt.fit(X_train_std,y_train)
print(adaDt.score(X_test_std, y_test))

'''
#建立模型
lr = LogisticRegression(C=20,max_iter=100)
lr.fit(X_train_std,y_train)

#scores=cross_val_score(lr,X_test_std,y_test,cv=6)

print(lr.score(X_test_std, y_test))

scores=cross_val_score(lr,X_train_std,y,cv=5)
print(scores.mean())

#res=lr.predict(X_test_std)
'''

data_test=pd.read_csv('test1.csv')
test_X=data_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','E_S','E_C','E_Q','T0','T1','T2']]
X_test_std=sc.transform(test_X)

data_test['Survived']=None
data_test['Survived']=adaDt.predict(X_test_std)

ress=data_test[['PassengerId','Survived']]
ress.to_csv('result_lr.csv',sep=',',index=False)