# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 17:23:06 2018

@author: chenyiping
"""

import pandas as pd
import numpy as np

from sklearn import preprocessing
import matplotlib.pyplot as plt

data_test=pd.read_csv('test.csv')
print(data_test.info()) 

#Embarked进行众数填充
print('Embarked')
print(data_test['Embarked'].describe())
data_test['Embarked']=data_test['Embarked'].fillna("S")
#进行独热编码
i=0
data_test['E_S']=None
data_test['E_C']=None
data_test['E_Q']=None
data_test.E_S[data_test['Embarked']=='S']=1
data_test.E_S[data_test['Embarked']!='S']=0
data_test.E_C[data_test['Embarked']=='C']=1
data_test.E_C[data_test['Embarked']!='C']=0
data_test.E_Q[data_test['Embarked']=='Q']=1
data_test.E_Q[data_test['Embarked']!='Q']=0

del data_test['Embarked']


#Fare缺失值填充
data_test['Fare']=data_test['Fare'].fillna(data_test['Fare'].mean())


#Cabin标记是否有,有->1,无->0
print('Cabin')
print(data_test['Cabin'].describe())
data_test.Cabin[data_test['Cabin'].notnull()]=1
data_test.Cabin[data_test['Cabin'].isnull()]=0




#Sex female->1,male->0
print('Sex')
print(data_test['Sex'].describe())
data_test.Sex[data_test['Sex']=='female']=1
data_test.Sex[data_test['Sex']=='male']=0


print(data_test.info()) 

data_test['TicketData']=None
i=0
for name in data_test['Ticket']:
    if(' 'in name):
        tmp=name.split()[-1].lstrip()
        if(tmp.isdigit()):
            data_test['TicketData'][i]=tmp        
    else:
        if(name.isdigit()):
            data_test['TicketData'][i]=name
    i+=1

print(data_test['TicketData'].describe())

data_test.to_csv('test0.csv',sep=',',index=False)

data_test['TicketData']=data_test['TicketData'].fillna(data_test['TicketData']).mean()
'''
plt.hist(data_test.TicketData)
plt.show()
'''
data_test['T0']=None
data_test['T1']=None
data_test['T2']=None
data_test.T0[data_test['TicketData']<150000]=1
data_test.T0[data_test['TicketData']>=150000]=0
data_test.T1[(data_test['TicketData']>=150000)&(data_test['TicketData']<400000)]=1
data_test.T1[(data_test['TicketData']<150000)|(data_test['TicketData']>=400000)]=0
data_test.T2[(data_test['TicketData']<1000000)]=1
data_test.T2[(data_test['TicketData']>=1000000)]=0
del data_test['TicketData']
del data_test['Ticket']
print(data_test['T1'].describe())

print(data_test.info()) 


i=0
for name in data_test['Name']:
    data_test['Name'][i]=name.split(',')[1].split('.')[0].lstrip()
    i+=1
print(data_test['Name'].unique())

data_train=pd.read_csv('train0.csv')

for k in data_test['Name'].unique():
    print(k)
    #print(data_test.Survived[data_test['Name']==k].describe())
    print(data_test[data_test['Name']==k].shape[0])
    print(data_test.Age[data_test['Name']==k].describe())
    
    #用的是train的均值和方差
    mean=data_train.Age[data_train['Name']==k].mean()
    std=data_train.Age[data_train['Name']==k].std()
    mind=data_train.Age[data_train['Name']==k].min()
    maxd=data_train.Age[data_train['Name']==k].max()
    #print(data_train.Age[data_train['Name']==name].isnull())
    size=np.shape(data_test[(data_test['Name']==k)&(data_test['Age'].isnull())])[0]
    #size=np.shape(data_test.Age[data_test['Name']==k].isnull())[0]
    if size>0:
        
        data_test.loc[(data_test.Age.isnull()&(data_test['Name']==k)),'Age']=np.random.normal(mean,std,size)
        data_test.Age[(data_test['Name']==k)&(data_test['Age']<mind)]=mind
        data_test.Age[(data_test['Name']==k)&(data_test['Age']>maxd)]=maxd

data_test.Age=data_test.Age.fillna(data_test.Age.mean())

print(data_test['Age'].describe())
print(data_test.info())

del data_test['Name']
data_test.to_csv('test1.csv',sep=',',index=False)










