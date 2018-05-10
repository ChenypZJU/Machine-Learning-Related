# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:21:20 2018

@author: chenyiping
"""

import pandas as pd
import numpy as np

from sklearn import preprocessing


data_train=pd.read_csv('train0.csv')
print(data_train.info()) 

#Embarked进行众数填充
print('Embarked')
print(data_train['Embarked'].describe())
data_train['Embarked']=data_train['Embarked'].fillna("S")
#进行独热编码
i=0
data_train['E_S']=None
data_train['E_C']=None
data_train['E_Q']=None
data_train.E_S[data_train['Embarked']=='S']=1
data_train.E_S[data_train['Embarked']!='S']=0
data_train.E_C[data_train['Embarked']=='C']=1
data_train.E_C[data_train['Embarked']!='C']=0
data_train.E_Q[data_train['Embarked']=='Q']=1
data_train.E_Q[data_train['Embarked']!='Q']=0

del data_train['Embarked']


#Cabin标记是否有,有->1,无->0
print('Cabin')
print(data_train['Cabin'].describe())
data_train.Cabin[data_train['Cabin'].notnull()]=1
data_train.Cabin[data_train['Cabin'].isnull()]=0
#这里表明有无Cabin，获救概率相差一半
print(data_train.Survived[data_train['Cabin']==1].describe())
print(data_train.Survived[data_train['Cabin']==0].describe())


#Sex female->1,male->0
print('Sex')
print(data_train['Sex'].describe())
data_train.Sex[data_train['Sex']=='female']=1
data_train.Sex[data_train['Sex']=='male']=0
print(data_train.Survived[data_train['Sex']==1].describe())
print(data_train.Survived[data_train['Sex']==0].describe())


#TicketData ,

print(data_train.info()) 
print(data_train['TicketData'].describe())
data_train['TicketData']=data_train['TicketData'].fillna(data_train['TicketData']).mean()
data_train['T0']=None
data_train['T1']=None
data_train['T2']=None
data_train.T0[data_train['TicketData']<150000]=1
data_train.T0[data_train['TicketData']>=150000]=0
data_train.T1[(data_train['TicketData']>=150000)&(data_train['TicketData']<400000)]=1
data_train.T1[(data_train['TicketData']<150000)|(data_train['TicketData']>=400000)]=0
data_train.T2[(data_train['TicketData']<1000000)]=1
data_train.T2[(data_train['TicketData']>=1000000)]=0
del data_train['TicketData']
print(data_train.info()) 

#data_train.to_csv('train1.csv',sep=',',index=False)

#年龄用相似身份的正态随机数填充，用最大值和最小值限制范围
for name in data_train['Name'].unique():
    print(name)
    #print(data_train.Name[data_train['Name']==name].describe())
    print(data_train.Age[data_train['Name']==name].describe())
    mean=data_train.Age[data_train['Name']==name].mean()
    std=data_train.Age[data_train['Name']==name].std()
    mind=data_train.Age[data_train['Name']==name].min()
    maxd=data_train.Age[data_train['Name']==name].max()
    #print(data_train.Age[data_train['Name']==name].isnull())
    size=np.shape(data_train[(data_train['Name']==name)&(data_train['Age'].isnull())])[0]
    print(size)
    #print(np.shape(pd.Series(np.random.normal(mean,std,size))))
    
    data_train.loc[(data_train.Age.isnull()&(data_train['Name']==name)),'Age']=np.random.normal(mean,std,size)
    
    #data_train.Age[(data_train['Age'].isnull())&(data_train['Name']==name)]=pd.Series(np.random.normal(mean,std,size))
    
    print(data_train.Age[data_train['Name']==name].describe())
    data_train.Age[(data_train['Name']==name)&(data_train['Age']<mind)]=mind
    data_train.Age[(data_train['Name']==name)&(data_train['Age']>maxd)]=maxd
        
    print(data_train.Age[data_train['Name']==name].describe())
    
print(data_train.Age.describe())   

   
del data_train['Name']
data_train.to_csv('train2.csv',sep=',',index=False)
