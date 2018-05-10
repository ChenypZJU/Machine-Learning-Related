# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 13:42:32 2018

@author: chenyiping

数据可视化
这里对Ticket与Name进行了分析，保存数据为train0.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei']  

#查看数据基本信息，共有891条数据
data_train=pd.read_csv('train.csv')
print(data_train.info())    
print(data_train.head())
    
#船舱等级与获救人数的柱状图 Pclass，标称型数值，分为1,2,3
print(u'船舱等级')
print(data_train['Pclass'].describe())
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts() 
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u'各乘客等级的获救情况') 
plt.xlabel(u'乘客等级') 
plt.ylabel(u'人数') 
plt.show()

#性别与获救人数的柱状图，标称型数据，分为female和male
print(u'性别')
print(data_train['Sex'].describe())
Survived_0 = data_train.Sex[data_train.Survived == 0].value_counts() 
Survived_1 = data_train.Sex[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u'各乘客性别的获救情况') 
plt.xlabel(u'乘客性别') 
plt.ylabel(u'人数') 
plt.show()

#年龄与获救情况分布
print('年龄')
print(data_train['Age'].describe())
print('未获救')
Survived_0 = data_train.Age[data_train.Survived == 0]
print(Survived_0 .describe())
print('获救')
Survived_1 = data_train.Age[data_train.Survived == 1]
print(Survived_1 .describe())
plt.hist(data_train.Age[data_train['Age'].notnull()])
plt.hist(data_train.Age[(data_train['Age'].notnull())&(data_train.Survived == 1)])
plt.title(u'乘客年龄分布')
plt.legend(labels = [u'未获救', u'获救'], loc = 'best')
plt.show()


#是否有兄弟情况与获救人数柱状图 SibSp
print('SibSp')
Survived_0 = data_train.SibSp[data_train.Survived == 0].value_counts() 
Survived_1 = data_train.SibSp[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u'各乘客SibSp情况的获救情况') 
plt.xlabel(u'乘客SibSp情况') 
plt.ylabel(u'人数') 
plt.show()


#是否有父母情况与获救人数柱状图 Parch
print('Parch')
Survived_0 = data_train.Parch[data_train.Survived == 0].value_counts() 
Survived_1 = data_train.Parch[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u'各乘客Parch情况的获救情况') 
plt.xlabel(u'乘客Parch情况') 
plt.ylabel(u'人数') 
plt.show()


#fare与获救的散点图，无缺失值，需离散化    
print('Fare')
plt.scatter(data_train['Survived'],data_train['Fare'])
plt.show()
print('未获救人员船费分布')
print(data_train[data_train['Survived']==0]['Fare'].describe())
print('获救人员船费分布')
print(data_train[data_train['Survived']==1]['Fare'].describe())
    
plt.hist(data_train.Fare[data_train['Fare'].notnull()])
plt.hist(data_train.Fare[(data_train['Fare'].notnull())&(data_train.Survived == 1)])
plt.title(u'获救人员船费分布')
plt.legend(labels = [u'未获救', u'获救'], loc = 'best')
plt.show()


print('Cabin')
print(data_train['Cabin'].describe())
plt.hist(data_train.Survived)
plt.hist(data_train.Survived[data_train['Cabin'].notnull()])
plt.legend(labels = [u'有Cabin', u'无Cabin'], loc = 'best')
plt.title(u'各乘客有无Cabin的获救情况') 
plt.xlabel(u'乘客获救情况') 
plt.ylabel(u'人数') 
plt.show()


print('Embarked')
print(data_train['Embarked'].describe())
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts() 
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u'各乘客Embarked情况的获救情况') 
plt.xlabel(u'乘客Embarked情况') 
plt.ylabel(u'人数') 
plt.show()


#查看Ticket与获救情况关系


print('Ticket')
print(data_train['Ticket'].describe())

#将字符串与数字分开
data_train['TicketData']=None
data_train['TicketStr']=None
i=0
for name in data_train['Ticket']:
    if(' 'in name):
        data_train['TicketData'][i]=name.split()[-1].lstrip()
        data_train['TicketStr'][i]=name.split()[0]
    else:
        if(name.isdigit()):
            data_train['TicketData'][i]=name
        else:
            data_train['TicketStr'][i]=name
    i+=1

#该行数据表明ticket字符串部分与获救无关
print(data_train.Survived.describe())
print(data_train.Survived[data_train['TicketStr'].isnull()].describe())
del data_train['TicketStr']
del data_train['Ticket']
data_train.to_csv('train0.csv',sep=',',index=False)

#重新读取
data_train=pd.read_csv('train0.csv')


#该图表明ticket部分与获救情况有关

plt.hist(data_train.TicketData[(data_train['TicketData'].notnull())])
plt.hist(data_train.TicketData[(data_train.Survived == 1)&(data_train['TicketData'].notnull())])
plt.show()

plt.hist(data_train.TicketData[(data_train['TicketData'].notnull())&(data_train['TicketData']<1000000)])
plt.hist(data_train.TicketData[(data_train.Survived == 1)&(data_train['TicketData'].notnull())&(data_train['TicketData']<1000000)])


plt.title(u'各乘客TicketData情况的获救情况') 
plt.legend(labels = [u'未获救', u'获救'], loc = 'best')
plt.show()

#这里ticketData分为三组，分界线为150000,400000
print(data_train.Survived[(data_train['TicketData'].notnull())&(data_train['TicketData']<150000)].describe())
print(data_train.Survived[(data_train['TicketData'].notnull())&(data_train['TicketData']>150000)&(data_train['TicketData']<400000)].describe())
print(data_train.Survived[(data_train['TicketData'].notnull())&(data_train['TicketData']>400000)].describe())
'''



#
'''
#提取名字中的身份信息，便于对年龄进行填充

print(data_train['Name'].unique())


i=0
for name in data_train['Name']:
    data_train['Name'][i]=name.split(',')[1].split('.')[0].lstrip()
    i+=1
print(data_train['Name'].unique())
    
for k in data_train['Name'].unique():
    print(k)
    print(data_train.Survived[data_train['Name']==k].describe())
    print(data_train.Age[data_train['Name']==k].describe())


del data_train['PassengerId']
data_train.to_csv('train0.csv',sep=',',index=False)

