# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:10:22 2018

@author: chenyiping
"""
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import numpy
import math


sales = pd.read_csv(open('home_data.csv'))


'''
## 1. Selection and summary statistics:
    
'''    
    
Seattle_sales=sales[sales['zipcode']==98039]
#print(Seattle_sales.head())
print(Seattle_sales['price'].mean())

#the answer is 2160606.6
'''
##2. Filtering data:

'''
sales_part=sales[(sales.sqft_living>2000) & (sales.sqft_living<4000)]
print(len(sales_part)/len(sales))

#the answer is 0.4215518437977143

'''
##3. Building a regression model with several more features:
'''
#the function is different from random.split() in SFrame
train_data,test_data=train_test_split(sales,test_size=0.2,random_state=0)


#print(len(train_data))
#print(len(test_data))

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
advanced_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]

#create the model
my_feature_model=linear_model.LinearRegression()
advanced_feature_model = linear_model.LinearRegression()

my_feature_model.fit(train_data.loc[:,my_features],train_data.loc[:,['price']])
advanced_feature_model.fit(train_data.loc[:,advanced_features],train_data.loc[:,['price']])

#predict 
res_my=my_feature_model.predict(test_data.loc[:,my_features])
res_advanced=advanced_feature_model.predict(test_data.loc[:,advanced_features])


#output the result
test_ans=numpy.array(test_data['price'])
rsme_my=[(x-y)**2 for x,y in zip(res_my,test_ans)]
rsme_advanced=[(x-y)**2 for x,y in zip(res_advanced,test_ans)]

print(math.sqrt(sum(rsme_my)/len(test_ans))-math.sqrt(sum(rsme_advanced)/len(test_ans)))







