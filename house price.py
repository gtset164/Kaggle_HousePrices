#!/usr/bin/env python
# coding: utf-8

# **House Prices: Advanced Regression Techniques
# 
# This project is a compeition in Kaggle with initially 79 features for the residential homes in Ames, Iowa. The goal of this challenge is to predict the final price of each home.
# As usual, data exploration, feature engineering are applied, then several machine learning models are built to predict the house price in the test set.

# In[70]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

get_ipython().run_line_magic('matplotlib', 'inline')

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# Functions which are used in this project. They can also be reused in other projects.

# In[71]:


# Returns a concatenated dataframe of training and test set on axis 0
def concat_df(train_data, test_data):
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

# Returns divided dfs of training and test set
def divide_df(all_data, sizeof_first):
    return all_data.loc[:sizeof_first-1,:], all_data.loc[sizeof_first:,:]    

# Onehot encoding function
def onehot_features(data, cols):
    multicols = data[cols]
    multicols = multicols.astype(str)
    one_hot = pd.get_dummies(multicols)
    return one_hot

# Feature scaling function
def scaling(data,cols):
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), cols)
    return scaled_data

# Map categorical strings in the features to numbers
def map_category(series, mapping):
    digit_data = series.map(mapping)
    return digit_data

# Calculate the probability mass function for statistics
def pmf(series, b_normal):
    a = series.value_counts()
    if b_normal:
        a = a/a.sum()
    return a


# Data Exploration

# In[72]:


print(df_train.shape)
print(df_test.shape)


# In[73]:


df_train.info()


# In[74]:


df_train.describe(include='all')


# In[75]:


df_train.head()


# In[76]:


# Extract the house prices to a new series
y_train = df_train['SalePrice']
df_train.drop('SalePrice', axis=1, inplace=True)

# Remove Id columns from both training and test set
df_train.drop('Id', axis=1, inplace=True)
df_test.drop('Id', axis=1, inplace=True)

# Check the data shape now
print(df_train.shape)
print(df_test.shape)
print(y_train.shape)


# Check the null values in training set

# In[77]:


df_nu = df_train.isnull().sum()

# Only list the features with null values for later processing
print(df_nu[df_nu > 0])


# Check the null values in test set

# In[78]:


df_nu = df_test.isnull().sum()

# Only list the features with null values for later processing
print(df_nu[df_nu > 0])


# There are 4 features 'Alley','PoolQC','Fence','MiscFeature' with almost all null values. Merge training and test data first then remove the 4 features.

# In[79]:


# Merge training and test set to process feature engineering together
df_all = concat_df(df_train, df_test)
print(df_all.shape)


# In[80]:


df_all.drop(['Alley','PoolQC','Fence','MiscFeature'], axis = 1, inplace= True)


# Initially fill in the null values for all features with null values above.

# In[81]:


df_all['MSZoning'].fillna('RL', inplace=True) # test set only
df_all['LotFrontage'].fillna(df_all['LotFrontage'].mean(), inplace =True)       
df_all['Utilities'].fillna('AllPub', inplace=True) # test set only
df_all['Exterior1st'].fillna('VinylSd', inplace=True) # test set only
df_all['Exterior2nd'].fillna('VinylSd', inplace=True) # test set only
df_all['MasVnrType'].fillna('None', inplace=True) 
df_all['MasVnrArea'].fillna('0', inplace=True)

df_all['BsmtQual'].fillna('TA', inplace=True)
df_all['BsmtCond'].fillna('TA', inplace=True)
df_all['BsmtExposure'].fillna('No', inplace=True)
df_all['BsmtFinType1'].fillna('GLQ', inplace=True)
df_all['BsmtFinType2'].fillna('Unf', inplace=True)
df_all['BsmtFinSF1'].fillna('0', inplace=True) # test set only
df_all['BsmtFinSF2'].fillna('0', inplace=True) # test set only
df_all['BsmtUnfSF'].fillna(df_all['BsmtUnfSF'].median(), inplace=True) # test set only
df_all['TotalBsmtSF'].fillna(df_all['BsmtUnfSF'].median(), inplace=True) # test set only
df_all['BsmtFullBath'].fillna('0', inplace=True) # test set only
df_all['BsmtHalfBath'].fillna('0', inplace=True) # test set only

df_all['Electrical'].fillna('SBrkr', inplace=True) # training set only
df_all['KitchenQual'].fillna('TA', inplace=True) # test set only
df_all['Functional'].fillna('Typ', inplace=True) # test set only
df_all['FireplaceQu'].fillna('TA', inplace=True)
       
df_all['GarageType'].fillna('Attchd', inplace=True)
df_all['GarageYrBlt'].fillna(df_all['GarageYrBlt'].mean(), inplace=True)
df_all['GarageFinish'].fillna('RFn', inplace=True)
df_all['GarageCars'].fillna('2', inplace=True) #test set only
df_all['GarageArea'].fillna(df_all['GarageArea'].mean(), inplace=True) #test set only
df_all['GarageQual'].fillna('TA', inplace=True)
df_all['GarageCond'].fillna('TA', inplace=True)
 
df_all['SaleType'].fillna('WD', inplace=True) #test set only

# Make sure there is no null items
df_all.isnull().sum()


# Now, handle several types of features in different ways:
# 1. Do one-hot encoding for categorical and norminal data
# 2. For categorical and ordinal data, map all values to numbers
# 3. Do normalization and scaling finally for all features - including both categorical data and regression data. It's not essential (optioanl) for tree related algorithms.

# In[82]:


# Check the data summary so far
print(df_all.columns)
print(df_all.head())
print(df_all.shape)


# Use one-hot encoding to handle categorical + norminal features below:
# MSZoning
# Street
# LotShape
# LandContour
# Utilities
# LotConfig
# LandSlope
# Neighborhood
# Condition1
# Condition2
# BldgType
# HouseStyle
# RoofStyle
# RoofMatl
# Exterior1st
# Exterior2nd
# MasVnrType
# Foundation
# Heating
# CentralAir
# Electrical
# Functional
# GarageType
# GarageFinish
# MoSold
# SaleType
# SaleCondition
# 

# In[83]:


cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 'GarageFinish', 'MoSold', 'SaleType', 'SaleCondition']
onehot = onehot_features(df_all, cols)

# Drop the original features
df_all.drop(cols, axis = 1, inplace = True)

print(type(onehot))
print(onehot.shape)
print(df_all.shape)

df_all = pd.concat([df_all, onehot], axis=1)

print(df_all.shape)


# Map all the categorical ordianl features which the order of the values make sense and their values are not numerical below:
# ExterQual
# ExterCond
# BsmtQual
# BsmtCond
# BsmtExposure
# BsmtFinType1
# BsmtFinType2
# HeatingQC
# KitchenQual
# FireplaceQu
# GarageQual
# GarageCond
# PavedDrive
# PoolQC
# Fence
# 
# The mappings can be determined by referring the data_description file which describes the meanings of each features.

# In[84]:


mapping = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0, 'Av':3, 'Mn':2, 'No':1}
df_all['ExterQual'] = df_all['ExterQual'].astype(str)
df_all['ExterQual'] = df_all['ExterQual'].map(mapping)
df_all['ExterCond'] = map_category(df_all['ExterCond'], mapping)
df_all['BsmtQual'] = map_category(df_all['BsmtQual'], mapping)
df_all['BsmtCond'] = map_category(df_all['BsmtCond'], mapping)
df_all['BsmtExposure'] = map_category(df_all['BsmtExposure'], mapping)
df_all['HeatingQC'] = map_category(df_all['HeatingQC'], mapping)
df_all['KitchenQual'] = map_category(df_all['KitchenQual'], mapping)
df_all['FireplaceQu'] = map_category(df_all['FireplaceQu'], mapping)
df_all['GarageQual'] = map_category(df_all['GarageQual'], mapping)
df_all['GarageCond'] = map_category(df_all['GarageCond'], mapping)

mapping = {'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'NA':0}
df_all['BsmtFinType1'] = map_category(df_all['BsmtFinType1'], mapping)
df_all['BsmtFinType2'] = map_category(df_all['BsmtFinType2'], mapping)

mapping = {'Y':2, 'P':1, 'N':0}
df_all['PavedDrive'] = map_category(df_all['PavedDrive'], mapping)


# In[85]:


df_nu = df_all.isnull().sum()

# Make sure if there is no null value
print(df_nu[df_nu>0])


# In[86]:


#df_all.to_csv(r'D:\ML\Kaggle projects\House Prices Advanced Regression Techniques\check None.csv')

# Check the data set dimensions so far again.
print(df_train.shape)
print(df_all.shape)
X_train, X_test = divide_df(df_all, df_train.shape[0])
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)    


# Scaling features

# In[87]:


col = X_train.columns
X_train = X_train.astype(float)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = col)

print(X_train.head())

col = X_test.columns
X_test = X_test.astype(float)
scaler = StandardScaler()
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns = col)


# In[88]:


# TBD PCA


# Discard polynomail features due to lack of data samples comparing number of features

# In[89]:


from sklearn.model_selection import train_test_split


# In[90]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

poly = PolynomialFeatures(degree = 2).fit(X_train)
X_train_poly = poly.transform(X_train)
X_test_poly = poly.transform(X_test)

print(X_train_poly.shape)

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_train_poly, y_train, test_size = 0.3, random_state = 42)

# Apply Ridge (L2) to polynomial features
ridge = Ridge(alpha = 1000).fit(X_train_t, y_train_t)

print('Score of Polynomial Ridge on training set', ridge.score(X_train_t, y_train_t))
print('Score of Polynomial Ridge on test set', ridge.score(X_test_t, y_test_t))


# In[91]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_train, y_train, test_size = 0.3, random_state = 42)

print(X_train_t.shape)
print(X_test_t.shape)



from sklearn.linear_model import LinearRegression



#print(X_train.dtypes)

#X_train.to_csv('X_train_final.csv')
#clf = LinearRegression()
#lr = clf.fit(X_train_t, y_train_t)
#print('Score of Linear Regression ', lr.score(X_test_t, y_test_t))

# TBD Try different alpha values and plot
ridge = Ridge(alpha = 1200).fit(X_train_t, y_train_t)
print('Score of Ridge on training set', ridge.score(X_train_t, y_train_t))
print('Score of Ridge on test set', ridge.score(X_test_t, y_test_t))


# Implement KNN. KNN is simple with only one parameter for tuning (k) so often be used for baseline for ML computing. However, it's not good for this case as KNN does not perform well on datasets with many features.

# In[92]:


from sklearn.neighbors import KNeighborsClassifier 

clf = KNeighborsClassifier(n_neighbors = 100)
knn = clf.fit(X_train_t, y_train_t)
print('Score of knn on training set', knn.score(X_train_t, y_train_t))
print('Score of knn on test set', knn.score(X_test_t, y_test_t))


# As expected, the performace result is poor for KNN. Therefore, it won't be applied and just for reference here.

# Implement Tree algorithms

# In[93]:


print(X_train_t.shape)
print(X_train_t.head())


# In[94]:


from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(max_depth = 20, random_state = 42, n_estimators = 100)
rf = clf.fit(X_train_t, y_train_t)
print('Score of Kernel Random Forest on training set', rf.score(X_train_t, y_train_t))
print('Score of Kernel Random Forest on test set', rf.score(X_test_t, y_test_t))


# In[97]:


# Compare predictions on test samples
pred = rf.predict(X_test_t)
print(pred.shape)
print(pred)

print(y_test_t)


# In[101]:


pred = rf.predict(X_test)
np.savetxt('D:\ML\Kaggle projects\House Prices Advanced Regression Techniques\predict.csv', pred, delimiter=',')
#pred.tofile('D:\ML\Kaggle projects\House Prices Advanced Regression Techniques\predict.csv')


# Implement Kernel SVM (Support Vector Machine). Normally it performs much better when features are scaled and when there are many features.

# In[96]:


from sklearn.svm import SVR

svm = SVR(kernel = 'rbf', C=1, gamma = 1).fit(X_train_t, y_train_t)
print('Score of Kernel SVM on training set', svm.score(X_train_t, y_train_t))
print('Score of Kernel SVM on test set', svm.score(X_test_t, y_test_t))


# In[ ]:




