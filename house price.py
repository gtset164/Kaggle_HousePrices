#!/usr/bin/env python
# coding: utf-8

# House Prices: Advanced Regression Techniques
# 
# This project is a compeition in Kaggle with initially 79 features for the residential homes in Ames, Iowa. The goal of this challenge is to predict the final price of each home.
# As usual, data exploration, feature engineering are applied, then several machine learning models are built to predict the house price in the test set.

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from scipy.stats import skew
get_ipython().run_line_magic('matplotlib', 'inline')

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# Functions which are used in this project. They can also be reused in other projects.

# In[34]:


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

#Function to measure accuracy.
def rmlse(val, target):
    return np.sqrt(np.sum(((np.log1p(val) - np.log1p(np.expm1(target)))**2) / len(target)))


# Data Exploration

# In[35]:


print(df_train.shape)
print(df_test.shape)


# In[36]:


df_train.info()


# In[37]:


df_train.describe(include='all')


# In[38]:


df_train.head()


# In[39]:


# Heatmap on training set including sales price to observe the Pearson correlations.
corr = df_train.corr(method = 'pearson')
print(corr.head())

plt.figure(figsize=(12, 12))
sns.heatmap(corr, vmax=1, vmin=-1, cmap="YlGnBu" )


# Note:
# 1. Large negative correlations are equaly important as large positive values in machine learning.
# 2. Pearson correlations can only represent numerical values including categorical ordinal feature values.
# 3. Pearson correlations only measure linear correlations. It does not work well when there is non-linear relation between two features. Therefore, there might be other relations even though the correlation values are low between some features.

# 
# The figure below can also show that there is lower correlation between lot area size and sale price.

# In[40]:


jitter = 150
x = df_train['LotArea']
y = df_train['SalePrice'] + np.random.uniform(-jitter, jitter, len(df_train))
#y = df_train['SalePrice']

plt.plot(x, y, 'o', markersize=5, alpha=0.1, color = 'Blue')
# Limit Sale Price range within 50000 as only limited lot area data greater than 20000 better easier observation.
plt.xlim(0,20000)
# Same reason to limit sale price
plt.ylim(0,400000)
plt.xlabel=('Lot Area')
plt.ylabel=('Sale Price')
plt.show()

print(corr.loc['LotArea', 'SalePrice'])


# The figure below can also show that there is higher correlation between built year and sale price.

# In[41]:


jitter = 150
x = df_train['YearBuilt']
y = df_train['SalePrice'] + np.random.uniform(-jitter, jitter, len(df_train))

plt.plot(x, y, 'o', markersize=5, alpha=0.08, color = 'Green')
plt.xlim(1900,2010)
plt.ylim(0, 500000)

plt.xlabel=('YearBuilt')
plt.ylabel=('Sale Price')
plt.show()

print(corr.loc['YearBuilt', 'SalePrice'])


# In[42]:


#It seems that SalePrice is skewered, so it needs to be transformed. Will do it later.
sns.distplot(df_train['SalePrice'], kde=False, color='c', hist_kws={'alpha': 0.9})


# In[43]:


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

# In[44]:


df_nu = df_train.isnull().sum()

# Only list the features with null values for later processing
print(df_nu[df_nu > 0])


# Check the null values in test set

# In[45]:


df_nu = df_test.isnull().sum()

# Only list the features with null values for later processing
print(df_nu[df_nu > 0])


# There are 4 features 'Alley','PoolQC','Fence','MiscFeature' with almost all null values. Merge training and test data first then remove the 4 features.

# In[46]:


# Merge training and test set to process feature engineering together
df_all = concat_df(df_train, df_test)
print(df_all.shape)


# In[47]:


df_all.drop(['Alley','PoolQC','Fence','MiscFeature'], axis = 1, inplace= True)


# Initially fill in the null values for all features with null values above.

# In[48]:


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

# In[49]:


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

# In[50]:


cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 'GarageFinish', 'MoSold', 'SaleType', 'SaleCondition']
onehot = onehot_features(df_all, cols)

# Drop the original features
df_all.drop(cols, axis = 1, inplace = True)

print(type(onehot))
print(onehot.shape)
print(df_all.shape)
df_all.info()


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

# In[51]:


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

# The following features are still type object but they should be numbers. Convert them to int64. 
df_all['BsmtFinSF1'] = df_all['BsmtFinSF1'].astype(int)
df_all['BsmtFinSF2'] = df_all['BsmtFinSF2'].astype(int)
df_all['BsmtFullBath'] = df_all['BsmtFinSF2'].astype(int)
df_all['BsmtHalfBath'] = df_all['BsmtFinSF2'].astype(int)
df_all['GarageCars'] = df_all['BsmtFinSF2'].astype(int)
df_all['MasVnrArea'] = df_all['BsmtFinSF2'].astype(int)


# Transform skewed features.

# In[52]:


for col in df_all.columns:
    if skew(df_all[col]) > 0.75:
        df_all[col] = np.log1p(df_all[col])

y_train = np.log1p(y_train)


#  Merge dummy features and original data set.

# In[53]:


df_all = pd.concat([df_all, onehot], axis=1)
print(df_all.shape)


# In[54]:


df_nu = df_all.isnull().sum()

# Make sure if there is no null value
print(df_nu[df_nu>0])


# In[55]:


# Check the data set dimensions so far again.
print(df_train.shape)
print(df_all.shape)
X_train, X_test = divide_df(df_all, df_train.shape[0])
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)


# Scaling features

# In[56]:


col = X_train.columns
X_train = X_train.astype(float)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = col)

print(X_train.head())

col = X_test.columns
X_test = X_test.astype(float)
scaler = StandardScaler()
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns = col)


# PCA transformation

# As the heapmap shows most of the correlations between two features are low, it's possible that PCA transformation/ decomposition
# cannot imrove performance, or even loss some information after PCA. Will still try the result to compare anyway.

# In[57]:


pca = PCA(200, whiten=True, random_state=42).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train.shape)
print(type(X_train_pca))


# Discard polynomail features due to lack of data samples comparing number of features

# In[58]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

poly = PolynomialFeatures(degree = 2).fit(X_train)
X_train_poly = poly.transform(X_train)
X_test_poly = poly.transform(X_test)

print(X_train_poly.shape)

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_train_poly, y_train, test_size = 0.3, random_state = 42)

# Apply Ridge (L2) to polynomial features
ridge = Ridge(alpha = 1000).fit(X_train_t, y_train_t)

pred = np.expm1(ridge.predict(X_test_t))
print(rmlse(pred, y_test_t))


# In[59]:


X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)

# For PCA features
X_train_t_pca, X_test_t_pca, y_train_t_pca, y_test_t_pca = train_test_split(X_train_pca, y_train, test_size = 0.25, random_state = 42)


# Linear Regression model with L2 regularization

# In[60]:


from sklearn.linear_model import LinearRegression

ridge = Ridge(alpha = 1200).fit(X_train_t, y_train_t)
pred = np.expm1(ridge.predict(X_test_t))
print(rmlse(pred, y_test_t))


# Implement Tree algorithms - Random Forest

# Without PCA

# In[61]:


from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(max_depth = 20, random_state = 42, n_estimators = 100)
rf = clf.fit(X_train_t, y_train_t)
pred = np.expm1(rf.predict(X_test_t))
print(rmlse(pred, y_test_t))


# With PCA

# In[62]:


clf = RandomForestRegressor(max_depth = 20, random_state = 42, n_estimators = 100)
rf = clf.fit(X_train_t_pca, y_train_t_pca)
pred = np.expm1(rf.predict(X_test_t_pca))
print(rmlse(pred, y_test_t_pca))


# The result for PCA meet the earlier guess - PCA does not help to improve the performance in this project, so will not apply PCA.

# 
# XG-Boost algorithm

# In[63]:


import xgboost as xgb

clf = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.1)
xgb = clf.fit(X_train_t, y_train_t)
#score=cross_val_score(clf, X_train, y_train, cv=k_fold)

pred = np.expm1(xgb.predict(X_test_t))
rmlse(pred, y_test_t)


# In[64]:


pred = np.expm1(xgb.predict(X_test))
np.savetxt('D:\ML\Kaggle projects\House Prices Advanced Regression Techniques\predict.csv', pred)


# The score (RMSLE) based on XG-Boost in this file is 0.13515 on Kaggle.
