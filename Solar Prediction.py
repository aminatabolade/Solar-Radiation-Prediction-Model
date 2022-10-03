#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries and Model

# In[243]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[244]:


# Model from Scikit-Learn
from sklearn.ensemble import RandomForestRegressor

# Model Evaluations and tools
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#Saving and loading model
from joblib import dump, load


# ### Exploratory Data Analysis

# In[245]:


sp = pd.read_csv("SolarPrediction.csv")


# In[246]:


sp.head(10)


# In[247]:


sp.info()


# In[248]:


len(sp), sp.shape


# In[249]:


sp.isna().sum()


# In[250]:


sp.describe()


# In[251]:


fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Feature points')

# Temperature
sns.lineplot(ax=axes[0], x=sp['Temperature'].values, y=sp['Radiation'].values)
axes[0].set(xlabel='Temperature', ylabel = 'Radiation')
axes[0].set_title(sp['Temperature'].name)
# Humidity
sns.lineplot(ax=axes[1], x=sp['Humidity'].values, y=sp['Radiation'].values)
axes[1].set(xlabel='Humidity', ylabel = 'Radiation')
axes[1].set_title(sp['Humidity'].name)

# Pressure
sns.lineplot(ax=axes[2],  x=sp['Pressure'].values, y=sp['Radiation'].values)
axes[2].set(xlabel='Pressure', ylabel = 'Radiation')
axes[2].set_title(sp['Pressure'].name);


# In[252]:


import datetime

#Date

sp['Year'] = pd.DatetimeIndex(sp['Data']).year
sp['Month'] = pd.DatetimeIndex(sp['Data']).month
sp['Day'] = pd.DatetimeIndex(sp['Data']).day
sp.head()

#Time 

sp['Hour'] = pd.DatetimeIndex(sp['Time']).hour
sp['Minute'] = pd.DatetimeIndex(sp['Time']).minute
sp['Second'] = pd.DatetimeIndex(sp['Time']).second

sp.head()


sp['SunPerDay'] = pd.DatetimeIndex(sp['TimeSunSet']) - pd.DatetimeIndex(sp['TimeSunRise'])
sp.head()

sp['SunPerDayHours'] = pd.DatetimeIndex(sp['TimeSunSet']).hour - pd.DatetimeIndex(sp['TimeSunRise']).hour 


# In[253]:


sp.drop('Time', axis = 1, inplace=True)
sp.drop('Data', axis = 1, inplace=True)
sp.drop('TimeSunRise', axis = 1, inplace=True)
sp.drop('TimeSunSet', axis = 1, inplace=True)
sp.drop('SunPerDay', axis = 1, inplace=True)

sp.head()


# In[254]:


sp.corr()


# In[255]:


fig = plt.figure(figsize=(20,10))
fig.suptitle('Feature Correlation', fontsize=18)
sns.heatmap(sp.corr(), annot=True, cmap='RdBu', center=0);


# In[256]:


sp.drop('UNIXTime', axis = 1, inplace=True)
sp.drop('Year', axis = 1, inplace=True)


# In[257]:


sp.head()


# In[258]:


fig = plt.figure(figsize=(20,10))
fig.suptitle('Feature Correlation', fontsize=18)
sns.heatmap(sp.corr(), annot=True, cmap='RdBu', center=0);


# In[259]:


fig2 = plt.figure(figsize=(15,5))
sns.barplot(x=sp['Temperature'],y=sp['Radiation']);


# In[260]:


fig3 = plt.figure(figsize=(15,5))
sns.barplot(x=sp['Humidity'],y=sp['Radiation']);


# In[261]:


fig3 = plt.figure(figsize=(15,5))
sns.barplot(x=sp['Pressure'],y=sp['Radiation']);


# In[262]:


fig3 = plt.figure(figsize=(15,5))
sns.barplot(x=sp['Month'],y=sp['Radiation']);


# In[263]:


fig4 = plt.figure(figsize=(15,5))
sns.kdeplot(x=sp['WindDirection(Degrees)'],y=sp['Radiation']);


# In[264]:


fig4 = plt.figure(figsize=(15,5))
sns.barplot(x=sp['Speed'],y=sp['Radiation']);


# In[265]:


fig4 = plt.figure(figsize=(15,5))
sns.barplot(x=sp['SunPerDayHours'],y=sp['Radiation']);


# In[266]:


fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Radiation Time')

sns.lineplot(ax=axes[0], x=sp['Hour'].values, y=sp['Radiation'].values)
axes[0].set(xlabel='Hour', ylabel = 'Radiation')
axes[0].set_title(sp['Hour'].name)

sns.lineplot(ax=axes[1], x=sp['Minute'].values, y=sp['Radiation'].values)
axes[1].set(xlabel='Minute', ylabel = 'Radiation')
axes[1].set_title(sp['Minute'].name)


sns.lineplot(ax=axes[2],  x=sp['Second'].values, y=sp['Radiation'].values)
axes[2].set(xlabel='Second', ylabel = 'Radiation')
axes[2].set_title(sp['Second'].name);


# ## ML MODEL

# In[267]:


sp.head()


# In[268]:


#Random seed
np.random.seed(42)

#split the data
X = sp.drop('Radiation', axis=1)
y = sp['Radiation']


# In[269]:


X.head()


# In[270]:


y.head()


# In[271]:


#split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[272]:


model = RandomForestRegressor()
model.fit(X_train, y_train)


# In[273]:


#Making predictions
y_preds = model.predict(X_test)
y_preds[:20]


# In[274]:


df_preds = pd.DataFrame()
df_preds["Actual Value"] = y_test
df_preds["Predicted value"] = y_preds
df_preds


# In[275]:


df_preds.head(20)


# In[276]:


df = pd.DataFrame(data={"actual values":y_test,
                       "predicted values":y_preds})
df["differences"] = df["predicted values"] - df["actual values"]
df.head(20)


# In[277]:


r2_score(y_test, y_test)


# In[278]:


#Evaluating model's predictions
print ("Regression model metrics on test set")
print(f'R^2: {r2_score(y_test, y_preds)}')
print(f'MAE: {mean_absolute_error(y_test, y_preds)}')
print(f'MSE: {mean_squared_error(y_test, y_preds)}')


# ### Improving Model

# In[279]:


model.get_params()


# In[280]:


np.random.seed(42)

sp_shuffled = sp.sample(frac=1)
X =  sp_shuffled.drop(['Radiation'], axis=1)
y = sp_shuffled['Radiation']

train_split = round(0.7 * len(sp_shuffled)) 
valid_split = round(train_split + 0.15 * len(sp_shuffled)) 

X_train, y_train = X[:train_split], y[:train_split]
X_valid, y_valid = X[train_split:valid_split], y[train_split:valid_split]
X_test, y_test = X[valid_split:], y[valid_split:]

len(X_train), len(X_valid), len(X_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[281]:



grid ={"n_estimators":[10, 100, 200, 500, 1000, 1200],
      "max_depth":[None, 5,10,20,30],
      "max_features": ["auto", "sqrt"],
      "min_samples_split": [2,4,6],
      "min_samples_leaf": [1,2,4]}


# In[282]:


np.random.seed(42)

X = sp_shuffled.drop(['Radiation'], axis=1)
y = sp_shuffled['Radiation']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_jobs=1)

rs_model = RandomizedSearchCV(estimator=model,
                          param_distributions=grid,
                          n_iter = 10,
                          cv=5,
                          verbose=2)

rs_model.fit(X_train, y_train)


# In[283]:


rs_model.best_params_


# In[284]:


#make predictions with the best hyperparameters
rs_y_preds = rs_model.predict(X_test)

#evalute the predictions
print ("Regression model metrics on test set")
print(f'R^2: {r2_score(y_test, rs_y_preds)}')
print(f'MAE: {mean_absolute_error(y_test, rs_y_preds)}')
print(f'MSE: {mean_squared_error(y_test, rs_y_preds)}')


# In[285]:


# Save model to file
dump(model, filename="sp_random_forest_model.joblib")

