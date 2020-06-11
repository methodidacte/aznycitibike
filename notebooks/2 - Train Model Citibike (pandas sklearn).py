# Databricks notebook source
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble, linear_model
from sklearn.feature_selection import RFE
from sklearn.svm import SVR, LinearSVC

%matplotlib inline

# COMMAND ----------

dbutils.library.installPyPI("mlflow", extras="extras")
dbutils.library.installPyPI("joblib")

# COMMAND ----------

df = spark.sql("select * from citibikeny2")

# COMMAND ----------

dataset = df.select("*").sample(0.1).toPandas()
dataset.count()

# COMMAND ----------

dataset.isnull().sum()

# COMMAND ----------

dataset = dataset.fillna(method='ffill')
dataset.describe()

# COMMAND ----------

X = dataset[[x for x in dataset.columns if x!='trip_duration']]
y = dataset['trip_duration']

# COMMAND ----------

X = dataset[['distance_bwn_stations','Subscriber','male_gender','female_gender']]
y = dataset['trip_duration']

# COMMAND ----------

# MAGIC %md
# MAGIC Linear Regression

# COMMAND ----------

# split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# COMMAND ----------

# fit (train)
from sklearn import linear_model

linmodel = linear_model.LinearRegression(fit_intercept=True, normalize=True)
linmodel.fit(X_train, y_train)

# COMMAND ----------

# predict
y_predict = linmodel.predict(X_test)

# COMMAND ----------

# metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#from sklearn.metrics import explained_variance_score
from sklearn.metrics import median_absolute_error

print("MSE : ", mean_squared_error(y_test,y_predict))
print("R2 : ", r2_score(y_test,y_predict))
print("RMCE : ", np.sqrt(mean_squared_error(y_test,y_predict)))
print("MAE : ", median_absolute_error(y_test,y_predict))

# COMMAND ----------

# model export (pickle)
import joblib

joblib.dump(linmodel, open('linmodel.pkl','wb'))

# COMMAND ----------

# scatter plot des valeurs réelles vs prédites

# Display results
fig = plt.figure(1)

#plt.scatter(X_test['distance_bwn_stations'], y_test,  color='black')
#plt.plot(X_test['distance_bwn_stations'], y_predict, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

display(fig)

# COMMAND ----------

# Save figure
fig.savefig("obs_vs_predict.png")

# Close plot
plt.close(fig)

# COMMAND ----------

# cross validation 
from sklearn.model_selection import cross_validate

scores = cross_validate(linmodel, X_train, y_train, cv=3, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
print(scores['test_neg_mean_squared_error']) 

# COMMAND ----------

# MAGIC %md
# MAGIC Ridge Regression

# COMMAND ----------

dbutils.fs.rm('/mnt/nycitibike/MODELS/',recurse=True)

# COMMAND ----------

dbutils.fs.mkdirs('/mnt/nycitibike/MODELS/')

# COMMAND ----------

# MAGIC %fs ls /mnt/nycitibike/

# COMMAND ----------

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn

alpha = 1.0

with mlflow.start_run():
  ridgemodel = Ridge(alpha=1.0, normalize=True)
  ridgemodel.fit(X_train, y_train) 
  y_predict = ridgemodel.predict(X_test)
  
  mse = mean_squared_error(y_test,y_predict)
  r2 = r2_score(y_test,y_predict)
  rmse = np.sqrt(mean_squared_error(y_test,y_predict))
  mae = median_absolute_error(y_test,y_predict)

  # Log mlflow attributes for mlflow UI
  mlflow.log_param("alpha", alpha)
  mlflow.log_param("normalize", True)
  mlflow.log_metric("mse", mse)
  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("r2", r2)
  mlflow.log_metric("mae", mae)
	
  mlflow.sklearn.log_model(linmodel, "model")
  modelpath = "/mnt/nycitibike/MODELS/model-%f-%f" % (alpha, True)
  mlflow.sklearn.save_model(linmodel, modelpath)
  
  # Log artifacts (output files)
  #mlflow.log_artifact("obs_vs_predict.png")

# COMMAND ----------

print("MSE : ", mean_squared_error(y_test,y_predict))
print("R2 : ", r2_score(y_test,y_predict))
print("RMSE : ", np.sqrt(mean_squared_error(y_test,y_predict)))
print("MAE : ", median_absolute_error(y_test,y_predict))

# COMMAND ----------

# hyperparameters tuning (grid search)
from sklearn.model_selection import GridSearchCV

dico_param = {'alpha': [1e-3, 1e-2, 1e-1, 1]}
search_hyperp_ridge = GridSearchCV(Ridge(), dico_param, scoring='neg_mean_squared_error', cv = 5)
search_hyperp_ridge.fit(X_train, X_train)
search_hyperp_ridge.predict(X_test)

print(search_hyperp_ridge.best_params_)
print(search_hyperp_ridge.best_score_)

# COMMAND ----------

# model export (pickle)
import joblib

joblib.dump(search_hyperp_ridge, open('duration_ridge_model.pkl','wb'))

# COMMAND ----------

# MAGIC %fs ls file:/databricks/driver/duration_ridge_model.pkl

# COMMAND ----------

# unpickle and test
my_pickle_model = joblib.load('linmodel.pkl')
my_pickle_model.predict(X_test)