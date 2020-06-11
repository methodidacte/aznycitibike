# Databricks notebook source
# MAGIC %md
# MAGIC <p>
# MAGIC   <a href="$./Clean Data Citibike CSV. (1)">Clener code at this location</a>
# MAGIC </p>
# MAGIC <p>
# MAGIC   <a href="$./Random Forests">Random Forest regression</a>
# MAGIC </p>
# MAGIC <p>
# MAGIC   <a href="$./Decision tree regression">Decision tree regression</a>
# MAGIC </p>
# MAGIC <p>
# MAGIC   <a href="$./Linear Regression">Linear Regression</a>
# MAGIC </p>
# MAGIC <p>
# MAGIC   <a href="$./Gradient-boosted tree regression">Gradient-boosted tree regression</a>
# MAGIC </p>

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

import mlflow
import mlflow.mleap
import mleap.pyspark

# COMMAND ----------

#df = spark.sql("SELECT trip_duration,start_station_id,birth_year,unknown_gender,male_gender,female_gender,Subscriber,Customer,real_distance,((real_distance / trip_duration)* 3.6) as vitesse, DATE(start_time) as date,HOUR(start_time) as hour FROM CitibikeNY NATURAL JOIN citybike_station_distance")

df = spark.sql("SELECT trip_duration,start_station_id,birth_year,unknown_gender,male_gender,female_gender,Subscriber,Customer,distance_bwn_stations,(((distance_bwn_stations * 1000) / trip_duration)* 3.6) as vitesse, DATE(start_time) as date,HOUR(start_time) as hour FROM CitibikeNY2 NATURAL JOIN citybike_station_distance")

# COMMAND ----------

df = df.filter((df.vitesse>13) & (df.vitesse<32))

# COMMAND ----------

display(df)

# COMMAND ----------

df2=spark.sql("SELECT DATE(Date) as date,Day,Day_Name,Day_of_month,Day_of_week,Month,Month_Name,Month_Number,Quarter_Number,Week_of_month,Year_Month,Year FROM calandar_2013_2020")

# COMMAND ----------

df3=df.join(df2,["date"],"left")

# COMMAND ----------

#df = spark.sql("select * from CitibikeNY")
df = spark.sql("SELECT trip_duration,start_station_id,end_station_id,birth_year,unknown_gender,male_gender,female_gender,Subscriber,Customer,distance_bwn_stations,real_distance FROM CitibikeNY NATURAL JOIN citybike_station_distance")

# COMMAND ----------

df= df3.drop("date","Day","Day_Name","Month","Month_Name","Year_Month","vitesse")

# COMMAND ----------

# create features vector
feature_columns = df.columns[1:]

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=feature_columns,outputCol="features")
df_2 = assembler.transform(df)

# COMMAND ----------

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 6 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=10).fit(df_2)

# COMMAND ----------

# Train a LinearRegression model.
lr = LinearRegression(featuresCol="features", labelCol="trip_duration")

# Chain indexer and tree in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, lr])

# COMMAND ----------

trainingData, testData = df_2.randomSplit([0.75, 0.25])

trainingData.cache()
testData.cache()

# COMMAND ----------

grid = ParamGridBuilder() \
  .addGrid(lr.maxIter, [1, 3, 5, 7, 9]) \
  .addGrid(lr.regParam, [1, 0.5, 0.25, 0.1]) \
  .addGrid(lr.elasticNetParam, [1, 0.5, 0.25]) \
  .build()

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol="trip_duration", predictionCol="prediction", metricName="rmse")


# COMMAND ----------

tuning = TrainValidationSplit(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=grid)

# COMMAND ----------

# Best practice: Use explicit MLflow run management via mlflow.start_run().
# This allows you to log custom tags, metrics, and artifacts more easily.
with mlflow.start_run(run_name='Linear_Regression'):
  # This fit() call will log to MLflow under the hood.
  tunedModel = tuning.fit(trainingData)

  # We log a custom tag, a custom metric, and the best model to the main run.
  mlflow.set_tag('Citibike_training', 'Data_team')
  
  rmse = evaluator.evaluate(tunedModel.transform(testData), {evaluator.metricName: "rmse"})
  r2 = evaluator.evaluate(tunedModel.transform(testData), {evaluator.metricName: "r2"})
  mae = evaluator.evaluate(tunedModel.transform(testData), {evaluator.metricName: "mae"})
  mse = evaluator.evaluate(tunedModel.transform(testData), {evaluator.metricName: "mse"})
  
  mlflow.log_metric('rmse', rmse)
  mlflow.log_metric('r2', r2)
  mlflow.log_metric('mae', mae)
  mlflow.log_metric('mse', mse)
  
  print("Tuned model r2: {}".format(r2))
  print("Tuned model rmse: {}".format(rmse))
  print("Tuned model mae: {}".format(mae))
  print("Tuned model mse: {}".format(mse))
  
  mlflow.end_run()

# COMMAND ----------

from mlflow import spark
mlflow.spark.save_model(tunedModel.bestModel, "spark-model")