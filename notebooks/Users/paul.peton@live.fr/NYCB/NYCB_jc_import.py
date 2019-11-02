# Databricks notebook source
param_dev = dbutils.widgets.get("wEnv")
# le paramètre est le répertoire contenant les fichiers sur le point de montage

# COMMAND ----------

# ajouter le schéma du nom de fichier en paramètre
filename = 'JC'

# COMMAND ----------

df_jc = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/mnt/"+param_dev+"/JC*")

# COMMAND ----------

df_jc = df_jc.withColumnRenamed('Trip Duration', 'tripduration')
df_jc = df_jc.withColumnRenamed('start time', 'starttime')
df_jc = df_jc.withColumnRenamed('stop time', 'stoptime')
df_jc = df_jc.withColumnRenamed('start station id', 'startstationid')
df_jc = df_jc.withColumnRenamed('start station name', 'startstationname')
df_jc = df_jc.withColumnRenamed('start station longitude', 'startstationlongitude')
df_jc = df_jc.withColumnRenamed('start station latitude', 'startstationlatitude')
df_jc = df_jc.withColumnRenamed('end station id', 'endstationid')
df_jc = df_jc.withColumnRenamed('end station name', 'endstationname')
df_jc = df_jc.withColumnRenamed('end station longitude', 'endstationlongitude')
df_jc = df_jc.withColumnRenamed('end station latitude', 'endstationlatitude')
df_jc = df_jc.withColumnRenamed('bike id', 'bikeid')
df_jc = df_jc.withColumnRenamed('user type', 'usertype')
df_jc = df_jc.withColumnRenamed('birth year', 'birthyear')

# COMMAND ----------

df_jc.write.mode('overwrite').format("parquet").saveAsTable("jc_citibike")

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW DATABASE

# COMMAND ----------

print(df_jc.count())

# COMMAND ----------

import json

dbutils.notebook.exit(
  json.dumps({
    "status": "OK",
    "table": "jc_citibike"
  })
)

# COMMAND ----------

dbutils.notebook.exit("end of data collection (initial notebook)")