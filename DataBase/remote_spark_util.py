import os
from pyspark.sql.types import LongType
from pytispark.pytispark import TiContext
from pyspark.sql import SparkSession
from pyspark.sql import functions

def get_spark(master, pd, host):
    spark = SparkSession.builder \
        .appName("demo") \
        .master(master) \
        .config("spark.tispark.pd.addresses", pd) \
        .config("spark.jars", "tispark-assembly-2.3.16.jar") \
        .config("spark.driver.host", host) \
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=Asia/Shanghai") \
        .config("spark.executor.extraJavaOptions", "-Duser.timezone=Asia/Shanghai") \
        .config("spark.sql.extensions", "org.apache.spark.sql.TiExtensions") \
        .config("spark.tispark.request.isolation.level", "RC") \
        .config("spark.tispark.request.command.priority", "Low") \
        .config("spark.tispark.table.scan_concurrency", "256") \
        .config("spark.tispark.meta.reload_period_in_sec", "60") \
        .config("spark.sql.autoBroadcastJoinThreshold", "104857600") \
        .config("spark.cores.max", "10") \
        .config("spark.executor.memory", "2G") \
        .getOrCreate()
    ti = TiContext(spark)
    ti.tidbMapDatabase("patest")
    spark.sql("use patest")
    return spark

def tryit(x):
    return x + 1

spark = get_spark("spark://10.64.1.22:7077", "10.64.7.210:2379,10.64.7.211:2379,10.64.7.212:2379", "172.16.4.9")
df = spark.read.load("jfs:///jfs/normal_problem_submission.csv")
