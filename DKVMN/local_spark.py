from pyspark.sql.session import SparkSession

def get_spark_session():
    spark = SparkSession.builder\
        .appName("xxr")\
        .master("local[*]")\
        .getOrCreate()
    return spark