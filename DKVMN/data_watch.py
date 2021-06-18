import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from local_spark import get_spark_session
from pyspark.sql.functions import array, udf, coalesce

spark_session = get_spark_session()
df = spark_session.read.option("header", True) \
    .option("inferSchema", True) \
        .csv("D:/files/csv_files/train_data.csv/*.csv") \
            .select("list")
df.coalesce(1).write.option("header", False).mode("overwrite").csv("D:/files/csv_files/train_data")


# ar = df.toPandas().values
# plt.hist(ar, bins=6, facecolor="blue", edgecolor="black", alpha=0.7)
# plt.show()
# spark_session.stop()