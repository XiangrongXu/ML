from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import split, explode, regexp_replace, udf, col
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType, MapType, StringType

def get_spark_session():
    spark = SparkSession.builder\
        .appName("xxr")\
        .master("local[*]")\
        .getOrCreate()
    return spark

def get_dataframe_from_csv(spark:SparkSession, file_name: str, dir="../files/csv_files/", header:bool=True, seperate:str=",", infer_schema:bool=True):
    file_name = file_name if file_name.endswith(".csv") else (file_name + ".csv")
    path = dir + file_name
    df = spark.read\
        .option("header", header)\
        .option("inferSchema", infer_schema)\
        .option("sep", seperate)\
        .csv(path)
    return df

def get_dataframe_from_db(spark:SparkSession, table_name:str, user:str, password:str):
    spark.read.jdbc("jdbc:mysql://127.0.0.1:3306/patest", table_name, properties={"user": user, "password": password})

def udf_array_to_map(array):
    if array is None:
        return array
    return dict((i, v) for i, v in enumerate(array))

def generate_idx_for_df(df:DataFrame, col_name:str, col_schema):
    idx_udf = udf(lambda x: udf_array_to_map(x), MapType(IntegerType(), col_schema, True))
    df = df.withColumn("map", idx_udf(col(col_name)))
    df = df.select("problem_type", "user_id", "oms_protected", "problem_id", "create_at", explode("map").alias("item_id", "answer"))
    return df

def get_problem_submission(spark:SparkSession):
    submission = get_dataframe_from_csv(spark, "format_normal_submission")
    problem_set = get_dataframe_from_csv(spark, "problem_set")
    problem_set_problem = get_dataframe_from_csv(spark, "problem_set_problem")
    problem_submission = submission\
        .join(problem_set_problem, submission.problem_set_problem_id == problem_set_problem.id, "inner")\
        .join(problem_set, problem_set_problem.problem_id == problem_set.id, "inner")\
        .select(submission.problem_type, submission.result, submission.user_id, problem_set.oms_protected, problem_set_problem.problem_id)
    problem_submission = problem_submission.withColumn("result", regexp_replace(regexp_replace("result", "\\[", ""), "\\]", ""))
    problem_submission = problem_submission.withColumn("result", split("result", ", "))
    col_schema = StringType()
    problem_submission = generate_idx_for_df(problem_submission, "result", col_schema)
    return problem_submission

spark = get_spark_session()
# problem_submission = get_problem_submission(spark)
# problem_submission.show()
df:DataFrame = get_dataframe_from_db(spark, "normal_problem_submission", "root", "RW0917jqw")
df.printSchema()
df.show()
