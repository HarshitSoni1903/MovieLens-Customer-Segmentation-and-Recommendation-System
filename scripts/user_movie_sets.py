import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set

id = sys.argv[1] if len(sys.argv) > 1 else "hs5666"
spark = SparkSession.builder.appName("UserMovieSets").getOrCreate()

ratings = spark.read.option("header", True).csv(f"ratings.csv")
ratings = ratings.selectExpr(
    "cast(userId as int) userId", 
    "cast(movieId as int) movieId"
)

user_movie_sets = ratings.groupBy("userId").agg(collect_set("movieId").alias("movieIds"))
user_movie_sets = user_movie_sets.filter(size("movieIds") >= 5)

user_movie_sets.write.mode("overwrite").parquet(f"data/user_movie_sets_parquet/")

spark.stop()
