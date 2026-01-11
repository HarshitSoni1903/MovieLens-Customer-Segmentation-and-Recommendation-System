from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, collect_list, avg, lit, udf, row_number, count
from pyspark.sql.types import IntegerType, ArrayType
from pyspark.sql.window import Window

TOP_K = 100

def get_true_items(df, active_users, top_k=TOP_K, exclude_items=None):

    w = Window.partitionBy("userId").orderBy(col("timestamp").desc())
    df = df.withColumn("rn", row_number().over(w)).filter(col("rn") <= top_k)

    if exclude_items is not None:
        df = df.filter(~col("movieId").isin(exclude_items))

    return df.join(active_users, "userId") \
             .groupBy("userId") \
             .agg(collect_list("movieId").alias("trueItems"))



def evaluate_ranking(preds_and_labels, model_name="Model"):
    metrics = RankingMetrics(preds_and_labels)
    print(f"{model_name} MAP:", metrics.meanAveragePrecision)
    print(f"{model_name} NDCG@{TOP_K}:", metrics.ndcgAt(TOP_K))
    print(f"{model_name} Precision@{TOP_K}:", metrics.precisionAt(TOP_K))
    print(f"{model_name} Recall@{TOP_K}:", metrics.recallAt(TOP_K))


def popularity_pipeline(train_df, test_df):
    from pyspark.sql.functions import col, avg, count, array, lit, collect_list, row_number
    from pyspark.sql.types import IntegerType, ArrayType
    from pyspark.sql.window import Window
    from pyspark.mllib.evaluation import RankingMetrics

    TOP_K = 100

    train_users = train_df.select("userId").distinct()
    train_movies = train_df.select("movieId").distinct()

    test_df = test_df.join(train_users, "userId").join(train_movies, "movieId")
    test_users = test_df.select("userId").distinct()

    top_movies = train_df.groupBy("movieId") \
        .agg(count("*").alias("count"), avg("rating").alias("avg_rating")) \
        .orderBy(col("count").desc(), col("avg_rating").desc()) \
        .limit(TOP_K)
    top_movie_ids = [row["movieId"] for row in top_movies.collect()]
    top_movie_set = set(top_movie_ids)

    top_k_array = array(*[lit(m) for m in top_movie_ids])
    recommendations = test_users.withColumn("recommendations", top_k_array)

    true_items = get_true_items(test_df, test_users)

    preds_and_labels = recommendations.join(true_items, "userId").rdd.map(
        lambda row: (
            row.recommendations,
            [movie for movie in row.trueItems if movie in top_movie_set]
        )
    ).filter(lambda row: len(row[1]) > 0)

    count_eval_users = preds_and_labels.count()
    print(f"\nEvaluating Popularity Model on {count_eval_users} users (after filtering)")

    if count_eval_users == 0:
        print("\nNo valid users to evaluate.")
        return

    metrics = RankingMetrics(preds_and_labels)
    print(f"Popularity MAP: {metrics.meanAveragePrecision:.4f}")
    print(f"Popularity NDCG@{TOP_K}: {metrics.ndcgAt(TOP_K):.4f}")
    print(f"Popularity Precision@{TOP_K}: {metrics.precisionAt(TOP_K):.4f}")
    print(f"Popularity Recall@{TOP_K}: {metrics.recallAt(TOP_K):.4f}")

    hit_rate = preds_and_labels.map(lambda x: int(bool(set(x[0]) & set(x[1])))).mean()
    print(f"Popularity Hit Rate@{TOP_K}: {hit_rate:.3f}")



def main():
    spark = SparkSession.builder \
    .appName("ALS_Evaluation_Corrected") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

    ratings_train = spark.read.parquet("data/ratings_train.parquet")
    ratings_test = spark.read.parquet("data/ratings_test.parquet")

    print("\nRunning Popularity Baseline...")
    popularity_pipeline(ratings_train, ratings_test)

    spark.stop()

if __name__ == "__main__":
    main()
