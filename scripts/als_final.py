from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, collect_list, avg, lit, udf, row_number, count
from pyspark.sql.types import IntegerType, ArrayType
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics

TOP_K = 100

def get_true_items(df, active_users, top_k=TOP_K):
    if "timestamp" in df.columns:
        w = Window.partitionBy("userId").orderBy(col("timestamp").desc())
        df = df.withColumn("rn", row_number().over(w)).filter(col("rn") <= top_k)
    return df.join(active_users, "userId") \
             .groupBy("userId") \
             .agg(collect_list("movieId").alias("trueItems"))

def evaluate_ranking(preds_and_labels, model_name="Model"):
    preds_and_labels = preds_and_labels.filter(lambda row: len(row[1]) > 0)

    count = preds_and_labels.count()
    print(f"Predictions and labels count: {count}")
    if count == 0:
        print("No valid pairs to evaluate.")
        return

    metrics = RankingMetrics(preds_and_labels)
    print(f"{model_name} MAP: {metrics.meanAveragePrecision}")
    print(f"{model_name} NDCG@{TOP_K}: {metrics.ndcgAt(TOP_K)}")
    print(f"{model_name} Precision@{TOP_K}: {metrics.precisionAt(TOP_K)}")
    print(f"{model_name} Recall@{TOP_K}: {metrics.recallAt(TOP_K)}")

    # Hit Rate
    hit_rate = preds_and_labels.map(lambda x: int(bool(set(x[0]) & set(x[1])))).mean()
    print(f"{model_name} Hit Rate@{TOP_K}: {hit_rate:.3f}")

def tune_als(train_df, validation_df, test_df):
    ranks = [10, 20, 50]
    reg_params = [0.01, 0.05, 0.1]
    best_score = -1
    best_model = None
    best_params = None

    valid_users = validation_df.select("userId").distinct().join(train_df.select("userId").distinct(), "userId")
    valid_movies = validation_df.select("movieId").distinct().join(train_df.select("movieId").distinct(), "movieId")
    validation_df = validation_df.join(valid_users, "userId").join(valid_movies, "movieId")

    test_users = test_df.select("userId").distinct().join(train_df.select("userId").distinct(), "userId")
    test_movies = test_df.select("movieId").distinct().join(train_df.select("movieId").distinct(), "movieId")
    test_df = test_df.join(test_users, "userId").join(test_movies, "movieId")

    for rank in ranks:
        for reg in reg_params:
            print(f"\nTraining ALS with rank={rank}, regParam={reg}")
            als = ALS(
                userCol="userId",
                itemCol="movieId",
                ratingCol="rating",
                rank=rank,
                regParam=reg,
                implicitPrefs=True,
                alpha=15.0,
                coldStartStrategy="drop",
                nonnegative=True
            )
            model = als.fit(train_df)

            active_users = validation_df.select("userId").distinct()
            recommendations = model.recommendForUserSubset(active_users, TOP_K).cache()
            true_items = get_true_items(validation_df, active_users).cache()

            preds_and_labels = recommendations.join(true_items, "userId").rdd.map(
                lambda row: ([r["movieId"] for r in row.recommendations], row.trueItems)
            )

            print(f"Evaluating on validation set")
            metrics = RankingMetrics(preds_and_labels.filter(lambda x: len(x[1]) > 0))
            map_score = metrics.meanAveragePrecision
            print(f"MAP: {map_score:.4f}")

            if map_score > best_score:
                best_score = map_score
                best_model = model
                best_params = (rank, reg)

    print(f"\n Best ALS: rank={best_params[0]}, regParam={best_params[1]}, MAP={best_score:.4f}")

    active_users = test_df.select("userId").distinct()
    recommendations = best_model.recommendForUserSubset(active_users, TOP_K).cache()
    true_items = get_true_items(test_df, active_users).cache()

    preds_and_labels = recommendations.join(true_items, "userId").rdd.map(
        lambda row: ([r["movieId"] for r in row.recommendations], row.trueItems)
    )

    evaluate_ranking(preds_and_labels, "ALS (Tuned)")

def main():
    spark = SparkSession.builder \
        .appName("ALS_Evaluation_Corrected") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

    ratings_train = spark.read.parquet("data/ratings_train.parquet")
    ratings_val = spark.read.parquet("data/ratings_val.parquet")
    ratings_test = spark.read.parquet("data/ratings_test.parquet")

    for df in [ratings_train, ratings_val, ratings_test]:
        df = df.withColumn("userId", col("userId").cast("int"))
        df = df.withColumn("movieId", col("movieId").cast("int"))

    print("Running ALS")
    tune_als(ratings_train, ratings_val, ratings_test)

    spark.stop()

if __name__ == "__main__":
    main()
