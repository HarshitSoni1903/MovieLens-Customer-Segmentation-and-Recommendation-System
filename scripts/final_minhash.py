from datasketch import MinHash,MinHashLSH
import numpy as np
import sys
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import collect_set, size
import pandas as pd
import pickle
from pathlib import Path

spark = SparkSession.builder \
    .appName("MinHashWithDatasketch") \
    .config("spark.submit.pyFiles", "datasketch-1.6.5-py3-none-any.whl") \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()

df = spark.read.parquet("data/user_movie_sets_parquet/")
def compute_hashvalues(partition):
    for row in partition:
        uid = row['userId']
        movie_ids = row['movieIds']
        m = MinHash(num_perm=64)
        for movie in movie_ids:
            m.update(str(movie).encode('utf8'))
        yield Row(userId=uid, hashvalues=m.hashvalues.tolist())

hashed_rdd = df.rdd.mapPartitions(compute_hashvalues)
hashed_df = spark.createDataFrame(hashed_rdd)
hashed_df.write.mode("overwrite").parquet("data/minhash_signatures/")

df = spark.read.parquet("data/minhash_signatures/")
print("hash read")

#Approach 2
def reconstruct_minhash(partition):
    for row in partition:
        uid = row['userId']
        m = MinHash(num_perm=64)
        m.hashvalues = np.array(row['hashvalues'])
        yield (uid, m)

minhashes_rdd = df.rdd.mapPartitions(reconstruct_minhash)
minhashes_rdd = minhashes_rdd.persist()
minhashes = dict(minhashes_rdd.collect())
print("minhash recalculated")

lsh = MinHashLSH(threshold=0.5, num_perm=64)
for uid, m in minhashes.items():
    lsh.insert(str(uid), m)
print("lsh calculating")

seen_pairs = set()
results = []

print("calculating jaccardian")
for uid1, m1 in minhashes.items():
    for uid2 in lsh.query(m1):
        uid1, uid2 = sorted([int(uid1), int(uid2)])
        if uid1 == uid2 or (uid1, uid2) in seen_pairs:
            continue
        sim = m1.jaccard(minhashes[uid2])
        results.append((uid1, uid2, sim))
        seen_pairs.add((uid1, uid2))

print("Saving results")        
df_top = pd.DataFrame(results, columns=["user1", "user2", "jaccard"])
df_top = df_top.sort_values(by="jaccard", ascending=False).head(100)

Path("results").mkdir(exist_ok=True)
df_top.to_csv("results/top_100_pairs.csv", index=False)

spark.stop()
