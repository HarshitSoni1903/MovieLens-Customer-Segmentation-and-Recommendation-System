import pandas as pd
from pathlib import Path
from collections import defaultdict

print("Started splitting process")

Path("data/").mkdir(parents=True, exist_ok=True)

df = pd.read_csv('ratings.csv')

user_counts = df['userId'].value_counts()
eligible_users = user_counts[user_counts >= 5].index
df = df[df['userId'].isin(eligible_users)]

user_ratings = defaultdict(list)
for row in df.itertuples(index=False):
    user_ratings[row.userId].append((row.movieId, row.rating, row.timestamp))

train_rows, val_rows, test_rows = [], [], []

for user, ratings in user_ratings.items():
    ratings.sort(key=lambda x: x[2])

    train = ratings[:-2]
    val = [ratings[-2]]
    test = [ratings[-1]]

    for movieId, rating, timestamp in train:
        train_rows.append([user, movieId, rating, timestamp])
    for movieId, rating, timestamp in val:
        val_rows.append([user, movieId, rating, timestamp])
    for movieId, rating, timestamp in test:
        test_rows.append([user, movieId, rating, timestamp])

train_df = pd.DataFrame(train_rows, columns=['userId', 'movieId', 'rating', 'timestamp'])
val_df = pd.DataFrame(val_rows, columns=['userId', 'movieId', 'rating', 'timestamp'])
test_df = pd.DataFrame(test_rows, columns=['userId', 'movieId', 'rating', 'timestamp'])

train_df.to_parquet('data/ratings_train.parquet', index=False)
val_df.to_parquet('data/ratings_val.parquet', index=False)
test_df.to_parquet('data/ratings_test.parquet', index=False)

print("Splitting complete.")
print(f"Users processed: {len(user_ratings)}")
print(f"Train ratings: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
print("Files saved as Parquet.")
