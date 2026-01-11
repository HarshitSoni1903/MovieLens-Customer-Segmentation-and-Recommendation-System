import pandas as pd
import random
from collections import defaultdict
from scipy.stats import pearsonr

print("Started script")
pairs_df = pd.read_csv("results/top_100_pairs.csv")
print("Loaded top 100 pairs")

users_of_interest = set(pairs_df['user1']).union(set(pairs_df['user2']))

chunk_size = 100000
user_ratings = defaultdict(dict)

print("Starting to read ratings chunks")
i=1
for chunk in pd.read_csv("ratings.csv", chunksize=chunk_size):
    print(f"\rReading chunk {i}",end="")
    i+=1
    for row in chunk.itertuples(index=False):
        if row.userId in users_of_interest:
            user_ratings[row.userId][row.movieId] = row.rating

print("Finished reading ratings chunks")

def get_rating_corr(u1, u2):
    m1 = user_ratings.get(u1, {})
    m2 = user_ratings.get(u2, {})
    common = set(m1) & set(m2)
    if len(common) < 2:
        return None
    r1 = [m1[m] for m in common]
    r2 = [m2[m] for m in common]
    if len(set(r1)) == 1 or len(set(r2)) == 1:
        return None
    return pearsonr(r1, r2)[0]

print("Calculating twin correlations")
twin_corrs = []
for row in pairs_df.itertuples(index=False):
    corr = get_rating_corr(row.user1, row.user2)
    if corr is not None:
        twin_corrs.append(corr)

print("Sampling random pairs")
users = list(user_ratings.keys())
random_pairs = set()
random_corrs = []
while len(random_pairs) < 100:
    if len(users) < 2:
        break
    u1, u2 = random.sample(users, 2)
    pair = tuple(sorted((u1, u2)))
    if pair not in random_pairs:
        random_pairs.add(pair)
        sim = get_rating_corr(u1, u2)
        if sim is not None:
            random_corrs.append(sim)
print("Calculating random correlations")
for u1, u2 in random_pairs:
    corr = get_rating_corr(u1, u2)
    if corr is not None:
        random_corrs.append(corr)

print("Saving output")
with open("results/validation_stats.txt", "w") as f:
    if len(twin_corrs) > 0:
        f.write(f"Average correlation for top 100 similar pairs: {sum(twin_corrs)/len(twin_corrs):.4f}\n")
    else:
        f.write("No valid twin correlations computed.\n")
    
    if len(random_corrs) > 0:
        f.write(f"Average correlation for 100 random pairs: {sum(random_corrs)/len(random_corrs):.4f}\n")
    else:
        f.write("No valid random correlations computed.\n")

print("Saving output")
with open("results/correlation_summary.txt", "w") as f:
    if len(twin_corrs) > 0:
        f.write(f"Average correlation for top 100 similar pairs: {sum(twin_corrs)/len(twin_corrs):.4f}\n")
    else:
        f.write("No valid twin correlations computed.\n")
    
    if len(random_corrs) > 0:
        f.write(f"Average correlation for 100 random pairs: {sum(random_corrs)/len(random_corrs):.4f}\n")
    else:
        f.write("No valid random correlations computed.\n")

# Safe print
if len(twin_corrs) > 0:
    print("Twin mean correlation:", sum(twin_corrs) / len(twin_corrs))
else:
    print("No valid twin correlations.")

if len(random_corrs) > 0:
    print("Random mean correlation:", sum(random_corrs) / len(random_corrs))
else:
    print("No valid random correlations.")

print("Done!")
