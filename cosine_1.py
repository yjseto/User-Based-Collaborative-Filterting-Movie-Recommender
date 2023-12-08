import pandas as pd
from sklearn.neighbors import NearestNeighbors 

# Read data
movies_table = pd.read_csv('ml-latest-small\ml-latest-small\movies.csv')
ratings_table = pd.read_csv('ml-latest-small\ml-latest-small\\ratings_test.csv') 

# Merge tables
movie_ratings_table = pd.merge(movies_table, ratings_table).drop(['genres','timestamp'], axis=1)

# Pivot table
user_ratings_matrix = movie_ratings_table.pivot_table(index='userId', columns='title', values='rating')

# Drop and fill NaN values
user_ratings_matrix = user_ratings_matrix.dropna(thresh=10, axis=1).fillna(0, axis=1)

# Fit Nearest Neighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(user_ratings_matrix)

# Example user ID
user_id = 1

# Find similar users
distances, indices = model.kneighbors([user_ratings_matrix.loc[user_id]])

# Recommend movies based on similar users
recommended_movies = user_ratings_matrix.columns[indices.flatten()]

# Print top 10 recommended movies
print("Top 10 Recommended Movies:")
print(recommended_movies[:10])  
