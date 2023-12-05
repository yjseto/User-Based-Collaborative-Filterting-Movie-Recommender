import pandas as pd
from sklearn.neighbors import NearestNeighbors 
from sklearn.impute import SimpleImputer

movies_table = pd.read_csv('ml-latest-small\ml-latest-small\movies.csv')
ratings_table = pd.read_csv('ml-latest-small\ml-latest-small\\ratings_test.csv') 
tags_table = pd.read_csv('ml-latest-small\ml-latest-small\\tags.csv')

# print(movies_table.info())
# print(ratings_table.info())
# print(tags_table.info())

# print(movies_table.head())
# print(ratings_table.head())
# print(tags_table.head()) 

movie_ratings_table = pd.merge(
    ratings_table,
    movies_table,
    on='movieId'
)
pd.set_option('display.max_columns', None)

movie_ratings_table.dropna(inplace= True)
#print(movie_ratings_table.head())

movie_ratings_matrix = movie_ratings_table.pivot_table(index='userId', columns='title', values = 'rating')
#movie_ratings_matrix = movie_ratings_table.pivot_table(index='title', columns='userId', values = 'rating')
#movie_ratings_matrix.dropna(inplace=True)

#imputer = SimpleImputer(strategy='mean')
imputer = SimpleImputer(strategy= 'constant', fill_value= -1)

movie_ratings_matrix_imputed = pd.DataFrame(imputer.fit_transform(movie_ratings_matrix), columns=movie_ratings_matrix.columns)

model = NearestNeighbors(metric= 'cosine', algorithm='brute')
model.fit(movie_ratings_matrix_imputed)


#print(movie_ratings_matrix_imputed.head())
user_id = 1  # Example user ID
user_ratings = movie_ratings_matrix_imputed.loc[user_id].dropna()
unrated_movies = movie_ratings_matrix_imputed.columns[user_ratings.isnull()]

# Find similar users
distances, indices = model.kneighbors([user_ratings])

# Recommend movies based on similar users
recommended_movies = movie_ratings_matrix_imputed.columns[indices.flatten()]

print(recommended_movies)