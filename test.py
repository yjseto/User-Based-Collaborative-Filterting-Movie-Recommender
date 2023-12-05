import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 



movies = pd.read_csv('ml-latest-small\ml-latest-small\movies.csv')
ratings = pd.read_csv('ml-latest-small\ml-latest-small\\ratings_test.csv') 

movie_ratings = pd.merge(movies,ratings).drop(['genres', 'timestamp'], axis = 1)

movie_ratings.fillna(0, inplace=True)
#print(movie_ratings.head())

# def standardize(row):
#     new_row = (row - row.mean())/(row.max()-row.min())
#     return new_row

# df_std = movie_ratings.apply(standardize).T
# #print(df_std.head())

# sparse_df = sparse.csr_matrix(df_std.values)
# corrMatrix = pd.DataFrame(cosine_similarity(sparse_df),index=ratings.columns,columns=ratings.columns)
# corrMatrix.head(100)

movie_ratings_matrix = movie_ratings.pivot_table(index=['userId'],columns=['title'],values='rating')
#print("Before: ",userRatings.shape)
movie_ratings_matrix = movie_ratings_matrix.dropna(thresh=10, axis=1).fillna(0,axis=1)
#movie_ratings_matrix.fillna(0, inplace=True)
#print("After: ",movie_ratings_matrix.shape)

corrMatrix = movie_ratings_matrix.corr(method='pearson')
print(corrMatrix.head(100))