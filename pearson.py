import pandas as pd
from scipy import sparse

ratings_table = pd.read_csv('ml-latest-small\ml-latest-small\\ratings_test.csv')
movies_table = pd.read_csv('ml-latest-small\ml-latest-small\movies.csv')

#merge and remove unneeded columns
ratings_table = pd.merge(movies_table,ratings_table).drop(['genres','timestamp'],axis=1)

#########table info###########
print(ratings_table.shape)
print(ratings_table)

#make matrix
userRatings = ratings_table.pivot_table(index=['userId'],columns=['title'],values='rating')
#print("Before: ",userRatings.shape)


userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0,axis=1)

print("After: ",userRatings.shape)


CorrelationMatrix = userRatings.corr(method='pearson')
#print(corrMatrix.head(100))

def get_similar(movie_name,rating):
    similar_ratings = CorrelationMatrix[movie_name]*(rating-2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    return similar_ratings

romantic_lover = [("(500) Days of Summer (2009)",5),("Alice in Wonderland (2010)",3),("Aliens (1986)",1),("2001: A Space Odyssey (1968)",2)]
similar_movies = pd.DataFrame()
for movie,rating in romantic_lover:
    similar_movies = similar_movies._append(get_similar(movie,rating),ignore_index = True)

#print(similar_movies.head(10))

#print(similar_movies.sum().sort_values(ascending=False).head(20))

action_lover = [("Amazing Spider-Man, The (2012)",5),("Mission: Impossible III (2006)",4),("Toy Story 3 (2010)",2),("2 Fast 2 Furious (Fast and the Furious 2, The) (2003)",4)]
similar_movies = pd.DataFrame()
for movie,rating in action_lover:
    similar_movies = similar_movies._append(get_similar(movie,rating))

similar_movies.head(10)
print(similar_movies.sum().sort_values(ascending=False).head(20))