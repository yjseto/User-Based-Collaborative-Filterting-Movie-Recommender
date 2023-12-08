import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

ratings_table = pd.read_csv('ml-latest-small\\ml-latest-small\\ratings.csv')
movies_table = pd.read_csv('ml-latest-small\\ml-latest-small\\movies.csv')


# Merge and remove unneeded columns
ratings_table = pd.merge(movies_table, ratings_table).drop(['genres', 'timestamp'], axis=1)

# Make matrix
userRatings = ratings_table.pivot_table(index=['userId'], columns=['title'], values='rating')

# Drop NaN values and fill missing values
userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0, axis=1)

# Compute the cosine similarity matrix
sparse_user_ratings = sparse.csr_matrix(userRatings.T)
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(sparse_user_ratings) #train the model

# Function to get similar movies based on user preferences using Nearest Neighbors
def get_similar(movie_name, rating):
    movie_index = userRatings.columns.get_loc(movie_name)
    query_movie = sparse_user_ratings[movie_index]
    distances, indices = model.kneighbors(query_movie, n_neighbors=10)
    
    similar_ratings = []
    for i in range(len(indices.flatten())):
        similar_movie = userRatings.columns[indices.flatten()[i]]
        similar_rating = rating * (1 - distances.flatten()[i])  # Adjust rating based on distance
        similar_ratings.append((similar_movie, similar_rating))
    
    return similar_ratings


# ################################Example for romantic lover##########################################

romantic_lover = [("(500) Days of Summer (2009)", 5), ("Alice in Wonderland (2010)", 3), ("Aliens (1986)", 1), ("2001: A Space Odyssey (1968)", 2)]
similar_movies_romantic = pd.DataFrame()
for movie, rating in romantic_lover:
    similar_movies_romantic = similar_movies_romantic._append(pd.DataFrame(get_similar(movie, rating), columns=['title', 'rating']))

print("Top 10 Recommended Romantic Movies:")
print(similar_movies_romantic.groupby('title').sum().sort_values(by='rating', ascending=False).head(10))



# ############################# Example for action lover#####################################################

action_lover = [("Amazing Spider-Man, The (2012)", 5), ("Mission: Impossible III (2006)", 4), ("Toy Story 3 (2010)", 5), ("2 Fast 2 Furious (Fast and the Furious 2, The) (2003)", 4)]
similar_movies_action = pd.DataFrame()
for movie, rating in action_lover:
    similar_movies_action = similar_movies_action._append(pd.DataFrame(get_similar(movie, rating), columns=['title', 'rating']))

print("\nTop 10 Recommended Action Movies:")
print(similar_movies_action.groupby('title').sum().sort_values(by='rating', ascending=False).head(10))


############################# Example for Me#####################################################

my_movie_list = [("Amazing Spider-Man, The (2012)", 5), ("Avengers: Infinity War - Part I (2018)",5)]
similar_movies_mine = pd.DataFrame()
for movie, rating in my_movie_list:
    similar_movies_mine = similar_movies_mine._append(pd.DataFrame(get_similar(movie, rating), columns=['title', 'rating']))

print("Top 10 Recommended Romantic Movies:")
print(similar_movies_mine.groupby('title').sum().sort_values(by='rating', ascending=False).head(10))

###############################################################################################################################