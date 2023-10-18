import pandas as pd
import numpy as np

# Importing data
data = pd.read_csv("./dataset/imdb_top_1000.csv", na_values = "?")

# Removing unimportant columns
del data['Poster_Link']
del data['Meta_score']
del data['Certificate']
del data['Gross']

# Start TF-IDF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
data['Overview'] = data['Overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(data['Overview'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(data.index, index=data['Series_Title'])
indices = indices[~indices.index.duplicated(keep='last')]

def get_films_by_name(movie_name, movie_indices):
    return movie_indices[movie_indices.index.str.contains(movie_name, na=False)]
def get_recommended_movies_tfidf(target_movie_index, movie_similarities,movies_df):
    similarity_scores = pd.DataFrame(movie_similarities[target_movie_index], columns=["score"])
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return data['Series_Title'].iloc[movie_indices]

print(get_films_by_name('Lord of the Rings', indices))
print(get_recommended_movies_tfidf(5, cosine_sim, data))
