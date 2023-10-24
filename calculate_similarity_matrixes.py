import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def calculate_overview_matrix(text_data):
    tfidf = TfidfVectorizer(stop_words='english')
    text_data['Overview'] = text_data['Overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(text_data['Overview'])
    overview_similarity = cosine_similarity(tfidf_matrix)
    return overview_similarity

def calculate_genre_matrix(text_data):
    genres = text_data['Genre'].str.split(', ')
    mlb = MultiLabelBinarizer()
    genre_matrix = pd.DataFrame(mlb.fit_transform(genres), columns=mlb.classes_, index=text_data.index)
    genre_similarity = cosine_similarity(genre_matrix)
    return genre_similarity

def calculate_director_matrix(text_data):
    text_data['Director_Label'] = LabelEncoder().fit_transform(text_data['Director'])

    # Mapping of movie indices to director labels
    movie_to_director = dict(zip(text_data.index, text_data['Director_Label']))

    # Initialize the director similarity matrix with zeros
    num_movies = len(text_data)
    director_similarity = np.zeros((num_movies, num_movies))

    # Populate the director similarity matrix
    for i in range(num_movies):
        for j in range(num_movies):
            # If the movies have the same director, set similarity to 1; otherwise, set it to 0
            director_similarity[i, j] = 1 if movie_to_director[i] == movie_to_director[j] else 0

    return director_similarity

def calculate_star_matrix(text_data):
    star_columns = ['Star1', 'Star2', 'Star3', 'Star4']
    text_data['Stars'] = text_data[star_columns].apply(lambda row: row.dropna().tolist(), axis=1)
    mlb = MultiLabelBinarizer()
    star_matrix = pd.DataFrame(mlb.fit_transform(text_data['Stars']), columns=mlb.classes_, index=text_data.index)
    star_similarity = star_matrix.dot(star_matrix.T)
    star_similarity_normalized = MinMaxScaler().fit_transform(star_similarity)
    return star_similarity

def calculate_year_matrix(text_data):
    released_years = text_data['Released_Year'].values
    year_differences = np.abs(np.subtract.outer(released_years, released_years))

    def similarity_function(difference):
        return 1 / (1 + difference)

    year_similarity = np.vectorize(similarity_function)(year_differences)
    return year_similarity

def calculate_runtime_matrix(text_data):
    runtime = text_data['Runtime'].str.replace(' min', '').values.astype(int)
    runtime_differences = np.abs(np.subtract.outer(runtime, runtime))

    def similarity_function(difference):
        return 1 / (1 + difference)

    runtime_similarity = np.vectorize(similarity_function)(runtime_differences)
    return runtime_similarity