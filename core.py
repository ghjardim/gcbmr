import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

# Creating adjacency matrix
threshold = 0.5
adjacency_matrix = np.where(cosine_sim > threshold, 1, 0)

# Creating the graph
G = nx.Graph()
num_nodes = cosine_sim.shape[0]
G.add_nodes_from(range(num_nodes))

for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        if adjacency_matrix[i, j] == 1:
            similarity_weight = cosine_sim[i, j]
            G.add_edge(i, j, weight=similarity_weight)

pos = nx.random_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=2, font_size=8, width=0.5)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
plt.show()

print(get_films_by_name('Lord of the Rings', indices))
print(get_recommended_movies_tfidf(5, cosine_sim, data))
