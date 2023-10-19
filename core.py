import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def load_data(file_path):
    data = pd.read_csv(file_path, na_values="?")
    columns_to_remove = ['Poster_Link', 'Meta_score', 'Certificate', 'Gross']
    data.drop(columns=columns_to_remove, inplace=True)
    return data

def calculate_tfidf_matrix(text_data):
    tfidf = TfidfVectorizer(stop_words='english')
    text_data['Overview'] = text_data['Overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(text_data['Overview'])
    return tfidf_matrix

def create_graph(cosine_sim):
    G = nx.Graph()
    num_nodes = cosine_sim.shape[0]
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if cosine_sim[i, j] == 1:
                similarity_weight = cosine_sim[i, j]
                G.add_edge(i, j, weight=similarity_weight)

    return G

def visualize_graph(G, pos=None, colors=None):
    if pos is None:
        pos = nx.random_layout(G)

    plt.figure(figsize=(10, 8))

    if colors is not None:
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=20, font_size=8, width=0.5, node_color=colors, cmap=plt.cm.viridis)
    else:
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=20, font_size=8, width=0.5)

    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
    plt.show()

def perform_clustering(G, adjacency_matrix, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(adjacency_matrix)

    for node, label in enumerate(cluster_labels):
        G.nodes[node]['cluster'] = label

    colors = [G.nodes[node]['cluster'] for node in G.nodes]
    visualize_graph(G, colors=colors)

    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    print("Number of nodes in each cluster:")
    print(cluster_counts)

def get_films_by_name(movie_name, movie_indices):
    return movie_indices[movie_indices.index.str.contains(movie_name, na=False)]

def get_recommended_movies_tfidf(target_movie_index, movie_similarities, movies_df):
    similarity_scores = pd.DataFrame(movie_similarities[target_movie_index], columns=["score"])
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return movies_df['Series_Title'].iloc[movie_indices]

if __name__ == "__main__":
    data = load_data("./dataset/imdb_top_1000.csv")

    tfidf_matrix = calculate_tfidf_matrix(data)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    G = create_graph(cosine_sim)
    visualize_graph(G)

    perform_clustering(G, cosine_sim)

    # Example usage of get_films_by_name and get_recommended_movies_tfidf
    movie_indices = pd.Series(data.index, index=data['Series_Title'])
    movie_indices = movie_indices[~movie_indices.index.duplicated(keep='last')]

    target_movie = "Inception"
    films_by_name = get_films_by_name(target_movie, movie_indices)
    print(f"Films containing '{target_movie}':\n{films_by_name}")

    target_movie_index = movie_indices[target_movie]
    recommended_movies = get_recommended_movies_tfidf(target_movie_index, cosine_sim, data)
    print(f"\nRecommended movies for '{target_movie}':\n{recommended_movies}")

