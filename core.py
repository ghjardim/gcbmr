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
            if cosine_sim[i, j] > 0.1:
                similarity_weight = cosine_sim[i, j]
                G.add_edge(i, j, weight=similarity_weight)

    return G

def visualize_graph(G, pos=None, colors=None):
    if pos is None:
        pos = nx.spring_layout(G, weight='weight', seed=42)

    plt.figure(figsize=(10, 8))

    if colors is not None:
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=20, font_size=8, width=0.1,
                node_color=colors, cmap=plt.cm.viridis)
    else:
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=20, font_size=8, width=0.1)

    plt.show()

def perform_clustering(G, num_clusters=10):
    adjacency_matrix = nx.adjacency_matrix(G).toarray()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(adjacency_matrix)

    for node, label in enumerate(cluster_labels):
        G.nodes[node]['cluster'] = label

    colors = [G.nodes[node]['cluster'] for node in G.nodes]
    visualize_graph(G, colors=colors)

def get_films_by_name(movie_name, movie_indices):
    return movie_indices[movie_indices.index.str.contains(movie_name, na=False)]

def get_recommended_movies_tfidf(target_movie_index, movie_similarities, movies_df):
    similarity_scores = pd.DataFrame(movie_similarities[target_movie_index], columns=["score"])
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return movies_df['Series_Title'].iloc[movie_indices]

def get_movie_cluster(movie_title, movie_indices, G):
    movie_index = movie_indices.get(movie_title)

    if movie_index is not None and movie_index in G.nodes:
        return G.nodes[movie_index].get('cluster', None)
    else:
        print(f"Movie '{movie_title}' not found or not present in the graph.")
        return None

def get_movies_in_cluster(cluster_number, G):
    cluster_indices = [node for node, data in G.nodes(data=True) if data.get('cluster') == cluster_number]

    if not cluster_indices:
        print(f"No movies found in cluster {cluster_number}.")
        return None

    return cluster_indices

def create_cluster_subgraph(G, cluster_number):
    cluster_indices = [node for node, data in G.nodes(data=True) if data.get('cluster') == cluster_number]

    if not cluster_indices:
        print(f"No nodes found in cluster {cluster_number}. Returning an empty subgraph.")
        return nx.Graph()

    # Create a subgraph with nodes and edges only from the specified cluster
    subgraph = G.subgraph(cluster_indices)

    return subgraph

def analyse():
    movie_indices = pd.Series(data.index, index=data['Series_Title'])
    movie_indices = movie_indices[~movie_indices.index.duplicated(keep='last')]

    target_movie = "Inception"
    target_movie_index = movie_indices[target_movie]

    movie_cluster = get_movie_cluster(target_movie, movie_indices, G)
    if movie_cluster is not None:
        print(f"The cluster of the movie '{target_movie}' is: {movie_cluster}")

    movies_in_cluster = get_movies_in_cluster(movie_cluster, G)
    if movies_in_cluster is not None:
        print(f"Movies in Cluster {movie_cluster}: {movies_in_cluster}", end="")
        num_movies = len(movies_in_cluster)
        print(f" -> {num_movies}")

    print("Movie " + target_movie + " has the following neighbours: " + str(list(G.neighbors(target_movie_index))))
    print("Movie " + target_movie + " has the following neighbours in cluster " + str(movie_cluster) + ": " + str(list(create_cluster_subgraph(G, movie_cluster).neighbors(target_movie_index))))
    print("Movie " + target_movie + " got the following TF-IDF recommended movies: " + str(sorted(get_recommended_movies_tfidf(target_movie_index, cosine_sim, data).index.tolist())))

    print("Graph: Connected components:\t" + str(len(list(nx.connected_components(G)))))
    for cluster in range(0,10):
        cluster_subgraph = create_cluster_subgraph(G, cluster)
        #visualize_graph(cluster_subgraph)
        print("Cluster " + str(cluster)
              + "\tConnected components:" + str(len(list(nx.connected_components(cluster_subgraph))))
              + "\tNodes:" + str(cluster_subgraph.number_of_nodes()))

if __name__ == "__main__":
    data = load_data("./dataset/imdb_top_1000.csv")

    tfidf_matrix = calculate_tfidf_matrix(data)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    G = create_graph(cosine_sim)

    perform_clustering(G)

    analyse()
