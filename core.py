import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import calculate_similarity_matrixes
import clustering

def load_data(file_path):
    data = pd.read_csv(file_path, na_values="?")
    columns_to_remove = ['Poster_Link', 'Meta_score', 'Certificate', 'Gross']
    data.drop(columns=columns_to_remove, inplace=True)
    return data

def create_graph(*matrix_weight_pairs):
    matrices, weights = zip(*matrix_weight_pairs)
    weighted_average_matrix = np.average(matrices, axis=0, weights=weights)

    G = nx.from_numpy_array(weighted_average_matrix)

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

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

def create_cluster_subgraph(G, cluster_number):
    cluster_indices = [node for node, data in G.nodes(data=True) if data.get('cluster') == cluster_number]

    if not cluster_indices:
        print(f"No nodes found in cluster {cluster_number}. Returning an empty subgraph.")
        return nx.Graph()

    # Create a subgraph with nodes and edges only from the specified cluster
    subgraph = G.subgraph(cluster_indices)

    return subgraph

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

if __name__ == "__main__":
    data = load_data("./dataset/imdb_top_1000.csv")

    overview_matrix = calculate_similarity_matrixes.calculate_overview_matrix(data)
    genre_matrix = calculate_similarity_matrixes.calculate_genre_matrix(data)
    director_matrix = calculate_similarity_matrixes.calculate_director_matrix(data)
    star_matrix = calculate_similarity_matrixes.calculate_star_matrix(data)
    year_matrix = calculate_similarity_matrixes.calculate_year_matrix(data)
    runtime_matrix = calculate_similarity_matrixes.calculate_runtime_matrix(data)

    G = create_graph(
            (overview_matrix,   0.6),
            (genre_matrix,      0.2),
            (director_matrix,   0.05),
            (star_matrix,       0.05),
            (year_matrix,       0.05),
            (runtime_matrix,    0.05)
        )

    perform_clustering(G)
    cluster_similarity_matrix = clustering.compute_cluster_similarity(G)

    movie_indices = pd.Series(data.index, index=data['Series_Title'])
    movie_indices = movie_indices[~movie_indices.index.duplicated(keep='last')]

    target_movie = "Inception"
    target_movie_index = movie_indices[target_movie]

    movie_cluster = clustering.get_movie_cluster(target_movie, movie_indices, G)
    movies_in_cluster = clustering.get_movies_in_cluster(movie_cluster, G)
    movie_cluster_subgraph = create_cluster_subgraph(G, movie_cluster)
    neighbours_in_cluster = clustering.get_sorted_neighbors(movie_cluster_subgraph, target_movie_index)

    most_similar_cluster = clustering.get_most_similar_cluster(movie_cluster, cluster_similarity_matrix)
    similar_cluster_subgraph = create_cluster_subgraph(G, most_similar_cluster)
    similar_subgraph_with_target = clustering.include_target_movie_in_cluster_subgraph(G, similar_cluster_subgraph, target_movie_index)
    neighbours_in_similar_cluster = clustering.get_sorted_neighbors(similar_subgraph_with_target, target_movie_index)

    #visualize_graph(create_cluster_subgraph(G, movie_cluster))
    #visualize_graph(similar_cluster_subgraph)
    #visualize_graph(similar_subgraph_with_target)

    print("Movie:", target_movie)
    print()
    print("Recommended movies:")
    print(data['Series_Title'].iloc[neighbours_in_cluster[0:15]])
    print()
    print("Try also:")
    print(data['Series_Title'].iloc[neighbours_in_similar_cluster[0:15]])

