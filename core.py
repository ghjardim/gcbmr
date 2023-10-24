import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def load_data(file_path):
    data = pd.read_csv(file_path, na_values="?")
    columns_to_remove = ['Poster_Link', 'Meta_score', 'Certificate', 'Gross']
    data.drop(columns=columns_to_remove, inplace=True)
    return data

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

def get_sorted_neighbors(graph, vertex):
    if vertex in graph:
        # Get the neighbors of the vertex
        neighbors = list(graph.neighbors(vertex))

        # Sort the neighbors based on the edge weights in descending order
        sorted_neighbors = sorted(
                neighbors,
                key=lambda neighbor: graph[vertex][neighbor]['weight'], reverse=True)

        return sorted_neighbors
    else:
        print(f"Vertex {vertex} is not in the graph.")
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

def compute_cluster_similarity(G):
    clusters = set(nx.get_node_attributes(G, 'cluster').values())
    cluster_weights = {}

    # Step 1: Compute Edge Weights Between Clusters
    for edge in G.edges(data=True):
        node1, node2, weight = edge
        cluster1 = G.nodes[node1].get('cluster')
        cluster2 = G.nodes[node2].get('cluster')

        if cluster1 is not None and cluster2 is not None and cluster1 != cluster2:
            key = tuple(sorted([cluster1, cluster2]))
            cluster_weights[key] = cluster_weights.get(key, 0) + weight['weight']

    # Step 2: Normalize the Weights
    max_weight = max(cluster_weights.values())
    cluster_weights_normalized = {key: value / max_weight for key, value in cluster_weights.items()}

    # Step 3: Create a Cluster Similarity Matrix
    num_clusters = len(clusters)
    cluster_similarity_matrix = np.zeros((num_clusters, num_clusters))

    for i, cluster_i in enumerate(clusters):
        for j, cluster_j in enumerate(clusters):
            if i != j:
                key = tuple(sorted([cluster_i, cluster_j]))
                cluster_similarity_matrix[i, j] = cluster_weights_normalized.get(key, 0)

    return cluster_similarity_matrix

def get_most_similar_cluster(movie_cluster, cluster_similarity_matrix):
    sorted_clusters_by_similarity = sorted(enumerate(
        cluster_similarity_matrix[movie_cluster]), key=lambda x: x[1], reverse=True)
    sorted_indices = [t[0] for t in sorted_clusters_by_similarity]
    most_similar_cluster = sorted_indices[0]
    return most_similar_cluster

def analyze():
    movie_indices = pd.Series(data.index, index=data['Series_Title'])
    movie_indices = movie_indices[~movie_indices.index.duplicated(keep='last')]

    target_movie = "Inception"
    target_movie_index = movie_indices[target_movie]

    movie_cluster = get_movie_cluster(target_movie, movie_indices, G)
    movies_in_cluster = get_movies_in_cluster(movie_cluster, G)
    neighbours_in_cluster = get_sorted_neighbors(
            create_cluster_subgraph(G, movie_cluster),
            target_movie_index)
    most_similar_cluster = get_most_similar_cluster(movie_cluster, cluster_similarity_matrix)

    for cluster in range(0,10):
        cluster_subgraph = create_cluster_subgraph(G, cluster)
        print("Cluster " + str(cluster)
              + "\tConnected components:" + str(len(list(nx.connected_components(cluster_subgraph))))
              + "\tNodes:" + str(cluster_subgraph.number_of_nodes()))

    print("---")
    print("Recommended movies:", str(neighbours_in_cluster[0:10]))
    print("Try also:", "movies from cluster", str(most_similar_cluster))

if __name__ == "__main__":
    data = load_data("./dataset/imdb_top_1000.csv")

    overview_matrix = calculate_overview_matrix(data)
    genre_matrix = calculate_genre_matrix(data)
    director_matrix = calculate_director_matrix(data)
    star_matrix = calculate_star_matrix(data)
    year_matrix = calculate_year_matrix(data)
    runtime_matrix = calculate_runtime_matrix(data)

    G = create_graph(
            (overview_matrix,   0.4),
            (genre_matrix,      0.4),
            (director_matrix,   0.05),
            (star_matrix,       0.05),
            (year_matrix,       0.05),
            (runtime_matrix,    0.05)
        )

    perform_clustering(G)
    cluster_similarity_matrix = compute_cluster_similarity(G)

    analyze()
