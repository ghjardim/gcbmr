import networkx as nx
import numpy as np
import plot_graph

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

def include_target_movie_in_cluster_subgraph(G, subgraph, target_movie):
    new_subgraph = subgraph.copy()

    new_subgraph.add_node(target_movie)
    target_movie_edges = [(target_movie, neighbor, data) for neighbor, data in G[target_movie].items() if neighbor in subgraph]
    new_subgraph.add_edges_from(target_movie_edges)

    return new_subgraph

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
    plot_graph.visualize_graph(G, colors=colors)