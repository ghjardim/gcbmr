import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

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