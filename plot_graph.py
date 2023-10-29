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