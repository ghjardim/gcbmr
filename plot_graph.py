import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def visualize_graph(G, color=False, pos=None):
    if pos is None:
        pos = nx.spring_layout(G, weight='weight', seed=42)

    plt.figure(figsize=(10, 8))

    if not color:
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=20, font_size=8, width=0.1)
    else:
        colors = [G.nodes[node]['cluster'] for node in G.nodes]
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=20, font_size=8, width=0.1,
                node_color=colors, cmap=plt.cm.viridis)

    plt.show()


def visualize_graph_clusters(G, pos=None, color=False, label_alpha=0.1):
    if pos is None:
        pos = nx.spring_layout(G, weight='weight', seed=42)

    plt.figure(figsize=(10, 8))

    if not color:
        nx.draw(G, pos, with_labels=False, font_weight='bold',
                node_size=20, font_size=8, width=0.1, edge_color='lightgray')
        nx.draw_networkx_labels(G, pos, labels={k: k for k in G.nodes},
                                font_size=8, font_weight='bold', alpha=label_alpha)
        nx.draw_networkx_edges(G, pos, width=0.1, edge_color='lightgray')
        nx.draw_networkx_nodes(G, pos, node_size=20)
    else:
        colors = [G.nodes[node]['cluster'] for node in G.nodes]
        nx.draw(G, pos, with_labels=False, font_weight='bold',
                node_size=20, font_size=8, width=0.1, edge_color='lightgray')
        nx.draw_networkx_labels(G, pos, labels={k: k for k in G.nodes},
                                font_size=8, font_weight='bold', alpha=label_alpha)
        nx.draw_networkx_edges(G, pos, width=0.1, edge_color='lightgray')
        nx.draw_networkx_nodes(G, pos, node_size=20,
                        node_color=colors, cmap=plt.cm.viridis)

    plt.show()
