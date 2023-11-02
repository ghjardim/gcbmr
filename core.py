import pandas as pd
import calculate_similarity_matrixes as calc_sim
import clustering
import plot_graph

def load_data(file_path):
    data = pd.read_csv(file_path, na_values="?")
    columns_to_remove = ['Poster_Link', 'Meta_score', 'Certificate', 'Gross']
    data.drop(columns=columns_to_remove, inplace=True)
    return data

def get_films_by_name(movie_name, movie_indices):
    return movie_indices[movie_indices.index.str.contains(movie_name, na=False)]

if __name__ == "__main__":
    data = load_data("./dataset/imdb_top_1000.csv")

    overview_matrix = calc_sim.calculate_overview_matrix(data)
    genre_matrix = calc_sim.calculate_genre_matrix(data)
    director_matrix = calc_sim.calculate_director_matrix(data)
    star_matrix = calc_sim.calculate_star_matrix(data)
    year_matrix = calc_sim.calculate_year_matrix(data)
    runtime_matrix = calc_sim.calculate_runtime_matrix(data)

    G = calc_sim.create_graph(
            (overview_matrix,   0.6),
            (genre_matrix,      0.2),
            (director_matrix,   0.05),
            (star_matrix,       0.05),
            (year_matrix,       0.05),
            (runtime_matrix,    0.05)
        )

    clustering.perform_clustering(G, method="kmeans")
    #clustering.perform_clustering(G, method="agglomerative")

    cluster_similarity_matrix = clustering.compute_cluster_similarity(G)

    movie_indices = pd.Series(data.index, index=data['Series_Title'])
    movie_indices = movie_indices[~movie_indices.index.duplicated(keep='last')]

    target_movie = "Harry Potter and the Deathly Hallows: Part 1"
    target_movie_index = movie_indices[target_movie]

    movie_cluster = clustering.get_movie_cluster(target_movie, movie_indices, G)
    movies_in_cluster = clustering.get_movies_in_cluster(movie_cluster, G)
    movie_cluster_subgraph = clustering.create_cluster_subgraph(G, movie_cluster)
    neighbours_in_cluster = clustering.get_sorted_neighbors(movie_cluster_subgraph, target_movie_index)

    most_similar_cluster = clustering.get_most_similar_cluster(movie_cluster, cluster_similarity_matrix)
    similar_cluster_subgraph = clustering.create_cluster_subgraph(G, most_similar_cluster)
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

