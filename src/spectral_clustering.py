from platform import node
from sklearn.datasets import make_circles, make_moons
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import math


# class K_means:
#     def __init__(X, k):
#         # this is more or less a max size
#         # the minus one is because one point is the center
#         cluster_size = (len(X) // k) - 1
#         self.centroids = []

#         for i in np.random.choice(len(X), k):
#             self.centroids.append(centroid(X[i], i, cluster_size))


# class centroid:
#     def __init__(point, index, cluster_size):
#         self.center = point
#         self.index = index
#         self.members = np.empty(cluster_size)


def plot_data(X, y):
    # When the label y is 0, the class is represented with a blue square.
    # When the label y is 1, the class is represented with a green triangle.
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "g^")

    # X contains two features, x1 and x2
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20)

    # Simplifying the plot by removing the axis scales.
    # plt.xticks([])
    # plt.yticks([])
    plt.show()


def k_neighbors(X, k):
    node_count = X.shape[0]

    affinity_matrix = np.zeros((node_count, node_count))

    affinity_matrix = np.exp(
        -1.0 / (2 * 1) * pairwise_distances(X, metric="sqeuclidean")
    )
    neighbor_indices = np.zeros((node_count, k), dtype=np.int64)

    for node in range(node_count):
        # Note on stop, since we want k elements, and we starting from -2(to exclude the node itself), and stop is exclusive
        # were ending on (k+2)
        neighbor_indices[node] = affinity_matrix[node].argsort()[-2 : -(k + 2) : -1]

    print(neighbor_indices)

    return affinity_matrix, neighbor_indices


def get_degree_matrix(neighbors):
    edge_counter = np.bincount(neighbors.flatten())
    return np.diag(edge_counter)


def get_weights(neighbors):
    node_count = neighbors.shape[0]
    weights = np.zeros((node_count, node_count))
    for row in range(node_count):
        np.put(weights[row], neighbors[row], 1)
    return weights


def k_means(eigenvectors, k):
    print("yeet")
    print(eigenvectors)
    center_indices = np.zeros(k, dtype=np.int64)
    center_values = np.zeros(
        (k, 2), dtype=np.float64
    )  # represents the centers of the k-means clusters
    centroid_membership = np.zeros()


def spectral_clustering(x, k):
    affinity_matrix, neighbors = k_neighbors(x, k)
    D = get_degree_matrix(neighbors)
    W = get_weights(neighbors)
    # graph_laplacian = K.T @ K
    graph_laplacian = D - W
    _, eigenvectors = np.linalg.eig(graph_laplacian)
    k_means(eigenvectors, k)


gen_moons_with_noise = lambda x: make_moons(n_samples=1000, noise=x)
gen_circles_with_noise = lambda x: make_circles(n_samples=1000, noise=x)

moon_noise = [0.15, 0.10, 0.01]
circle_noise = [0.05, 0.03, 0.01]
for i in range(3):
    x_moon, y_moon = gen_moons_with_noise(moon_noise[i])
    print(x_moon)

    # plot_data(x_moon, y_moon)
    spectral_clustering(x_moon, 22)
    print(neighbors.max())
    # print(KDTree.tree)
    break
