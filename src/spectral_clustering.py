from sklearn.datasets import make_circles, make_moons

from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from k_means import k_means
import math

# from k_means import k_means


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


def spectral_clustering(X, k, n_clusters):
    affinity_matrix, neighbors = k_neighbors(X, k)
    degree = get_degree_matrix(neighbors)
    weights = get_weights(neighbors)
    # graph_laplacian = K.T @ K
    graph_laplacian = degree - weights
    _, eigenvectors = np.linalg.eig(graph_laplacian)
    eigenvectors = np.real(eigenvectors)[:, :n_clusters]
    # print(eigenvectors.shape)
    cluster = KMeans(n_clusters)
    s = cluster.fit(eigenvectors)
    # print(eigenvectors.shape)
    # k2 = 2
    # centroids, memberships = k_means(eigenvectors, n_clusters)
    # distances = pairwise_distances(X, centroids, metric="sqeuclidean")
    # memberships = np.zeros((X.shape[0]), dtype=np.int64)
    # np.argmin(distances, axis=1, out=memberships)
    # print(centroids.shape)

    # print(s.labels_.shape)
    return s.labels_


gen_moons_with_noise = lambda x: make_moons(n_samples=1000, noise=x)
gen_circles_with_noise = lambda x: make_circles(n_samples=1000, noise=x, factor=0.5)

moon_noise = [0.15, 0.10, 0.01]
circle_noise = [0.05, 0.03, 0.01]
for i in range(3):
    # matplotlib.use("QT5Agg")
    x_moon, y_moon = gen_moons_with_noise(moon_noise[i])

    labels = spectral_clustering(x_moon, 50, 2)

    plot_data(x_moon, labels)

    x_circle, y_circle = gen_circles_with_noise(moon_noise[i])

    labels = spectral_clustering(x_circle, 50, 2)

    plot_data(x_circle, labels)
