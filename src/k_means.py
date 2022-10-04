from bz2 import compress
import numpy as np
from sklearn.metrics import pairwise_distances


def k_means(X, k):
    # this should work for 2d or 3d points
    node_count, point_length = X.shape

    # we'll use 2 arrays of centroids to determine when the centroids converge
    previous_centroids = np.zeros((k, point_length))
    current_centroids = X[np.random.choice(node_count, size=k), ::]

    # the next few arrays are created to avoid allocating the space in a loop
    distances = np.zeros((node_count, k), dtype=np.float64)
    cluster_membership = np.zeros((node_count), dtype=np.int64)

    while not np.allclose(current_centroids, previous_centroids):
        # compute the euclidean distance between each point and the current centroids
        distances = pairwise_distances(X, current_centroids, metric="sqeuclidean")

        # determine which cluster each point belongs
        np.argmin(distances, axis=1, out=cluster_membership)

        # copy the current centroids for comparison
        previous_centroids = np.copy(current_centroids)

        # compute the new centroids
        for i in range(k):  # there's gotta be a one-liner for this
            current_centroids[i] = np.mean(X[np.where(cluster_membership == i)], axis=0)

    # returning memberships since they are for the current centroids(that have not changed)
    return current_centroids, cluster_membership


def compress(X, centroids, memberships=None):
    A = np.copy(X)
    if memberships is None:
        memberships = np.zeros((X.shape[0]), dtype=np.int64)
        distances = pairwise_distances(X, centroids, metric="sqeuclidean")
        np.argmin(distances, axis=1, out=memberships)

    for i in range(centroids.shape[0]):
        A[np.where(memberships == i)] = centroids[i]

    return A


if __name__ == "__main__":
    from PIL import Image

    building_data = np.asarray(Image.open("./data/building.png"))
    # Image.fromarray(building_data.T).show()
    building_height, building_width, building_depth = building_data.shape
    building_data = building_data.reshape((building_height * building_width, 3))

    centroids, memberships = k_means(building_data, 3)
    print(centroids.shape)
    building_data = compress(building_data, centroids, memberships)
    # Image.fromarray(building_data.reshape(building_height, building_width, 3)).save(
    #    "./data/building_compress.png"
    # )

    bulldog_data = np.asarray(Image.open("./data/bulldog.png"))
    # Image.fromarray(building_data.T).show()
    bulldog_height, bulldog_width, bulldog_depth = bulldog_data.shape
    bulldog_data = bulldog_data.reshape((bulldog_height * bulldog_width, 3))

    centroids, memberships = k_means(bulldog_data, 3)

    bulldog_data = compress(bulldog_data, centroids, memberships)
    # Image.fromarray(bulldog_data.reshape(bulldog_height, bulldog_width, 3)).save(
    #    "./data/bulldog_compress.png"
    # )
