import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def generate_clusters(k=20, points_per_cluster=100, dim=20, min_distance=20, cluster_std=1.0, seed=None):
    """
    Generate a dataset with k clusters, each having points_per_cluster points,
    distributed Normally around the cluster center.

    Parameters:
        k (int): Number of clusters.
        points_per_cluster (int): Number of points in each cluster.
        dim (int): Dimensionality of the points.
        min_distance (float): Minimum distance between cluster centers.
        cluster_std (float): Standard deviation of points around cluster centers.
        seed (int): Seed for reproducibility.

    Returns:
        np.ndarray: Dataset of shape (k * points_per_cluster, dim).
        np.ndarray: Cluster labels of shape (k * points_per_cluster,).
    """
    if seed is not None:
        np.random.seed(seed)

    cluster_centers = []
    while len(cluster_centers) < k:
        # Propose a new cluster center
        new_center = np.random.uniform(-100, 100, size=dim)

        # Check if it's sufficiently far from existing centers
        if all(np.linalg.norm(new_center - center) >= min_distance for center in cluster_centers):
            cluster_centers.append(new_center)

    # Generate points for each cluster
    data = []
    labels = []
    for i, center in enumerate(cluster_centers):
        points = np.random.normal(loc=center, scale=cluster_std, size=(points_per_cluster, dim))
        data.append(points)
        labels.extend([i] * points_per_cluster)

    return np.vstack(data), np.array(labels)


def plot_clusters(data, labels, dim=2):
    """
    Visualize the dataset by reducing its dimensionality using PCA.
    """
    if dim not in [2, 3]:
        raise ValueError("Visualization dimension can only be 2 or 3.")

    pca = PCA(n_components=dim)
    reduced_data = pca.fit_transform(data)

    fig = plt.figure(figsize=(10, 8))

    if dim == 2:
        for label in np.unique(labels):
            cluster_points = reduced_data[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}", alpha=0.6)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
    elif dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        for label in np.unique(labels):
            cluster_points = reduced_data[labels == label]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f"Cluster {label}",
                       alpha=0.6)
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")

    plt.title("Cluster Visualization")
    plt.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    k = 20
    points_per_cluster = 100
    dim = 20
    min_distance = 30
    cluster_std = 1.0
    seed = 42

    data, labels = generate_clusters(k, points_per_cluster, dim, min_distance, cluster_std, seed)

    print("Dataset shape:", data.shape)
    print("Labels shape:", labels.shape)
    np.savetxt("data/benchmark_dist30_k20.csv", data, delimiter=",")

    # Visualize in 2D
    plot_clusters(data, labels, dim=2)
