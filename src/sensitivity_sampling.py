from sklearn.cluster import KMeans
import numpy as np


class SensitivitySampling:
    def __init__(self, point_cloud, k=10):
        self.point_cloud = point_cloud
        self.k = k

    def kmeans_cost(self, pc):
        points_zipped = np.array(
            list(zip(pc.get_points_x(), pc.get_points_y(), pc.get_points_z()))
        )

        centers = np.array(pc.get_cluster_centers())
        distances = np.linalg.norm(points_zipped[:, np.newaxis] - centers, axis=2)

        min_distances = np.min(distances, axis=1)
        result = np.sum(min_distances**2)

        return result

    def get_cost(self):
        # if the clustering data is not available on the point cloud, run clustering
        if len(self.point_cloud.labels) == 0:
            self.kmeans()

        # if the cost data is already available on the point cloud, return it
        if self.point_cloud.get_kmeans_cost() is not None:
            return self.point_cloud.get_kmeans_cost()

        # calculate the cost of the clustering
        cost = self.kmeans_cost(self.point_cloud)
        self.point_cloud.set_kmeans_cost(cost)
        return cost

    def kmeans(self):
        results = KMeans(n_clusters=self.k).fit(self.point_cloud.to_array())
        self.point_cloud.set_labels(results.labels_)
        self.point_cloud.set_cluster_centers(results.cluster_centers_)

    def sample(self):
        self.kmeans()
        print(f"cost: {self.get_cost()}")
        print(f"cost: {self.get_cost()}")
