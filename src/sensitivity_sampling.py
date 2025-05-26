from sklearn.cluster import KMeans
import numpy as np
from point_cloud import PointCloud


class SensitivitySampling:
    def __init__(self, point_cloud, k=10):
        self.point_cloud = point_cloud
        self.k = k

    def sample(self, sample_size):
        # run kmeans if labels are not yet computed
        if len(self.point_cloud.get_labels()) == 0:
            self.point_cloud.kmeans(self.k)

        P = self.point_cloud.to_array()  # all points: N x 3
        labels = np.array(self.point_cloud.get_labels())  # assigned cluster labels

        # group points into clusters
        clusters = [P[labels == i] for i in range(self.k)]

        dist = dict()
        idx = 0

        for cluster in clusters:
            if len(cluster) == 0:
                continue  # avoid empty clusters

            # compute center of cluster
            center = np.mean(cluster, axis=0)

            # cost of cluster (scaled by k)
            cost_cluster_k = self.k * np.sum(
                np.linalg.norm(cluster - center, axis=1) ** 2
            )

            for point in cluster:
                cost_p = np.linalg.norm(point - center) ** 2
                mu_p = 1 / (self.k * len(cluster)) + cost_p / cost_cluster_k
                dist[idx] = mu_p
                idx += 1

        # build normalized probability distribution from sensitivities
        p_vals = np.array(list(dist.values()), dtype=np.float64)
        p_vals /= np.sum(p_vals)

        indices = np.arange(len(P))
        sampled_indices = np.random.choice(
            indices, size=sample_size, replace=True, p=p_vals
        )
        W = [1 / (sample_size * dist[i]) for i in sampled_indices]
        return sampled_indices, W

    def compress(self, sample_size):
        # Use the sample method to get sampled points and weights
        sampled_indices, W = self.sample(sample_size)
        P = self.point_cloud.to_array()
        all_classes = np.array(self.point_cloud.get_points_class())

        S = P[sampled_indices]
        sampled_classes = all_classes[sampled_indices]

        x = S[:, 0]
        y = S[:, 1]
        z = S[:, 2]

        return PointCloud(x, y, z, sampled_classes)

    def generate_lods(self, num_lods):
        # generate multiple levels of detail (LODs) for the point cloud
        # num_lods is the number of all desired LODs including the original point cloud

        # LOD num_lods is the original point cloud
        # LOD num_lods - 1 is a smaller point cloud
        # and so on until LOD 1 which is the smallest point cloud

        # clear existing LODs
        self.point_cloud.set_lods([])

        n_points = len(self.point_cloud.get_points_x())
        print(f"Number of all points: {n_points}")
        lod_point_count_step = n_points // num_lods

        for lod_level in range(0, num_lods - 1):
            n_lod_points = (lod_level + 1) * lod_point_count_step
            print(f"Generating LOD {lod_level} with {n_lod_points} points")

            lod = self.compress(n_lod_points)
            self.point_cloud.add_lod(lod)

        # add the original point cloud as the last LOD
        self.point_cloud.add_lod(self.point_cloud)
