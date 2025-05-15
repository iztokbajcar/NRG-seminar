from sklearn.cluster import KMeans
import numpy as np


class SensitivitySampling:
    def __init__(self, point_cloud, k=10):
        self.point_cloud = point_cloud
        self.k = k

    def sample(self, sample_size):
        self.point_cloud.kmeans(self.k)

        squared_distances = self.point_cloud.get_squared_distances()
        total_cost = np.sum(squared_distances)

        if total_cost == 0:
            length = len(squared_distances)
            probabilities = np.ones(length) / length
        else:
            probabilities = squared_distances / total_cost

        sampled_indices = np.random.choice(
            len(squared_distances), size=sample_size, replace=True, p=probabilities
        )

        return self.point_cloud.to_array()[sampled_indices]
