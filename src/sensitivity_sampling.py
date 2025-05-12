from sklearn.cluster import KMeans


class SensitivitySampling:
    def __init__(self, point_cloud, k=10):
        self.point_cloud = point_cloud
        self.k = k

    def sample(self):
        results = KMeans(n_clusters=self.k).fit(self.point_cloud.to_array())
        self.point_cloud.set_cluster_indices(results.labels_)

        # TODO sensitivity sampling

        return results
