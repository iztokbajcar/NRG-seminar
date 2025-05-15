from sklearn.cluster import KMeans
import numpy as np


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

        S = P[sampled_indices]  # sampled coreset points

        # assign weights: inverse of sampling probability × sample size
        weights = {}
        for i in sampled_indices:
            key = tuple(P[i])
            weights[key] = 1 / (sample_size * dist[i])

        W = [weights[tuple(p)] for p in S]

        return S, W
