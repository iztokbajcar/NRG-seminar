from sklearn.cluster import KMeans
import numpy as np
from src.point_cloud import PointCloud


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

    def sample2(self, sample_size, t=1):
        # run kmeans if labels are not yet computed
        if len(self.point_cloud.get_labels()) == 0:
            self.point_cloud.kmeans(self.k)

        P = self.point_cloud.to_array()  # all points: N x 3
        labels = np.array(self.point_cloud.get_labels())  # assigned cluster labels

        # group points into clusters
        clusters = [P[labels == i] for i in range(self.k)]

        # (2) Select the cheapest t clusters
        def cluster_cost(cluster):
            if len(cluster) == 0:
                return float("inf")
            center = np.mean(cluster, axis=0)
            return np.sum(np.linalg.norm(cluster - center, axis=1) ** 2)

        clusters_sorted = sorted(clusters, key=cluster_cost)
        C_cheap = clusters_sorted[:t]  # t cheapest clusters
        not_C_cheap = clusters_sorted[t:]  # remaining clusters

        # Flatten not_C_cheap for processing
        P_not = (
            np.concatenate([c for c in not_C_cheap if len(c) > 0], axis=0)
            if not_C_cheap
            else np.empty((0, 3))
        )

        # (3) Summarize each cheap cluster by its mean
        S = []
        W = []
        cheap_indices = []
        for cluster in C_cheap:
            if len(cluster) == 0:
                continue
            mu = np.mean(cluster, axis=0)  # Compute cluster mean
            S.append(mu)
            wp = len(cluster)  # Weight is the size of the cluster
            W.append(wp)
            # Find the index of the first point in the cluster to get its class
            P_arr = P
            idx = np.where((P_arr == cluster[0]).all(axis=1))[0][0]
            cheap_indices.append(idx)

        # (4) Apply sensitivity sampling on remaining clusters
        if len(P_not) > 0 and sample_size - t > 0:
            # Build a fake point cloud for the remaining points
            from point_cloud import PointCloud

            all_classes = np.array(self.point_cloud.get_points_class())
            # Find indices of P_not in P
            mask = np.any(np.all(P[:, None] == P_not[None, :], axis=2), axis=1)
            indices_not = np.where(mask)[0]
            pc_not = PointCloud(
                P_not[:, 0], P_not[:, 1], P_not[:, 2], all_classes[indices_not]
            )
            sampler_not = SensitivitySampling(pc_not, k=len(not_C_cheap))
            sampled_indices, W_not = sampler_not.sample(sample_size - t)
            S_not = P_not[sampled_indices]
            # Find classes for S_not
            sampled_classes_not = all_classes[indices_not][sampled_indices]
        else:
            S_not = np.empty((0, 3))
            W_not = []
            sampled_classes_not = np.array([])

        # Combine results from cheap and non-cheap clusters
        S = np.array(S + list(S_not))
        W = list(W) + list(W_not)
        # For classes: cheap_indices from above, then sampled_classes_not
        all_classes = np.array(self.point_cloud.get_points_class())
        sampled_classes = list(all_classes[cheap_indices]) + list(sampled_classes_not)
        return S, W, sampled_classes

    def compress(self, sample_size, sample_func=None, **kwargs):
        # Use the provided sample_func for sampling, default to self.sample
        if sample_func is None:
            sample_func = self.sample
        result = sample_func(sample_size, **kwargs)
        # Support both (indices, W) and (S, W, sampled_classes) return types
        if len(result) == 2:
            sampled_indices, W = result
            P = self.point_cloud.to_array()
            all_classes = np.array(self.point_cloud.get_points_class())
            S = P[sampled_indices]
            sampled_classes = all_classes[sampled_indices]
        else:
            S, W, sampled_classes = result
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
            lod.lod = lod_level + 1
            self.point_cloud.add_lod(lod)

        # add the original point cloud as the last LOD
        self.point_cloud.add_lod(self.point_cloud)
