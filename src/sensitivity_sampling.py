from sklearn.cluster import KMeans
import numpy as np
from src.point_cloud import PointCloud
import math
import time


class SensitivitySampling:
    def __init__(self, point_cloud, k=10):
        self.point_cloud = point_cloud
        self.k = k

    def sample(self, sample_size, P=None, labels=None):
        # run kmeans if labels are not yet computed
        if labels is None:
            labels = self.point_cloud.get_labels()
            if len(labels) == 0:
                self.point_cloud.kmeans(self.k)
                labels = self.point_cloud.get_labels()
        if P is None:
            P = self.point_cloud.to_array()  # all points: N x 3
        labels = np.asarray(labels)
        N = len(P)
        sensitivities = np.zeros(N, dtype=np.float64)

        for i in range(self.k):
            mask = (labels == i)
            cluster = P[mask]
            if len(cluster) == 0:
                continue
            center = np.mean(cluster, axis=0)
            dists = np.linalg.norm(cluster - center, axis=1) ** 2
            cost_cluster_k = self.k * np.sum(dists)
            mu_p = 1 / (self.k * len(cluster)) + dists / cost_cluster_k
            sensitivities[mask] = mu_p

        # Normalize sensitivities to get probabilities
        total_sensitivity = np.sum(sensitivities)
        indices = np.arange(N)
        if total_sensitivity == 0 or np.isnan(total_sensitivity):
            # Fallback: use uniform probabilities and uniform weights
            p_vals = np.ones(N) / N
            sampled_indices = np.random.choice(indices, size=sample_size, replace=True, p=p_vals)
            W = np.ones(sample_size)  # or W = np.ones(sample_size) / sample_size for normalized weights
        else:
            p_vals = sensitivities / total_sensitivity
            sampled_indices = np.random.choice(indices, size=sample_size, replace=True, p=p_vals)
            W = 1 / (sample_size * sensitivities[sampled_indices])
        return sampled_indices, W


    def compress(self, sample_size, sample_func=None, **kwargs):
        # Use the provided sample_func for sampling, default to self.sample
        if sample_func is None:
            sample_func = self.sample
        P = self.point_cloud.to_array()
        labels = self.point_cloud.get_labels()
        start_time = time.time()
        # Pass P and labels to sample_func if it supports them
        try:
            result = sample_func(sample_size, P=P, labels=labels, **kwargs)
        except TypeError:
            result = sample_func(sample_size, **kwargs)
        elapsed = time.time() - start_time
        print(f"Sampling took {elapsed:.4f} seconds.")
        # Support both (indices, W) and (S, W, sampled_classes) return types
        if len(result) == 2:
            sampled_indices, W = result
            all_classes = np.array(self.point_cloud.get_points_class())
            S = P[sampled_indices]
            sampled_classes = all_classes[sampled_indices]
        else:
            S, W, sampled_classes = result
        x = S[:, 0]
        y = S[:, 1]
        z = S[:, 2]
        return PointCloud(x, y, z, sampled_classes)

    
    def get_n_points_for_lod(self, num_lods, lod_level, n_points, func="exponential2", lower_bound_compression=0.005, upper_bound_compression=0.7):
        min_point_count = n_points * lower_bound_compression
        max_point_count = n_points * upper_bound_compression
        if (func == "linear"):
            # Linear interpolation between min and max point count
            if num_lods == 1:
                return int(max_point_count)
            step = (max_point_count - min_point_count) / (num_lods - 1)
            n_lod_points = min_point_count + step * lod_level
            return int(round(n_lod_points))
        elif (func == "logarithmic"):
            # Logarithmic interpolation between min and max point count
            if num_lods == 1:
                return int(max_point_count)
            # Avoid log(0) by shifting lod_level by 1
            log_min = math.log(1)
            log_max = math.log(num_lods)
            log_lod = math.log(lod_level + 1)
            n_lod_points = min_point_count + (max_point_count - min_point_count) * (log_lod - log_min) / (log_max - log_min)
            return int(round(n_lod_points))
        elif (func == "exponential"):
            # Exponential interpolation between min and max point count
            if num_lods == 1:
                return int(max_point_count)
            exp_min = 1
            exp_max = math.exp(num_lods - 1)
            exp_lod = math.exp(lod_level)
            n_lod_points = min_point_count + (max_point_count - min_point_count) * (exp_lod - exp_min) / (exp_max - exp_min)
            return int(round(n_lod_points))
        elif (func == "exponential2"):
            # Less steep exponential interpolation between min and max point count
            if num_lods == 1:
                return int(max_point_count)
            exp_divisor = 0.5  # or 0.5 for even steeper
            exp_min = 1
            exp_max = math.exp((num_lods - 1) / exp_divisor)
            exp_lod = math.exp(lod_level / exp_divisor)
            n_lod_points = min_point_count + (max_point_count - min_point_count) * (exp_lod - exp_min) / (exp_max - exp_min)
            return int(round(n_lod_points))
        else:
            raise ValueError(f"Invalid function: {func}")

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

        for lod_level in range(0, num_lods):
            n_lod_points = self.get_n_points_for_lod(num_lods, lod_level, n_points)
            print(f"Generating LOD {lod_level} with {n_lod_points} points")

            lod = self.compress(n_lod_points)
            lod.lod = lod_level + 1
            self.point_cloud.add_lod(lod)

