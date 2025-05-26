"""
Evaluation module for sensitivity sampling algorithms.
This module provides methods for evaluating the quality of clustering solutions
and coresets, including cost computation and distortion measurement.
"""

import numpy as np


class Evaluation:
    """
    Class for evaluating clustering solutions and coresets.
    Provides methods for computing costs, distortions, and benchmark construction.
    """
    
    def cost(self, points, centers):
        """
        Compute the k-means cost of a set of points with respect to centers.
        The cost is the average squared distance from each point to its nearest center.
        
        Args:
            points (numpy.ndarray): Set of points
            centers (numpy.ndarray): Set of centers
            
        Returns:
            float: Average squared distance to nearest center
        """
        P = np.array(points)
        S = np.array(centers)

        distances = np.linalg.norm(P[:, np.newaxis] - S, axis=2)
        min_distances = np.min(distances, axis=1)
        result = np.sum(min_distances ** 2)

        return float(result) / len(P)

    def cost_omega(self, points, centers, weights):
        """
        Compute the weighted k-means cost of a set of points with respect to centers.
        The cost is the weighted average squared distance from each point to its nearest center.
        
        Args:
            points (numpy.ndarray): Set of points
            centers (numpy.ndarray): Set of centers
            weights (numpy.ndarray): Weights of the points
            
        Returns:
            float: Weighted average squared distance to nearest center
        """
        P = np.array(points)
        S = np.array(centers)
        W = np.array(weights)

        distances = np.linalg.norm(P[:, np.newaxis] - S, axis=2)
        min_distances = np.min(distances, axis=1)
        result = np.sum((min_distances ** 2) * W)

        return float(result) / sum(W)

    def distortion(self, input_data, omega, weights, centers):
        """
        Compute the distortion of a coreset.
        Distortion measures how well the coreset approximates the original dataset.
        It is defined as the maximum ratio between the costs of the original data
        and the coreset, or vice versa.
        
        Args:
            input_data (numpy.ndarray): Original input data
            omega (numpy.ndarray): Coreset points
            weights (numpy.ndarray): Weights of coreset points
            centers (numpy.ndarray): Set of centers
            
        Returns:
            float: Distortion value (≥ 1, where 1 is perfect approximation)
        """
        return max(self.cost(input_data, centers) / self.cost_omega(omega, centers, weights),
                   self.cost_omega(omega, centers, weights) / self.cost(input_data, centers))

    def get_v_vector(self, a, b, alpha, k):
        """
        Generate a v vector for benchmark construction.
        The v vector represents one column of the benchmark matrix.
        
        Args:
            a (int): Index in {0, ..., alpha - 1}
            b (int): Index in {1, ..., k}
            alpha (int): Parameter controlling number of points and dimensions
            k (int): Number of centers
            
        Returns:
            numpy.ndarray: The v vector
        """
        l = a + 1

        v = []
        for j in range(1, k + 1):
            if b != j:
                v.append(-1 / k)
            else:
                v.append((k-1) / k)

        for _ in range(l, alpha):
            v = np.kron(v, np.ones(k))

        ones_vec = np.ones(k ** a)
        result = np.kron(ones_vec, v)

        return list(reversed(result))

    def benchmark_construction(self, alpha, k):
        """
        Construct a benchmark matrix for testing clustering algorithms.
        The matrix is constructed using v vectors for each combination of a and b.
        
        Args:
            alpha (int): Parameter controlling number of points and dimensions
            k (int): Number of centers
            
        Returns:
            numpy.ndarray: Benchmark matrix
        """
        A = []

        for a in range(alpha):
            for b in range(1, k + 1):
                v = self.get_v_vector(a, b, alpha, k)
                A.append(np.array(v))

        return np.array(list(reversed(A))).T
