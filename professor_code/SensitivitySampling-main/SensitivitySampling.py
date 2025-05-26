"""
Sensitivity Sampling implementation for k-means clustering.
This module implements the basic sensitivity sampling algorithm that creates a weighted coreset
for k-means clustering problems.
"""

import hashlib
import numpy as np

from math import sqrt, log, ceil
from sklearn.cluster import kmeans_plusplus


class SensitivitySampling:
    """
    Implements the sensitivity sampling algorithm for k-means clustering.
    The algorithm creates a weighted coreset that approximates the original dataset
    while maintaining the quality of k-means clustering.
    """
    
    def __init__(self, P=None, k=0, m=0):
        """
        Initialize the SensitivitySampling class.
        
        Args:
            P (numpy.ndarray, optional): Input dataset
            k (int, optional): Number of clusters
            m (int, optional): Size of the coreset
        """
        self.P = P
        self.k = k
        self.m = m
        self.cost_cache = {}  # Cache for cost computations

    def normalize(self, v):
        """
        Normalize a vector to sum to 1.
        
        Args:
            v (numpy.ndarray): Input vector
            
        Returns:
            numpy.ndarray: Normalized vector
        """
        s = sum(v)
        if s == 0:
            return v
        return v / s

    def hash_key(self, P, S):
        """
        Generate a unique hash key for caching cost computations.
        
        Args:
            P (numpy.ndarray): Set of points
            S (numpy.ndarray): Set of centers
            
        Returns:
            str: Unique hash key
        """
        P = np.array(P)
        S = np.array(S)

        P_hash = hashlib.sha256(P.tobytes()).hexdigest()
        S_hash = hashlib.sha256(S.tobytes()).hexdigest()
        return f"{P_hash}-{S_hash}"

    def cached_cost(self, P, S):
        """
        Compute or retrieve cached cost between points and centers.
        
        Args:
            P (numpy.ndarray): Set of points
            S (numpy.ndarray): Set of centers
            
        Returns:
            float: Cached cost value
        """
        key = self.hash_key(P, S)
        if key not in self.cost_cache:
            self.cost_cache[key] = self.cost(P, S)
        return self.cost_cache[key]

    def cost(self, points, centers):
        """
        Compute the k-means cost between points and centers.
        
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

    def local_search_pp(self, P, centers):
        """
        Perform local search optimization on centers using k-means++ initialization.
        
        Args:
            P (numpy.ndarray): Set of points
            centers (numpy.ndarray): Current centers
            
        Returns:
            numpy.ndarray: Optimized centers
        """
        cost_pc = self.cached_cost(P, centers)
        cost_single_points = np.array([self.cached_cost([q], centers) for q in P])
        cost_sum = cost_single_points.sum()

        # (1) Compute sampling probabilities
        probs = cost_single_points / cost_sum

        # (2) Randomly sample a point p from P according to the probabilities
        pi = np.random.choice(range(len(P)), 1, p=probs)[0]
        p = P[pi]

        # (3) Get p_prime minimizing the cost
        cost_p_prime = float('inf')
        p_prime = []

        centers_array = np.array(centers)

        for q in centers:
            q_idx = np.where(centers_array == q)[0][0]
            new_centers = np.delete(centers_array, q_idx, axis=0)
            new_centers = np.vstack([new_centers, p])

            cost_q = self.cached_cost(P, new_centers)
            if cost_q < cost_p_prime:
                p_prime = q
                cost_p_prime = cost_q

        solutions = []
        p_prime_idx = np.where(centers == p_prime)[0][0]
        centers_array = np.array(centers)

        for p in P:
            new_centers = np.delete(centers_array, p_prime_idx, axis=0)
            new_centers = np.vstack([new_centers, p])

            cost_p_prime = self.cached_cost(P, new_centers)

            if cost_p_prime < cost_pc:
                solutions.append((p, cost_p_prime))

        if len(solutions) == 0:
            return centers

        min_solution = min(solutions, key=lambda x: x[1])

        index_to_delete = np.where((centers == p_prime).all(axis=1))[0][0]
        new_centers = np.delete(centers, index_to_delete, axis=0)
        new_centers = np.vstack([new_centers, min_solution[0]])

        return new_centers

    def kmeans_pp_local_search(self, P, k):
        """
        Find initial centers using k-means++ with local search optimization.
        
        Args:
            P (numpy.ndarray): Set of points
            k (int): Number of clusters
            
        Returns:
            numpy.ndarray: Optimized centers
        """
        centers, indices = kmeans_plusplus(P, n_clusters=k)

        z = ceil(k * log(log(k)))
        for i in range(z):
            centers = self.local_search_pp(P, centers)

        return centers

    def get_clusters(self, P, centers):
        """
        Assign points to their nearest centers to form clusters.
        
        Args:
            P (numpy.ndarray): Set of points
            centers (numpy.ndarray): Set of centers
            
        Returns:
            list: List of clusters, where each cluster is a list of points
        """
        P = np.array(P)
        centers = np.array(centers)

        distances = np.linalg.norm(P[:, np.newaxis] - centers, axis=2)
        nearest_center_indices = np.argmin(distances, axis=1)

        clusters = [[] for _ in range(len(centers))]

        for i, idx in enumerate(nearest_center_indices):
            clusters[idx].append(P[i])

        return clusters

    def get_clustering_solution(self, P, k):
        """
        Compute initial clustering solution using k-means++ with local search.
        
        Args:
            P (numpy.ndarray): Set of points
            k (int): Number of clusters
            
        Returns:
            tuple: (centers, clusters)
        """
        centers = self.kmeans_pp_local_search(P, k)
        clusters = self.get_clusters(P, centers)

        return centers, clusters

    def sensitivity_sampling(self, P, A, C, k, m):
        """
        Compute the sensitivity sampling algorithm to create a coreset.
        
        Args:
            P (numpy.ndarray): Set of all points
            A (numpy.ndarray): k cluster centers
            C (list): k clusters
            k (int): Number of clusters
            m (int): Size of the coreset
            
        Returns:
            tuple: (S, W) where S is the coreset and W are the weights
        """
        i = 0
        dist = dict()

        for cluster in C:
            cost_cluster_k = k * self.cached_cost(cluster, A)

            for point in cluster:
                cost_p = self.cached_cost([point], A)
                mu_p = 1 / (k * len(cluster)) + cost_p / cost_cluster_k
                dist[i] = mu_p
                i += 1

        p = np.array(list(map(float, dist.values())))
        p = self.normalize(p)

        idx = np.array(list(range(len(P))))
        S = np.random.choice(idx, m, replace=True, p=p)

        W = dict()
        for point in S:
            p = P[point]
            key = tuple(map(float, p))
            wp = 1 / (m * dist[point])
            W[key] = wp

        S = P[S]

        return S, [W[tuple(p)] for p in S]

    def run(self):
        """
        Run the complete sensitivity sampling algorithm.
        
        Returns:
            tuple: (S, W) where S is the coreset and W are the weights
        """
        centers, clusters = self.get_clustering_solution(self.P, self.k)
        return self.sensitivity_sampling(self.P, centers, clusters, self.k, self.m)
