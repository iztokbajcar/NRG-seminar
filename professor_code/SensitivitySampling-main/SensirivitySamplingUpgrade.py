"""
Upgraded Sensitivity Sampling implementation for k-means clustering.
This module extends the basic sensitivity sampling algorithm with an improved version
that handles cheap clusters differently to improve efficiency.
"""

import heapq
import numpy as np

from itertools import chain
from SensitivitySampling import SensitivitySampling


class SensirivitySamplingUpgrade(SensitivitySampling):
    """
    Implements an upgraded version of the sensitivity sampling algorithm.
    This version improves efficiency by handling cheap clusters differently.
    """
    
    def __init__(self, P, k, m):
        """
        Initialize the SensirivitySamplingUpgrade class.
        
        Args:
            P (numpy.ndarray): Input dataset
            k (int): Number of clusters
            m (int): Size of the coreset
        """
        super().__init__(P, k, m)

    def sensitivity_sampling_upgrade(self, P, centers, clusters, k, t):
        """
        Compute the upgraded sensitivity sampling algorithm.
        This version improves efficiency by handling cheap clusters differently.
        
        Args:
            P (numpy.ndarray): Set of all points
            centers (numpy.ndarray): k cluster centers
            clusters (list): k clusters
            k (int): Number of clusters
            t (int): Number of cheapest clusters to handle differently
            
        Returns:
            tuple: (S, W) where S is the coreset and W are the weights
        """
        # (2) Select the cheapest t clusters
        clusters = sorted(clusters, key=lambda x: self.cost(P, x))
        C_cheap = clusters[:t]  # t cheapest clusters
        not_C_cheap = clusters[t:]  # remaining clusters
        centers_not_cheap = centers[t:]  # centers for remaining clusters

        # Flatten not C cheap for processing
        P_not = list(chain(*not_C_cheap))

        # (3) Summarize each cheap cluster by its mean
        S = []
        W = []
        for cluster in C_cheap:
            mu = np.mean(cluster, axis=0)  # Compute cluster mean
            S.append(mu)
            wp = len(cluster)  # Weight is the size of the cluster
            W.append(wp)

        # (4) Apply sensitivity sampling on remaining clusters
        S_not, W_not = self.sensitivity_sampling(
            np.array(P_not), 
            centers_not_cheap, 
            not_C_cheap, 
            k - t,  # Number of remaining clusters
            self.m - t  # Remaining coreset size
        )

        # Combine results from cheap and non-cheap clusters
        return S + list(S_not), W + W_not
