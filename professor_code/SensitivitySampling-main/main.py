"""
Main script for evaluating and comparing sensitivity sampling algorithms.
This script implements the evaluation pipeline for both basic and upgraded
sensitivity sampling algorithms, including visualization and testing capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer, fetch_california_housing, \
    fetch_covtype

from Evaluation import Evaluation
from SensitivitySampling import SensitivitySampling
from SensirivitySamplingUpgrade import SensirivitySamplingUpgrade


def eval_pipeline(P, sens, sens_up, centers, clusters, k, m, t, split_factor, n=10):
    """
    Run the evaluation pipeline for both sensitivity sampling algorithms.
    
    Args:
        P (numpy.ndarray): Input dataset
        sens (SensitivitySampling): Basic sensitivity sampling instance
        sens_up (SensirivitySamplingUpgrade): Upgraded sensitivity sampling instance
        centers (numpy.ndarray): Initial cluster centers
        clusters (list): Initial clusters
        k (int): Number of clusters
        m (int): Size of the coreset
        t (int): Number of cheapest clusters for upgraded version
        split_factor (int): Factor for splitting data
        n (int, optional): Number of iterations for evaluation
        
    Returns:
        tuple: (s_mean, s_std, s_up_mean, s_up_std) - Mean and std of distortions
    """
    print(f"Evaluating for t = {t}...")
    S = []
    eval = Evaluation()

    # Evaluate basic sensitivity sampling
    for i in range(n):
        s, w, centers = sens.sensitivity_sampling(P, centers, clusters, k, m)
        distortion = eval.distortion(P, s, w, centers)
        S.append(distortion)

    # Evaluate upgraded sensitivity sampling
    S_up = []
    for i in range(n):
        s_up, w_up, centers = sens_up.sensitivity_sampling_upgrade(P, centers, clusters, k, t)
        distortion = eval.distortion(P, s_up, w_up, centers)
        S_up.append(distortion)

    S = np.array(S)
    S_up = np.array(S_up)

    # Compute statistics
    s_mean = np.mean(S)
    s_std = np.std(S)
    s_up_mean = np.mean(S_up)
    s_up_std = np.std(S_up)

    return s_mean, s_std, s_up_mean, s_up_std


def make_plot(P, centers, clusters, k, m, ts):
    """
    Create a plot comparing the performance of both algorithms.
    
    Args:
        P (numpy.ndarray): Input dataset
        centers (numpy.ndarray): Initial cluster centers
        clusters (list): Initial clusters
        k (int): Number of clusters
        m (int): Size of the coreset
        ts (list): List of t values to evaluate
    """
    sens_means = []
    sens_stds = []
    sens_up_means = []
    sens_up_stds = []

    sens = SensitivitySampling(P, k, m)
    sens_up = SensirivitySamplingUpgrade(P, k, m)

    # Evaluate for each t value
    for t in ts:
        s_mean, s_std, s_up_mean, s_up_std = eval_pipeline(P, sens, sens_up, centers, clusters, k, m, t)
        sens_means.append(s_mean)
        sens_stds.append(s_std)
        sens_up_means.append(s_up_mean)
        sens_up_stds.append(s_up_std)

    # Create and save plot
    plt.figure()
    plt.plot(ts, sens_means, label="Sensitivity Sampling")
    plt.plot(ts, sens_up_means, label="Sensitivity Sampling Upgrade")
    plt.xlabel("t")
    plt.ylabel("Distortion")
    plt.title(f"Distortion vs t for k = {k} and m = {m}")
    plt.legend()
    plt.savefig(f"plots/distortion_vs_t_k{k}_m{m}.png")


def split(arr, t):
    """
    Split a numpy array into multiple smaller arrays of size t.
    
    Args:
        arr (numpy.ndarray): Input array
        t (int): Size of each split
        
    Returns:
        list: List of split arrays
    """
    return [arr[i:i + t] for i in range(0, len(arr), t)]


def test_senisitivity_sampling_split(P, k, m, split_factor):
    """
    Test basic sensitivity sampling with data splitting.
    
    Args:
        P (numpy.ndarray): Input dataset
        k (int): Number of clusters
        m (int): Size of the coreset
        split_factor (int): Factor for splitting data
        
    Returns:
        tuple: (sol_s, sol_w, centers_s) - Solution coreset and centers
    """
    sens = SensitivitySampling(P, k, m)

    S = []
    W = []

    # Process each split
    for split_P in split(P, split_factor):
        print(f"Generating clusters for k={k} ...")
        centers, clusters = sens.get_clustering_solution(split_P, k)
        print(f"Clusters generated for k={k}.")

        s, w = sens.sensitivity_sampling(split_P, centers, clusters, k, m)
        S.append(s)
        W.append(w)

    S = np.vstack(S)
    centers_s, clusters_s = sens.get_clustering_solution(S, k)
    sol_s, sol_w = sens.sensitivity_sampling(S, centers_s, clusters_s, k, m)

    return sol_s, sol_w, centers_s


def process_split(split_P, k, sens_up, t):
    """
    Process a single split of data for parallel processing.
    
    Args:
        split_P (numpy.ndarray): Split of input data
        k (int): Number of clusters
        sens_up (SensirivitySamplingUpgrade): Upgraded sensitivity sampling instance
        t (int): Number of cheapest clusters
        
    Returns:
        tuple: (s, w) - Coreset and weights for the split
    """
    centers, clusters = sens_up.get_clustering_solution(split_P, k)
    s, w = sens_up.sensitivity_sampling_upgrade(split_P, centers, clusters, k, t)
    return s, w


def test_senisitivity_sampling_upgrade_split(P, k, m, t, split_factor):
    """
    Test upgraded sensitivity sampling with parallel processing of splits.
    
    Args:
        P (numpy.ndarray): Input dataset
        k (int): Number of clusters
        m (int): Size of the coreset
        t (int): Number of cheapest clusters
        split_factor (int): Factor for splitting data
        
    Returns:
        tuple: (sol_s, sol_w, centers_s) - Solution coreset and centers
    """
    sens_up = SensirivitySamplingUpgrade(P, k, m)

    S = []
    W = []

    print("Splitting data...")
    splits = list(split(P, split_factor))
    print("Done.")

    # Process splits in parallel
    print("Processing splits...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        i = 1
        futures = [executor.submit(process_split, split_P, k, sens_up, t) for split_P in splits]
        for future in concurrent.futures.as_completed(futures):
            s, w = future.result()
            print(f"Future {i} done: {len(s)}")
            S.append(s)
            W.append(w)
            i += 1

    S = np.vstack(S)

    print("Generating clusters for solution...")
    centers_s, clusters_s = sens_up.get_clustering_solution(S, k)
    print("Done.")

    sol_s, sol_w = sens_up.sensitivity_sampling_upgrade(S, centers_s, clusters_s, k, t)

    return sol_s, sol_w, centers_s


def main():
    """
    Main function to run the evaluation pipeline.
    """
    eval = Evaluation()
    cov_type = fetch_covtype()
    P = cov_type.data

    k = 7
    t = 6
    split_factor = 1000
    m = 40 * k
    
    # Run the upgraded sensitivity sampling with split processing
    S, W, centers = test_senisitivity_sampling_upgrade_split(P, k, m, t, split_factor)
    distortion = eval.distortion(P, S, W, centers)
    print(distortion)


if __name__ == '__main__':
    main()
