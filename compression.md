# Compression in SensitivitySampling

This document explains in detail how the `compress` method works in the `SensitivitySampling` class (see `src/sensitivity_sampling.py`). It covers both the basic and upgraded sensitivity sampling strategies, the meaning of the parameter `t`, and the improvements made to the code for efficiency and correctness.

---

## Overview

The goal of the `compress` method is to create a **coreset**: a small, weighted subset of the original point cloud that preserves the essential structure for k-means clustering. This allows for faster and more memory-efficient clustering, while maintaining high accuracy.

The method supports two sampling strategies:
- **Basic Sensitivity Sampling** (`sample`)
- **Upgraded Sensitivity Sampling** (`sample2`)

You can choose which strategy to use by passing the appropriate function to the `compress` method.

---

## Basic Sensitivity Sampling (`sample`)

### How it works
1. **Clustering:**
   - If the point cloud does not already have cluster labels, k-means is run to assign each point to a cluster.
2. **Sensitivity Calculation:**
   - For each point, its sensitivity is calculated based on its distance to the cluster center and the size/cost of its cluster.
   - Sensitivity formula:
     \[
     \text{sensitivity}(p) = \frac{1}{k \cdot |C_j|} + \frac{\text{cost}(\{p\}, A)}{k \cdot \text{cost}(C_j, A)}
     \]
     where \( |C_j| \) is the size of the cluster containing \( p \), and cost is the squared distance.
3. **Sampling:**
   - Points are sampled with replacement according to their normalized sensitivities.
   - Sampling is done **by index** for efficiency and correctness.
4. **Weight Assignment:**
   - Each sampled point receives a weight: \( w_p = \frac{1}{m \cdot \text{sensitivity}(p)} \), where \( m \) is the sample size.
5. **Return:**
   - Returns the sampled indices and their weights.

### Example
Suppose you have 6 points and want a coreset of 3:
- Original indices: `[0, 1, 2, 3, 4, 5]`
- Sampled indices: `[1, 4, 1]` (index 1 appears twice)
- Weights: `[1.2, 0.8, 1.2]`

---

## Upgraded Sensitivity Sampling (`sample2`)

### What is `t`?
- `t` is the number of "cheapest" clusters (those with the lowest k-means cost) to be handled differently.
- These clusters are summarized by their centroid (mean), and the rest are sampled as in the basic method.

### How it works
1. **Clustering:**
   - As before, k-means assigns points to clusters.
2. **Cheapest Clusters:**
   - The `t` clusters with the lowest cost are identified.
   - Each is summarized by its mean, with a weight equal to the cluster size.
3. **Sampling from Remaining Clusters:**
   - The remaining clusters are flattened into a new point cloud.
   - Sensitivity sampling is applied to select the rest of the coreset points.
4. **Combining Results:**
   - The final coreset is the union of the centroids of the cheap clusters and the sampled points from the rest, with their respective weights.
5. **Return:**
   - Returns the sampled points, weights, and their classes.

### Example
Suppose you have 5 clusters and set `t=2`:
- The 2 cheapest clusters are summarized by their centroids (each centroid gets a weight equal to the size of its cluster).
- The remaining 3 clusters are sampled from using sensitivity sampling to fill the rest of the coreset.

---

## The `compress` Method

### Signature
```python
def compress(self, sample_size, sample_func=None, **kwargs):
```
- `sample_size`: Number of points in the coreset.
- `sample_func`: The sampling function to use (`self.sample` or `self.sample2`). Defaults to `self.sample`.
- `**kwargs`: Additional arguments for the sampling function (e.g., `t` for `sample2`).

### How it works
1. **Sampling:**
   - Calls the chosen sampling function to get either:
     - Sampled indices and weights (basic), or
     - Sampled points, weights, and classes (upgrade).
2. **Extracting Data:**
   - If indices are returned, uses them to extract coordinates and classes from the original point cloud.
   - If points and classes are returned, uses them directly.
3. **Constructing the Compressed Point Cloud:**
   - Builds a new `PointCloud` object from the sampled coordinates and classes.

### Example Usage
```python
# Basic compression
compressed_pc = sampler.compress(100)  # uses self.sample by default

# Upgraded compression (e.g., with t=2 cheapest clusters)
compressed_pc_upgrade = sampler.compress(100, sample_func=sampler.sample2, t=2)
```

---

## Improvements: Sampling by Index

- **Old approach:** Sampled by value, requiring matching coordinates to original data to get classes (slow and error-prone).
- **New approach:** Sample by index, allowing direct lookup of all attributes (fast and robust).
- **Result:** Handles duplicates naturally, avoids ambiguity, and is easier to maintain.

---

## Summary Table

| Approach         | Need to match by value? | Handles duplicates? | Fast? | Simple? |
|------------------|------------------------|---------------------|-------|---------|
| By value         | Yes                    | Yes (with care)     | No    | No      |
| By index         | No                     | Yes                 | Yes   | Yes     |

---

## Why is all this needed?

- To ensure that the compressed point cloud (coreset) accurately represents the original data, including all attributes (like class), even when points are sampled multiple times.
- To make the code efficient, robust, and easy to extend.

---

## References
- See `src/sensitivity_sampling.py` for implementation details.
- See chat history for practical examples and further discussion. 